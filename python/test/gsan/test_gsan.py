from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_cuda
from triton.experimental.gsan import create_mem_pool
from triton._C.libtriton.gsan_testing import ScalarClock, AtomicScope
from triton.experimental.gsan._testing_utils import (load_one_i32, shadow_cell_from_address, store_one_i32,
                                                     thread_state_from_smid)

from triton.experimental import gluon
import triton.experimental.gluon.language as gl


@pytest.fixture()
def with_gsan(fresh_knobs):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        yield


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_load_store_updates_shadow(with_gsan):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")

    store_one_i32[(1, )](target, num_warps=1)
    cell0 = shadow_cell_from_address(target.data_ptr())

    tid = cell0.write_clock.thread_id
    epoch0 = thread_state_from_smid(tid).vector_clock[tid]

    assert cell0.write_clock.thread_id == tid
    assert cell0.write_clock.epoch == epoch0
    assert cell0.read_clocks[0].thread_id == 0
    assert cell0.read_clocks[0].epoch == 0
    assert cell0.num_reads == 0

    load_one_i32[(1, )](target, scratch, num_warps=1)
    cell1 = shadow_cell_from_address(target.data_ptr())
    epoch1 = thread_state_from_smid(tid).vector_clock[tid]

    assert epoch1 == epoch0 + 1
    assert cell1.write_clock == cell0.write_clock
    assert cell1.read_clocks[0] == ScalarClock(epoch1, tid, AtomicScope.NON_ATOMIC)
    # Scalar accesses are instrumented once via the redundant-thread predicate.
    assert cell1.num_reads == 1


@gluon.jit
def _gluon_ws_completion_default(out_ptr, layout: gl.constexpr):
    offsets = gl.arange(0, 128, layout=layout)
    gl.store(out_ptr + offsets, offsets)


@gluon.jit
def _gluon_ws_completion_worker(out_ptr, layout: gl.constexpr):
    offsets = 128 + gl.arange(0, 128, layout=layout)
    gl.store(out_ptr + offsets, offsets)


@gluon.jit
def _gluon_ws_completion_kernel(out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    gl.warp_specialize([
        (_gluon_ws_completion_default, (out_ptr, layout)),
        (_gluon_ws_completion_worker, (out_ptr, layout)),
    ], [4], [24])


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_gluon_warp_specialize_completes(with_gsan):
    expected = torch.arange(256, dtype=torch.int32, device="cuda")

    out = torch.full((256, ), -1, dtype=torch.int32, device="cuda")
    _gluon_ws_completion_kernel[(1, )](out, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected)


@triton.jit
def _write_blocks_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(ptr + offsets, 1, mask=mask)


@triton.jit
def _read_reversed_blocks_kernel(ptr, scratch_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    src_pid = tl.num_programs(0) - 1 - pid
    src_offsets = src_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dst_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_offsets < n_elements
    value = tl.load(ptr + src_offsets, mask=mask)
    tl.store(scratch_ptr + dst_offsets, value, mask=mask)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_implicit_stream_ordering(with_gsan, capfd):
    block_size = 128
    size = block_size * 1024
    target = torch.zeros(size, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(size, dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(size, block_size), )
    _write_blocks_kernel[grid](target, size, BLOCK_SIZE=block_size)
    _read_reversed_blocks_kernel[grid](target, scratch, size, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    assert scratch.sum().item() == size
    assert "GSanLibrary.cu" not in capfd.readouterr()
