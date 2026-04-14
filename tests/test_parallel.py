
# SPDX-License-Identifier: Apache-2.0

"""Parameterized tests for PipelineParallel.iterate_weights() interface.

Tests verify that tensors loaded via iterate_weights() match those loaded
directly via safetensors.load_file(). Backend (nogds, 3fs, ...) is
parameterized so every test automatically runs against all available backends.
"""

import os

import pytest

from fastsafetensors import ParallelLoader, SingleGroup
from fastsafetensors import cpp as fstcpp
from fastsafetensors.common import is_gpu_found

# ---------------------------------------------------------------------------
# 3FS availability detection
# ---------------------------------------------------------------------------
try:
    import fastsafetensor_3fs_reader  # noqa: F401

    _HAS_3FS = True
except ImportError:
    _HAS_3FS = False


# ---------------------------------------------------------------------------
# Backend factory functions
# ---------------------------------------------------------------------------
def _create_nogds_loader(pg, files, device, framework_name):
    return ParallelLoader(
        pg=pg,
        hf_weights_files=files,
        device=device,
        nogds=True,
        debug_log=True,
        framework=framework_name,
    )


def _create_3fs_loader(pg, files, device, framework_name):
    from fastsafetensors.threefs_loader import ParallelThreeFSLoader

    return ParallelThreeFSLoader(
        pg=pg,
        hf_weights_files=files,
        device=device,
        debug_log=True,
        framework=framework_name,
    )


# ---------------------------------------------------------------------------
# Parameterized backends
# ---------------------------------------------------------------------------
_BACKENDS = [
    pytest.param(("nogds", _create_nogds_loader), id="nogds"),
    pytest.param(
        ("3fs", _create_3fs_loader),
        id="3fs",
        marks=pytest.mark.skipif(
            not _HAS_3FS,
            reason="fastsafetensor_3fs_reader not available",
        ),
    ),
]


@pytest.fixture(params=_BACKENDS)
def loader_factory(request):
    """Yield a factory function that creates a parallel loader for the
    parameterized backend."""
    _name, factory = request.param
    return factory


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _get_device(framework):
    if is_gpu_found():
        if framework.get_name() == "pytorch":
            return "cuda:0"
        elif framework.get_name() == "paddle":
            return "gpu:0"
    return "cpu"


def _load_expected(files, device, framework):
    """Load all files via safetensors.load_file as ground truth."""
    if framework.get_name() == "pytorch":
        from safetensors.torch import load_file
    elif framework.get_name() == "paddle":
        from safetensors.paddle import load_file
    else:
        raise Exception(f"unknown framework: {framework.get_name()}")
    all_tensors = {}
    for f in files:
        all_tensors.update(load_file(f, device))
    return all_tensors


def _tensors_equal(a, b, framework_name):
    """Compare two raw tensors for exact equality."""
    if framework_name == "pytorch":
        import torch
        return bool(torch.all(a.eq(b)))
    elif framework_name == "paddle":
        import paddle
        return bool(paddle.all(a == b))
    raise Exception(f"unknown framework: {framework_name}")


def _verify_iterate_weights(loader, expected, framework):
    """Core assertion: iterate_weights output must match *expected* exactly.

    Note: iterate_weights() yields raw tensors whose underlying memory may be
    freed once the next batch is consumed (fb.close() in _consume_single_batch).
    Therefore we must verify each tensor **immediately** when it is yielded,
    rather than collecting all tensors first and comparing afterwards.
    """
    fw_name = framework.get_name()
    seen_keys = set()
    for key, tensor in loader.iterate_weights():
        assert key not in seen_keys, f"Duplicate key yielded: {key}"
        seen_keys.add(key)
        assert key in expected, f"Unexpected key yielded: {key}"
        exp = expected[key]
        assert list(tensor.shape) == list(exp.shape), f"Shape mismatch: {key}"
        assert tensor.dtype == exp.dtype, f"Dtype mismatch: {key}"
        assert _tensors_equal(tensor, exp, fw_name), f"Value mismatch: {key}"

    assert seen_keys == set(expected.keys()), (
        f"Key mismatch: "
        f"extra={seen_keys - set(expected.keys())}, "
        f"missing={set(expected.keys()) - seen_keys}"
    )


# ---------------------------------------------------------------------------
# Tests — each runs against every available backend via loader_factory
# ---------------------------------------------------------------------------
def test_single_file(fstcpp_log, input_files, framework, loader_factory):
    """Single file: iterate_weights tensors == safetensors.load_file."""
    device = _get_device(framework)
    files = [input_files[0]]

    expected = _load_expected(files, device, framework)
    loader = loader_factory(SingleGroup(), files, device, framework.get_name())
    _verify_iterate_weights(loader, expected, framework)

    loader.close()
    assert framework.get_mem_used() == 0


def test_multiple_files(fstcpp_log, input_files, framework, loader_factory):
    """Multiple files: merged iterate_weights tensors == merged load_file."""
    device = _get_device(framework)

    expected = _load_expected(input_files, device, framework)
    loader = loader_factory(
        SingleGroup(), input_files, device, framework.get_name()
    )
    _verify_iterate_weights(loader, expected, framework)

    loader.close()
    assert framework.get_mem_used() == 0


def test_memory_release(fstcpp_log, input_files, framework, loader_factory):
    """After full iteration + close, device memory must be fully released."""
    device = _get_device(framework)
    loader = loader_factory(
        SingleGroup(), input_files, device, framework.get_name()
    )

    count = 0
    for key, tensor in loader.iterate_weights():
        assert tensor is not None
        count += 1
    assert count > 0

    loader.close()
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_close_without_iterate(fstcpp_log, input_files, framework, loader_factory):
    """Close without iterating must not leak memory."""
    device = _get_device(framework)
    loader = loader_factory(
        SingleGroup(), input_files, device, framework.get_name()
    )

    loader.close()
    assert framework.get_mem_used() == 0


def test_distributed(fstcpp_log, input_files, pg, framework, loader_factory):
    """Distributed iterate_weights: each rank gets correct tensors."""
    world_size = (
        pg.size()
        if framework.get_name() == "pytorch"
        else pg.process_group.size()
    )
    if world_size <= 1:
        pytest.skip("Requires WORLD_SIZE > 1")

    device = _get_device(framework)
    loader = loader_factory(pg, input_files, device, framework.get_name())

    loaded = {}
    for key, tensor in loader.iterate_weights():
        loaded[key] = tensor

    # In distributed mode every rank should receive tensors
    assert len(loaded) > 0, "No tensors received in distributed mode"

    loader.close()
    assert framework.get_mem_used() == 0


# ---------------------------------------------------------------------------
# Entry point for torchrun / paddle.distributed.launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    os.environ["PADDLE_DISTRI_BACKEND"] = "gloo"
    sys.exit(pytest.main(sys.argv[1:]))
