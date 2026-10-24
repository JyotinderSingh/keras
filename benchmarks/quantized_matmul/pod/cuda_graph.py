"""The CUDA-graph capture of the pod scripts. Fork only; not part of a PR.

rows.py, tune.py and the probes capture through `capture`. Import it
only with the torch backend (place this file at
benchmarks/quantized_matmul/pod/cuda_graph.py).

The cause of the CUDA launch failures of S1r job 1, S1x and SX
(logs/sx-20261024-1/REPORT.md), in torch 2.14.1:
- The old captures made a new `torch.cuda.Stream()` for each warm-up.
  torch hands out the streams of a pool in turn, 32 per pool, and the
  default capture stream of `torch.cuda.graph` is one of them. So every
  32nd warm-up ran on the capture stream, outside a capture. The cuBLAS
  workspace of that warm-up (one per handle and stream, from the caching
  allocator) went into the next graph, and the reset of any other graph
  of that stream freed it (`CUDAGraph::reset` ->
  `clearCublasWorkspacesForStream`). A graph that lived on then replayed
  on freed memory.
- `torch.cuda.graph.__exit__` does not restore the stream when
  `capture_end` raises. After a failed capture the capture stream stayed
  the current stream, and each later eager cuBLAS call used its
  workspace.

`capture` therefore:
- warms up on one warm-up stream and captures on one capture stream.
  The process makes the two streams once, and they are never the same
  stream;
- saves the current stream before the capture and restores it in a
  `finally` block, also when the capture fails;
- clears the cuBLAS workspaces before the capture
  (`torch._C._cuda_clearCublasWorkspaces`, as torch inductor does), so
  that a graph allocates its workspace in its own memory pool;
- after the capture, replays the graph once and compares its outputs
  with an eager call of the same function on the same inputs, bit for
  bit. The caller reports a difference as a failed check.
"""

import torch

# Device index -> (the warm-up stream, the capture stream).
_STREAMS = {}


def streams():
    """`(warm-up stream, capture stream)` of the current device, made
    once per process."""
    device = torch.cuda.current_device()
    if device not in _STREAMS:
        warmup = torch.cuda.Stream()
        capture_stream = torch.cuda.Stream()
        if warmup == capture_stream:
            raise RuntimeError("the warm-up stream is the capture stream")
        _STREAMS[device] = (warmup, capture_stream)
    return _STREAMS[device]


def _leaves(outputs):
    if isinstance(outputs, (list, tuple)):
        return [leaf for item in outputs for leaf in _leaves(item)]
    if isinstance(outputs, dict):
        return _leaves([outputs[key] for key in sorted(outputs)])
    return [outputs]


def _bits(tensor):
    return tensor.detach().reshape(-1).contiguous().view(torch.uint8)


def same_bits(a, b):
    """Whether the tensors of `a` and `b` have the same shapes, dtypes and
    bits (a NaN equals the same NaN)."""
    a, b = _leaves(a), _leaves(b)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.shape != y.shape or x.dtype != y.dtype:
                return False
            if not torch.equal(_bits(x), _bits(y)):
                return False
        elif isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            return False
        elif x != y:
            return False
    return True


def capture(fn, warmup=3, graph=None):
    """A CUDA graph of `fn()`, after `warmup` calls of `fn` on the warm-up
    stream.

    Returns `(graph, outputs, same)`. `outputs` is what the captured call
    returned; each replay writes it again. `same` is `True` when the first
    replay gives the bits of an eager call of `fn` on the current stream.
    `graph` can be a `torch.cuda.CUDAGraph` that the caller made (for
    example with the debug mode on). A failed capture raises its error,
    with the current stream restored.
    """
    warm, capture_stream = streams()
    previous = torch.cuda.current_stream()
    if previous == capture_stream:
        raise RuntimeError("the capture stream is the current stream")
    graph = torch.cuda.CUDAGraph() if graph is None else graph
    try:
        warm.wait_stream(previous)
        with torch.cuda.stream(warm):
            for _ in range(warmup):
                fn()
        previous.wait_stream(warm)
        torch.cuda.synchronize()
        clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear is not None:
            clear()
        with torch.cuda.graph(graph, stream=capture_stream):
            outputs = fn()
    finally:
        torch.cuda.set_stream(previous)
    eager = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs, same_bits(outputs, eager)
