# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Cold-L2 CUDA-graph benchmark and CTA-group sweep for AttnRes combine.

Run on a Blackwell GPU after building the CUDA extension::

    python -m tokenspeed_kernel.benchmark.attnres_combine

Each sample evicts L2 before timing one replay. Inputs rotate across at least
eight independently captured graphs. Reported values are median microseconds.
"""

import argparse
import statistics

import torch
from tokenspeed_kernel.ops.activation.triton import attnres_combine
from tokenspeed_kernel.thirdparty.cuda.attn_res import attn_res_combine as combine_cuda

H = 7168
TRITON_STANDALONE_BAR_US = 3.07


def _capture(call):
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def _measure(graphs, eviction, samples):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    stream = torch.cuda.current_stream()
    for i in range(samples):
        eviction.zero_()
        starts[i].record(stream)
        graphs[i % len(graphs)].replay()
        ends[i].record(stream)
    ends[-1].synchronize()
    return statistics.median(start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--evict-mib", type=int, default=256)
    parser.add_argument("--no-out-norm", action="store_true")
    args = parser.parse_args()
    if args.copies < 8:
        parser.error("--copies must be at least 8")

    torch.manual_seed(7)
    eps = 1e-5
    prefixes = torch.randn(args.copies, 1, H, device="cuda", dtype=torch.bfloat16)
    wps = torch.randn(args.copies, H, device="cuda", dtype=torch.bfloat16) * 0.01
    ms = torch.randn(args.copies, 1, device="cuda")
    ss = torch.rand(args.copies, 1, device="cuda") + 0.5
    accs = torch.randn(args.copies, 1, H, device="cuda")
    outputs = torch.empty_like(prefixes)
    out_w = None
    if not args.no_out_norm:
        out_w = torch.rand(H, device="cuda", dtype=torch.bfloat16) + 0.5
    eviction = torch.empty(args.evict_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)

    def call(i, groups=None):
        scratch = (ms[i], ss[i], accs[i])
        if groups is None:
            return lambda: attnres_combine(prefixes[i], wps[i], out_w, eps, scratch, outputs[i])
        return lambda: combine_cuda(
            prefixes[i], wps[i], out_w, eps, scratch, outputs[i], groups=groups
        )

    triton_graphs = [_capture(call(i)) for i in range(args.copies)]
    triton_us = _measure(triton_graphs, eviction, args.samples)
    print(f"triton: {triton_us:.3f} us")
    results = []
    for groups in (1, 2, 4, 7, 8):
        graphs = [_capture(call(i, groups)) for i in range(args.copies)]
        elapsed = _measure(graphs, eviction, args.samples)
        results.append((elapsed, groups))
        print(f"cuda groups={groups}: {elapsed:.3f} us")
    best_us, best_groups = min(results)
    print(
        f"best: groups={best_groups}, {best_us:.3f} us "
        f"({triton_us / best_us:.3f}x vs Triton)"
    )
    if best_us >= TRITON_STANDALONE_BAR_US:
        print(
            f"STOP: best CUDA result did not clear the "
            f"{TRITON_STANDALONE_BAR_US:.2f} us standalone bar"
        )


if __name__ == "__main__":
    main()
