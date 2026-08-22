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

"""Numerics and graph-capture tests for the split-CTA AttnRes combine."""

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import attnres_combine
from tokenspeed_kernel.thirdparty.cuda.attn_res import (
    attn_res_combine as attn_res_combine_cuda,
)
from tokenspeed_kernel.thirdparty.cuda.attn_res import has_attn_res_combine

H = 7168
EPS = 1e-5


def _require_kernel() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("Blackwell (SM100+) required")
    if not has_attn_res_combine():
        pytest.skip("attn_res extension was built without the combine kernel")


def _case(dominance: str):
    torch.manual_seed({"random": 1, "prefix": 2, "block": 3}[dominance])
    prefix = torch.randn(1, H, device="cuda", dtype=torch.bfloat16)
    wp = (torch.randn(H, device="cuda") * 0.01).to(torch.bfloat16)
    m = torch.randn(1, device="cuda")
    s = torch.rand(1, device="cuda") + 0.5
    acc = torch.randn(1, H, device="cuda")
    if dominance == "prefix":
        m.fill_(-100.0)
    elif dominance == "block":
        m.fill_(100.0)
    return prefix, wp, (m, s, acc)


@pytest.mark.parametrize("out_norm", [False, True])
@pytest.mark.parametrize("dominance", ["random", "prefix", "block"])
@pytest.mark.parametrize("groups", [1, 2, 4, 7, 8])
def test_attnres_combine_cuda_matches_triton(out_norm, dominance, groups):
    _require_kernel()
    prefix, wp, scratch = _case(dominance)
    out_w = (
        (torch.rand(H, device="cuda", dtype=torch.bfloat16) + 0.5)
        if out_norm
        else None
    )
    expected = torch.empty_like(prefix)
    actual = torch.empty_like(prefix)
    attnres_combine(prefix, wp, out_w, EPS, scratch, expected)
    attn_res_combine_cuda(prefix, wp, out_w, EPS, scratch, actual, groups=groups)
    torch.testing.assert_close(actual, expected, atol=0.5, rtol=2e-2)


def test_attnres_combine_cuda_graph_capture():
    _require_kernel()
    prefix, wp, scratch = _case("random")
    out_w = torch.rand(H, device="cuda", dtype=torch.bfloat16) + 0.5
    output = torch.empty_like(prefix)
    attn_res_combine_cuda(prefix, wp, out_w, EPS, scratch, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attn_res_combine_cuda(prefix, wp, out_w, EPS, scratch, output)
    graph.replay()
    torch.cuda.synchronize()
