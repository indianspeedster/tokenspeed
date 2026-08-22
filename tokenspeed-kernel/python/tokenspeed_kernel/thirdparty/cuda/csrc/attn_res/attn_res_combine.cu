/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Single-token split-CTA AttnRes partial-state combine for Kimi-K3.  The
 * cooperative cluster reduction follows attn_res_fwd_s1_splitk_kernel.
 */

#include "attn_res.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace {

template <int GROUPS, bool OUT_NORM>
__global__ void __launch_bounds__(256, 1) attn_res_combine_s1_kernel(
    const bf16_t* __restrict__ prefix, const bf16_t* __restrict__ wp,
    const bf16_t* __restrict__ out_norm_weight,
    const float* __restrict__ partial_m, const float* __restrict__ partial_s,
    const float* __restrict__ partial_acc, bf16_t* __restrict__ output,
    float rms_eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  namespace cg = cooperative_groups;
  constexpr int H = 7168;
  constexpr int THREADS = 256;
  constexpr int WARPS = THREADS / 32;
  constexpr int K_PER_CTA = H / GROUPS;
  constexpr float LOG2_E = 1.4426950408889634f;
  static_assert(H % GROUPS == 0);

  __shared__ float2 warp_stats[WARPS];
  __shared__ float scalars[4];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  cg::cluster_group cluster = cg::this_cluster();
  const int group = cluster.block_rank();
  const int h_begin = group * K_PER_CTA;

  float sq = 0.0f;
  float dot = 0.0f;
  for (int ki = tid; ki < K_PER_CTA; ki += THREADS) {
    const int h = h_begin + ki;
    const float v = __bfloat162float(prefix[h]);
    sq = fmaf(v, v, sq);
    dot = fmaf(v, __bfloat162float(wp[h]), dot);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sq += __shfl_down_sync(0xffffffff, sq, offset);
    dot += __shfl_down_sync(0xffffffff, dot, offset);
  }
  if (lane == 0) warp_stats[warp] = make_float2(sq, dot);
  __syncthreads();
  if (tid == 0) {
    float2 total = {};
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      total.x += warp_stats[w].x;
      total.y += warp_stats[w].y;
    }
    warp_stats[0] = total;
  }

  cluster.sync();
  if (tid == 0) {
    float2 total = {};
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
      const float2 remote = *cluster.map_shared_rank(warp_stats, g);
      total.x += remote.x;
      total.y += remote.y;
    }
    const float logit = total.y * rsqrtf(total.x / H + rms_eps);
    const float m_b = partial_m[0];
    const float m = fmaxf(m_b, logit);
    const float corr = exp2f((m_b - m) * LOG2_E);
    const float w_prefix = exp2f((logit - m) * LOG2_E);
    scalars[0] = corr;
    scalars[1] = w_prefix;
    scalars[2] = 1.0f / (partial_s[0] * corr + w_prefix);
  }
  __syncthreads();

  float mix_sq = 0.0f;
  for (int ki = tid; ki < K_PER_CTA; ki += THREADS) {
    const int h = h_begin + ki;
    const float v = __bfloat162float(prefix[h]);
    const float mix =
        (partial_acc[h] * scalars[0] + scalars[1] * v) * scalars[2];
    const bf16_t rounded_mix = __float2bfloat16_rn(mix);
    output[h] = rounded_mix;
    const float rounded = __bfloat162float(rounded_mix);
    mix_sq = fmaf(rounded, rounded, mix_sq);
  }
  if constexpr (!OUT_NORM) return;

  for (int offset = 16; offset > 0; offset >>= 1) {
    mix_sq += __shfl_down_sync(0xffffffff, mix_sq, offset);
  }
  if (lane == 0) warp_stats[warp].x = mix_sq;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += warp_stats[w].x;
    warp_stats[0].x = total;
  }

  cluster.sync();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
      total += cluster.map_shared_rank(warp_stats, g)->x;
    }
    scalars[3] = rsqrtf(total / H + rms_eps);
  }
  __syncthreads();
  for (int ki = tid; ki < K_PER_CTA; ki += THREADS) {
    const int h = h_begin + ki;
    const float mix = __bfloat162float(output[h]);
    output[h] = __float2bfloat16_rn(
        mix * scalars[3] * __bfloat162float(out_norm_weight[h]));
  }
#endif
}

template <int GROUPS, bool OUT_NORM>
void launch_combine(const bf16_t* prefix, const bf16_t* wp,
                    const bf16_t* out_norm_weight, const float* partial_m,
                    const float* partial_s, const float* partial_acc,
                    bf16_t* output, float rms_eps, cudaStream_t stream) {
  auto kernel = &attn_res_combine_s1_kernel<GROUPS, OUT_NORM>;
  void* args[] = {const_cast<bf16_t**>(&prefix),
                  const_cast<bf16_t**>(&wp),
                  const_cast<bf16_t**>(&out_norm_weight),
                  const_cast<float**>(&partial_m),
                  const_cast<float**>(&partial_s),
                  const_cast<float**>(&partial_acc), &output, &rms_eps};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(GROUPS);
  config.blockDim = dim3(256);
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = GROUPS;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), args);
}

template <int GROUPS>
void launch_combine_norm_dispatch(
    const bf16_t* prefix, const bf16_t* wp,
    const bf16_t* out_norm_weight, const float* partial_m,
    const float* partial_s, const float* partial_acc, bf16_t* output,
    float rms_eps, cudaStream_t stream) {
  if (out_norm_weight) {
    launch_combine<GROUPS, true>(prefix, wp, out_norm_weight, partial_m,
                                 partial_s, partial_acc, output, rms_eps, stream);
  } else {
    launch_combine<GROUPS, false>(prefix, wp, out_norm_weight, partial_m,
                                  partial_s, partial_acc, output, rms_eps, stream);
  }
}

}  // namespace

void run_attn_res_combine(
    const bf16_t* prefix, const bf16_t* wp,
    const bf16_t* out_norm_weight, const float* partial_m,
    const float* partial_s, const float* partial_acc, bf16_t* output,
    float rms_eps, int groups, cudaStream_t stream) {
  switch (groups) {
    case 1:
      return launch_combine_norm_dispatch<1>(prefix, wp, out_norm_weight,
                                              partial_m, partial_s, partial_acc,
                                              output, rms_eps, stream);
    case 2:
      return launch_combine_norm_dispatch<2>(prefix, wp, out_norm_weight,
                                              partial_m, partial_s, partial_acc,
                                              output, rms_eps, stream);
    case 4:
      return launch_combine_norm_dispatch<4>(prefix, wp, out_norm_weight,
                                              partial_m, partial_s, partial_acc,
                                              output, rms_eps, stream);
    case 7:
      return launch_combine_norm_dispatch<7>(prefix, wp, out_norm_weight,
                                              partial_m, partial_s, partial_acc,
                                              output, rms_eps, stream);
    case 8:
      return launch_combine_norm_dispatch<8>(prefix, wp, out_norm_weight,
                                              partial_m, partial_s, partial_acc,
                                              output, rms_eps, stream);
  }
}
