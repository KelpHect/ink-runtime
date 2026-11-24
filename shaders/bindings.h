#pragma once
/*
 * shaders/bindings.h — HOST-SIDE CONTRACT for the Vulkan compute kernels in
 * this directory. GLSL 460, compiled with:
 *   glslc -O --target-env=vulkan1.4 -o <name>.spv <name>.comp
 *
 * GLOBAL RULES (boring on purpose):
 *   - Every shader uses descriptor set 0 only; binding numbers are per-shader
 *     and listed below in order.
 *   - All buffers are std430 storage buffers. "f32[n]" = float array,
 *     "u32[n]" = packed sub-word data (bf16 = 2/u16 lanes, u8 = 4/u8 lanes,
 *     NVFP4 = 2 E2M1 nibbles per byte INSIDE the packed u8 stream,
 *     low nibble = even element index).
 *   - Packed rows are read via u32 words; shapes guarantee each ROW starts on
 *     a 4-byte boundary (requirements noted per shader; all model K values
 *     satisfy them: 2304, 1152, 1024, 128, 16).
 *   - Push constants are plain 4-byte scalars, fields in listed order; sizes
 *     <= 32 bytes (well under the 128-byte guarantee). Struct layouts below
 *     are the exact push-constant blocks (std430 scalar layout, no padding).
 *   - All math is scalar fp32; no subgroup/cooperative-matrix/fp8 hardware
 *     ops are required. Only core Vulkan 1.0 compute features are needed.
 *   - Workgroup size is 64 (one wave64) unless noted. Dispatch dims given per
 *     shader; any grid.x > 65535 must be chunked via the m0/base PC field.
 *   - "rw" buffers are read+written by the shader; "ro" readonly, "wo" write.
 *
 * Model constants (from src/io/config.h) referenced below:
 *   H=2304 hidden, HQ=24 q-heads, HK=8 kv-heads, DH=128 head_dim, DR=16 d_rel,
 *   EXT=1024 rel_extent, E=128 experts, K_TOP=8, I=1152 intermediate,
 *   GCAP=4.0 (mlp_gate_softcap), UCAP=25.0 (mlp_up_softcap), ROUTE_SCALE=8.0,
 *   VOCAB_UNPAD=200058, eps=1e-6.
 */

#include <cstdint>

namespace ink::shaders {

/* flag bits shared across shaders */
constexpr uint32_t F_ACCUMULATE = 1u << 0;  /* out += wscale*res (matvec_*)   */
constexpr uint32_t F_UPDATE     = 1u << 0;  /* shift hist in-place (conv4_*)  */
constexpr uint32_t F_USE_GAMMA  = 1u << 0;  /* weighted rmsnorm               */
constexpr uint32_t F_USE_SCALE  = 1u << 0;  /* scaled_l2_clip                 */
constexpr uint32_t F_SRC_BF16   = 1u << 0;  /* embed row dtype (embed_norm)   */

/* ------------------------------------------------------------------ */
/* rmsnorm.comp   local_size_x=64                                       */
/* grid.x = n_rows. One vector per workgroup.                           */
/*   b0 ro f32 X[n_rows*n]   b1 ro f32 GAMMA[n] (dummy ok if unweighted)*/
/*   b2 wo f32 OUT[n_rows*n]                                          */
struct PcRmsnorm { uint32_t n; float eps; uint32_t flags; };

/* ------------------------------------------------------------------ */
/* matvec_f32.comp / matvec_bf16.comp / matvec_f8e4m3.comp              */
/*   local_size_x=64, grid.x = M (output rows; row = m0 + wgid.x).      */
/*   b0 ro W  (f32[M,K] | u32 bf16-packed[M*K/2] | u32 u8-packed[M*K/4])*/
/*   b1 ro f32 X[K]     b2 wo f32 OUT[M]                              */
/* bf16 requires K even; f8e4m3 requires K%4==0 (u32 row alignment).    */
struct PcMatvec { uint32_t K; uint32_t m0; };

/* ------------------------------------------------------------------ */
/* matvec_nvfp4.comp  local_size_x=64, grid.x = M                       */
/*   Expert GEMV: row = row0 + m*row_stride (w13 de-interleave:         */
/*   gate rows -> row0=0,stride=2 ; up rows -> row0=1,stride=2 ;        */
/*   w2 -> row0=0,stride=1). Result *= amax[e]/osd[0], then *wscale.    */
/*   flags F_ACCUMULATE adds into OUT instead of overwriting.           */
/*   Requires K%64==0. Shapes: w13 [E,2304,K=2304], w2 [E,2304,K=1152]. */
/*   b0 ro u32 W[E*R*K/8]   b1 ro u32 S[E*R*K/64]                       */
/*   b2 ro f32 AMAX[E]      b3 ro f32 OSD[1]                          */
/*   b4 ro f32 X[K]         b5 rw f32 OUT[M]                          */
struct PcNvfp4 {
    uint32_t e, R, K, row0, row_stride;
    float    wscale;
    uint32_t flags;
};

/* ------------------------------------------------------------------ */
/* matvec_mxfp8.comp  local_size_x=64, grid.x = M                       */
/*   Same contract minus amax/osd: F8E4M3 bytes + UE8M0 (2^(s-127))     */
/*   per 32 elems. Requires K%128==0.                                   */
/*   b0 ro u32 W[E*R*K/4]   b1 ro u32 S[E*R*K/128]                      */
/*   b2 ro f32 X[K]         b3 rw f32 OUT[M]                          */
struct PcMxfp8 {
    uint32_t e, R, K, row0, row_stride;
    float    wscale;
    uint32_t flags;
};

/* ------------------------------------------------------------------ */
/* conv4_static.comp  local_size_x=64, grid.x = ceil(C/64)              */
/*   out[c] = (sum_lag w[c,3-lag]*x_{t-lag} + x[c]) / sqrt(5);          */
/*   lag<=filled gates history (cold start). F_UPDATE shifts hist.      */
/*   b0 ro f32 X[C]  b1 ro f32 W[C*4] (tap3=current)  b2 rw f32 HIST[C*3]*/
/*   b3 wo f32 OUT[C]                                                 */
struct PcConv4Static { uint32_t C, filled, flags; };

/* conv4_dynamic.comp  local_size_x=64, grid.x = ceil(C/64)             */
/*   tap(lag,c) = sw[lag*C+c] + dyn[(c/head_dim)*4+lag], lag0=current;  */
/*   out = (conv+x)/3. C=1024, head_dim=128, dyn[32].                   */
/*   b0 ro f32 X[C]  b1 ro f32 SW[4*C]  b2 ro f32 DYN[H*4]              */
/*   b3 rw f32 HIST[C*3]  b4 wo f32 OUT[C]                            */
struct PcConv4Dyn { uint32_t C, head_dim, filled, flags; };

/* ------------------------------------------------------------------ */
/* rel_bias.comp  local_size_x=64, grid.x = ceil(HQ*EXT/64)             */
/*   rel[h*ext+d] = sum_e r[h*dr+e]*proj[e*ext+d]                       */
/*   b0 ro f32 R[HQ*dr]  b1 ro f32 PROJ[dr*ext]  b2 wo f32 REL[HQ*ext]  */
struct PcRelBias { uint32_t dr, ext, n /* = HQ*ext */; };

/* attn_scores.comp  local_size_x=64                                    */
/*   grid.x = HQ, grid.y = ceil(n/64). Raw logits into P[h*pstride+jj]. */
/*   b0 ro f32 Q[HQ*DH]  b1 ro f32 K[len*HK*DH]  b2 ro f32 REL[HQ*ext]  */
/*   b3 wo f32 P[HQ*pstride]                                          */
struct PcAttnScores {
    uint32_t hq, hk, dh, j0, n /* len-j0 */, pos, ext, pstride /* >= n+1 */;
};

/* attn_softmax.comp  local_size_x=64, grid.x = HQ                      */
/*   In-place softmax over window scores + sink; sink prob -> index n.  */
/*   b0 rw f32 P[HQ*pstride]   b1 ro f32 SINK[HQ]                       */
struct PcAttnSoftmax { uint32_t n, pstride; };

/* attn_accum.comp  local_size_x=64, grid.x = HQ                        */
/*   o[h*DH+d] = sum_jj p[h,jj] * v[(j0+jj)*HK*DH + g*DH + d]           */
/*   b0 ro f32 P[HQ*pstride]  b1 ro f32 V[len*HK*DH]  b2 wo f32 O[HQ*DH]*/
struct PcAttnAccum { uint32_t hq, hk, dh, j0, n, pstride; };

/* ------------------------------------------------------------------ */
/* attn_res.comp  local_size_x=64, grid = (1,1,1)                       */
/*   Learned softmax read over N sources, Q queries. N<=16, Q<=4.       */
/*   b0 ro f32 SRC[N*D] (block_states ++ {partial}, host-packed)        */
/*   b1 ro f32 Q[Q*D]   b2 wo f32 OUT[Q*D]                            */
struct PcAttnRes { uint32_t N, D, Q; };

/* ------------------------------------------------------------------ */
/* msg_gate.comp  local_size_x=64, grid = (1,1,1). rank<=64.            */
/*   h = bf16io(down @ bf16(cat(rnu(dest),rnu(msg))));                  */
/*   out = (1-tanh(forget@h))*dest + tanh(gain@h)*msg                   */
/*   b0 ro u32 DOWN[rank*2D/2] bf16-packed  b1 ro f32 FORGET[D*rank]    */
/*   b2 ro f32 GAIN[D*rank]  b3 ro f32 DEST[D]  b4 ro f32 MSG[D]        */
/*   b5 wo f32 OUT[D]                                                 */
struct PcMsgGate { uint32_t D, rank; };

/* ------------------------------------------------------------------ */
/* router.comp  local_size_x=128 (NOTE: 128, not 64), grid = (1,1,1)    */
/*   sigmoid(gate_w@x); top-k by score+bias; renorm raw scores          */
/*   w = w/(sum+1e-20)*route_scale. n_experts<=128.                     */
/*   b0 ro f32 GW[E*K]  b1 ro f32 GB[E]  b2 ro f32 X[K]                 */
/*   b3 wo i32 IDX[k]  b4 wo f32 W[k]                                 */
struct PcRouter { uint32_t n_experts, k, K; float route_scale; };

/* ------------------------------------------------------------------ */
/* SLOT VARIANTS (GpuGraph single-submit decode):                       */
/*   The router's top-k output lives on the DEVICE; in a fully recorded */
/*   command buffer the expert id / router weight cannot ride in push   */
/*   constants, so these variants read them from the router's IDX/W     */
/*   buffers instead. pc.e is reinterpreted as the SLOT index (0..k-1): */
/*   the shader does e = idx[pc.e] (and for moe_accum wscale = wt[pc.e]).*/
/*   Math is otherwise IDENTICAL to the base shader.                    */
/*   matvec_nvfp4_slot.comp: bindings 0-5 as matvec_nvfp4 + b6 IDX i32  */
/*   matvec_mxfp8_slot.comp: bindings 0-3 as matvec_mxfp8 + b4 IDX i32  */
/*   moe_accum_slot.comp:    bindings 0-6 as moe_accum + b7 IDX i32,    */
/*                           b8 WT f32; pc.wscale unused                */
/*                                                                    */
/* moe_accum.comp  local_size_x=64, grid.x = D (2304).                  */
/*   Fused: a = softcap(gate,gcap)*sigmoid(gate)*softcap(up,ucap) into  */
/*   shared, then out[r] += wscale * dot(W2[e,r,:], a). Host pre-zeros  */
/*   OUT once per token and dispatches once per selected expert.        */
/*   Pre-step per expert: 2 expert-matvec dispatches on w13 filling     */
/*   GATE/UP (row0 0/1, stride 2, M=I). I<=1152. K=I%64==0 (nvfp4) or   */
/*   %128 (mxfp8). amax/osd only read when quant==0 (bind dummies else).*/
/*   b0 ro f32 GATE[I]  b1 ro f32 UP[I]  b2 ro u32 W2 packed            */
/*   b3 ro u32 S2  b4 ro f32 AMAX[E]  b5 ro f32 OSD[1]  b6 rw f32 OUT[D]*/
struct PcMoeAccum {
    uint32_t e, R, I, quant;   /* quant: 0=NVFP4 1=MXFP8 */
    float    gcap, ucap, wscale;
};

/* ------------------------------------------------------------------ */
/* Small elementwise kernels (all local_size_x=64, grid.x=ceil(n/64)    */
/* unless noted):                                                     */
/*                                                                    */
/* sigmoid_mul.comp:  o[i] *= sigmoid(g[i])                           */
/*   b0 rw f32 O[n]  b1 ro f32 G[n];  PC {uint n}                     */
/*                                                                    */
/* softcap.comp:      l[i] = cap*tanh(l[i]/cap)   (final logits, 30)  */
/*   b0 rw f32 L[n];  PC {uint n; float cap}                          */
/*                                                                    */
/* vadd.comp:         out[i] = a[i]+b[i]  (residual adds)             */
/*   b0 ro f32 A[n]  b1 ro f32 B[n]  b2 wo f32 OUT[n];  PC {uint n}   */
/*                                                                    */
/* l2_clip.comp:      per-head clip, grid.x = heads                   */
/*   v[h,d] *= min(1, thr/(||v[h]||+1e-6)); F_USE_SCALE multiplies    */
/*   v[h,d] *= sc[h,d] first (scaled_l2_clip).                        */
/*   b0 rw f32 V[heads*dim]  b1 ro f32 SCALE[heads*dim];              */
/*   PC {uint dim; float thr; uint flags}                             */
/*                                                                    */
/* log_scale.comp:    v[i] *= tau  (tau computed host-side in double) */
/*   b0 rw f32 V[n];  PC {uint n; float tau}                          */
/*                                                                    */
/* embed_norm.comp:   out = rmsnorm_w(embed[token]) (grid=1 wg)       */
/*   b0 ro u32 EMB[V*n] (F_SRC_BF16: bf16-packed else f32 bits)       */
/*   b1 ro f32 GAMMA[n]  b2 wo f32 OUT[n];                            */
/*   PC {uint token; uint n; float eps; uint flags}                   */
/*                                                                    */
/* argmax_part.comp:  stage 1, grid.x = nchunks                       */
/*   b0 ro f32 X[n]  b1 wo f32 PV[nchunks]  b2 wo u32 PI[nchunks];    */
/*   PC {uint n; uint chunk}                                          */
/* argmax_final.comp: stage 2, grid=1 wg                              */
/*   b0 ro f32 PV[P]  b1 ro u32 PI[P]  b2 wo u32 RES[1] (argmax idx)  */
/*   PC {uint p}                                                      */

}  // namespace ink::shaders
