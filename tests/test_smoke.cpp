// Smoke test: dtype codecs, safetensors mmap parse, op sanity.
// Later wired to testdata/golden/ for bit-compare vs HF reference.
#include "ink/tensor.h"
#include "io/config.h"
#include "io/safetensors.h"
#include "model/ops.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace ink;

static int fails = 0;
#define CHECK(cond)                                                  \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
      fails++;                                                       \
    }                                                                \
  } while (0)
static bool close(float a, float b, float tol = 1e-5f) {
  return std::fabs(a - b) <= tol * (1.f + std::fabs(b));
}

static void test_codecs() {
  CHECK(bf16_to_f32(f32_to_bf16(1.0f)) == 1.0f);
  CHECK(bf16_to_f32(f32_to_bf16(-2.5f)) == -2.5f);
  CHECK(close(bf16_to_f32(f32_to_bf16(0.1f)), 0.1f, 1e-2f));
  CHECK(f8e4m3_to_f32(0x38) == 1.0f);   // e4m3: 0 0111 000
  CHECK(f8e4m3_to_f32(0x3E) == 1.75f);  // 0 0111 110
  CHECK(f8e4m3_to_f32(0xB8) == -1.0f);
  CHECK(f8e4m3_to_f32(0x01) == 0.001953125f);  // subnormal: m/8 * 2^-6 = 2^-9
  CHECK(e2m1_to_f32(0x2) == 1.0f);
  CHECK(e2m1_to_f32(0x7) == 6.0f);
  CHECK(e2m1_to_f32(0x9) == -0.5f);
  CHECK(ue8m0_to_f32(127) == 1.0f);
  CHECK(ue8m0_to_f32(128) == 2.0f);
}

static void test_ops() {
  // rmsnorm weighted: x=[3,4], rstd=1/sqrt(12.5+eps)
  float x[2] = {3.f, 4.f}, w[2] = {2.f, 0.5f}, o[2];
  ops::rmsnorm_w(x, w, o, 2, 1e-6f);
  float r = 1.f / std::sqrt(12.5f + 1e-6f);
  CHECK(close(o[0], 3 * r * 2));
  CHECK(close(o[1], 4 * r * 0.5f));

  // l2_clip: norm 10 > thr 2 -> scaled to ~2
  float v[4] = {10, 0, 0, 0};
  ops::l2_clip(v, 1, 4, 2.f);
  CHECK(close(v[0], 2.f, 1e-4f));

  // scaled_l2_clip: scale then cap
  float sv[4] = {1, 0, 0, 0};
  std::vector<int64_t> sh = {1, 4};
  std::vector<uint16_t> sb(4);
  for (int i = 0; i < 4; i++) sb[i] = f32_to_bf16(i == 0 ? 10.f : 1.f);
  TensorView sc = TensorView::make(DType::BF16, sb.data(), sh);
  ops::scaled_l2_clip(sv, sc, 1, 4, 2.f);
  CHECK(close(sv[0], 2.f, 1e-4f));

  // softmax
  float p[3] = {1, 2, 3};
  ops::softmax(p, 3);
  CHECK(close(p[0] + p[1] + p[2], 1.f));
  CHECK(p[2] > p[1] && p[1] > p[0]);

  // softcap / sigmoid
  CHECK(close(ops::softcap(0.f, 30.f), 0.f));
  CHECK(close(ops::sigmoid(0.f), 0.5f));

  // static conv4: C=1, w=[1,1,1,1]; inputs 1,2,3,4,5 -> conv sums window
  {
    uint16_t wb[4];
    for (int i = 0; i < 4; i++) wb[i] = f32_to_bf16(1.f);
    TensorView wv = TensorView::make(DType::BF16, wb, {1, 1, 4});
    ops::ConvState4 st;
    float in[5] = {1, 2, 3, 4, 5}, y;
    float expect[5] = {1, 3, 6, 10, 14};  // causal sums
    for (int t = 0; t < 5; t++) {
      ops::sconv4_static(&in[t], wv, st, &y);
      float want = (expect[t] + in[t]) / std::sqrt(5.f);
      CHECK(close(y, want, 1e-3f));
    }
  }

  // dynamic conv4: C=2, one head of dim 2, static=0, dyn taps [1,0,0,0]
  {
    std::vector<uint16_t> sw(8, f32_to_bf16(0.f));
    TensorView swv = TensorView::make(DType::BF16, sw.data(), {4, 2});
    ops::ConvState4 st;
    float dyn[4] = {2, 0, 0, 0};  // tap0*2 = current*2
    float in = 3.f, y;
    ops::sconv4_dynamic(&in, dyn, swv, 2, st, &y);
    CHECK(close(y, (2 * 3.f + 3.f) / 3.f));
  }

  // attn_res: sources [1,0],[0,1], query picks via logits
  {
    float s0[2] = {1, 0}, s1[2] = {0, 1}, q[2] = {10, 0}, out[2];
    const float* srcs[2] = {s0, s1};
    ops::attn_res_read(srcs, 2, q, 1, 2, out);
    CHECK(out[0] > out[1]);
    // sources rms-normalized to norm sqrt(D): out ~ sqrt(2)*p0 at coord 0
    CHECK(out[0] > 1.4f && out[1] < 0.01f);
  }
}

static void test_router() {
  // 4 experts, k=2, bias flips selection but weights stay raw scores
  std::vector<uint16_t> gw(4 * 2);
  float x[2] = {1, 0};
  // logits: e0=3, e1=2, e2=1, e3=0
  float gvals[8] = {3, 0, 2, 0, 1, 0, 0, 0};
  for (int i = 0; i < 8; i++) gw[i] = f32_to_bf16(gvals[i]);
  std::vector<uint16_t> gb(4);
  float bvals[4] = {0, 0, 100.f, 0};  // bias pushes e2 into top-2
  for (int i = 0; i < 4; i++) gb[i] = f32_to_bf16(bvals[i]);
  TensorView w = TensorView::make(DType::BF16, gw.data(), {4, 2});
  TensorView b = TensorView::make(DType::BF16, gb.data(), {4});
  int idx[2];
  float wt[2];
  ops::router_topk(w, b, x, 4, 2, 8.f, idx, wt);
  CHECK(idx[0] == 2 && idx[1] == 0);  // sorted by selection score desc
  float s0 = ops::sigmoid(3), s2 = ops::sigmoid(1);
  CHECK(close(wt[0], s2 / (s0 + s2) * 8.f, 1e-4f));
  CHECK(close(wt[1], s0 / (s0 + s2) * 8.f, 1e-4f));
}

static void test_safetensors(const std::filesystem::path& tmp) {
  // hand-build a tiny safetensors: u64 len + JSON + blob
  float vals[6] = {1, 2, 3, 4, 5, 6};
  std::string hdr =
      "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
      "\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[24,28]}}";
  std::filesystem::path p = tmp / "tiny.safetensors";
  {
    std::ofstream f(p, std::ios::binary);
    uint64_t n = hdr.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(hdr.data(), hdr.size());
    f.write(reinterpret_cast<const char*>(vals), 24);
    uint8_t u[4] = {7, 8, 9, 10};
    f.write(reinterpret_cast<const char*>(u), 4);
  }
  SafetensorsFile sf;
  std::string err;
  CHECK(sf.open(p, &err));
  if (!err.empty()) std::printf("  err: %s\n", err.c_str());
  const TensorView* a = sf.find("a");
  CHECK(a && a->nd == 2 && a->shape[0] == 2 && a->shape[1] == 3);
  CHECK(a && close(a->f32(1, 2), 6.f));
  const TensorView* b = sf.find("b");
  CHECK(b && b->f32(3) == 10.f);
  sf.close();
}

int main() {
  test_codecs();
  test_ops();
  test_router();
  std::filesystem::path tmp = std::filesystem::temp_directory_path() / "ink_smoke";
  std::filesystem::create_directories(tmp);
  test_safetensors(tmp);
  if (fails == 0) std::printf("smoke: all checks passed\n");
  else std::printf("smoke: %d FAILURES\n", fails);
  return fails ? 1 : 0;
}
