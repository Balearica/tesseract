///////////////////////////////////////////////////////////////////////
// File:        intsindmatrixsse.cpp
// Description: SSE implementation of 8-bit int SIMD matrix multiply.
// Author:      Ray Smith
//
// (C) Copyright 2017, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#if !defined(__SSE4_1__)
#  if defined(__i686__) || defined(__x86_64__)
#    error Implementation only for SSE 4.1 capable architectures
#  endif
#else

#  include "intsimdmatrix.h"

#  include <emmintrin.h>
#  include <smmintrin.h>
#  include <cstdint>

namespace tesseract {

// --- SSE backend configuration ---
constexpr int kNumOutputsPerRegister   = 1;
constexpr int kMaxOutputRegisters      = 1;
constexpr int kNumInputsPerRegister    = 1;
constexpr int kNumInputsPerGroup       = 2;

static inline void IntDotProduct4xSSE_u16_w8(
    const int16_t* __restrict u16,
    const int8_t*  __restrict w0,
    const int8_t*  __restrict w1,
    const int8_t*  __restrict w2,
    const int8_t*  __restrict w3,
    int n,
    int32_t& out0, int32_t& out1, int32_t& out2, int32_t& out3) {

  int off = 0;
  __m128i acc0 = _mm_setzero_si128();
  __m128i acc1 = _mm_setzero_si128();
  __m128i acc2 = _mm_setzero_si128();
  __m128i acc3 = _mm_setzero_si128();

  for (; off + 16 <= n; off += 16) {
    // Load 16 int16 inputs: split into low/high 8.
    __m128i u_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(u16 + off));
    __m128i u_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(u16 + off + 8));

    // Load 4 rows of weights
    __m128i w0v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w0 + off));
    __m128i w1v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w1 + off));
    __m128i w2v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w2 + off));
    __m128i w3v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w3 + off));

    // Expand low/high 8 bytes -> 8x i16
    __m128i w0_lo = _mm_cvtepi8_epi16(w0v);
    __m128i w0_hi = _mm_cvtepi8_epi16(_mm_srli_si128(w0v, 8));
    __m128i w1_lo = _mm_cvtepi8_epi16(w1v);
    __m128i w1_hi = _mm_cvtepi8_epi16(_mm_srli_si128(w1v, 8));
    __m128i w2_lo = _mm_cvtepi8_epi16(w2v);
    __m128i w2_hi = _mm_cvtepi8_epi16(_mm_srli_si128(w2v, 8));
    __m128i w3_lo = _mm_cvtepi8_epi16(w3v);
    __m128i w3_hi = _mm_cvtepi8_epi16(_mm_srli_si128(w3v, 8));

    // Pmaddwd
    acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(u_lo, w0_lo));
    acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(u_hi, w0_hi));

    acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(u_lo, w1_lo));
    acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(u_hi, w1_hi));

    acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(u_lo, w2_lo));
    acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(u_hi, w2_hi));

    acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(u_lo, w3_lo));
    acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(u_hi, w3_hi));
  }

  // horizontal sum helper
  auto hsum4 = [](__m128i v) -> int32_t {
    __m128i tmp = _mm_shuffle_epi32(v, _MM_SHUFFLE(1,0,3,2));
    v = _mm_add_epi32(v, tmp);
    tmp = _mm_shuffle_epi32(v, _MM_SHUFFLE(2,3,0,1));
    v = _mm_add_epi32(v, tmp);
    return _mm_cvtsi128_si32(v);
  };

  int32_t s0 = hsum4(acc0);
  int32_t s1 = hsum4(acc1);
  int32_t s2 = hsum4(acc2);
  int32_t s3 = hsum4(acc3);

  // scalar tail
  for (; off < n; ++off) {
    int32_t uval = static_cast<int32_t>(u16[off]);
    s0 += uval * static_cast<int32_t>(w0[off]);
    s1 += uval * static_cast<int32_t>(w1[off]);
    s2 += uval * static_cast<int32_t>(w2[off]);
    s3 += uval * static_cast<int32_t>(w3[off]);
  }

  out0 = s0; out1 = s1; out2 = s2; out3 = s3;
}

// Dot product between pre-expanded u16 (int16) and weights w8 (int8).
// n = rounded_num_in (multiple of kNumInputsPerGroup).
static inline int32_t IntDotProductSSE_u16(const int16_t* __restrict u16,
                                           const int8_t*  __restrict w8,
                                           int n) {
  int offset = 0;
  __m128i acc0 = _mm_setzero_si128();
  __m128i acc1 = _mm_setzero_si128();

  // Main loop: 16 inputs per iteration.
  for (; offset + 16 <= n; offset += 16) {
    // Load 16 int16 inputs: split into low/high 8.
    __m128i u_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(u16 + offset));      // 8x i16
    __m128i u_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(u16 + offset + 8));  // 8x i16

    // Load 16 weight bytes.
    __m128i w   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w8 + offset));
    __m128i w_lo = _mm_cvtepi8_epi16(w);                      // low 8 -> 8x i16
    __m128i w_hi = _mm_cvtepi8_epi16(_mm_srli_si128(w, 8));   // high 8 -> 8x i16

    // Pmaddwd
    acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(u_lo, w_lo));
    acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(u_hi, w_hi));
  }

  __m128i sum = _mm_add_epi32(acc0, acc1);

  // Leftover 8 inputs.
  if (offset + 8 <= n) {
    __m128i u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(u16 + offset)); // 8x i16
    __m128i w8v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w8 + offset)); // 8x i8 in low 64b
    offset += 8;

    __m128i w16 = _mm_cvtepi8_epi16(w8v);
    sum = _mm_add_epi32(sum, _mm_madd_epi16(u8, w16));
  }

  // Horizontal sum of 4 lanes.
  __m128i tmp = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1,0,3,2));
  sum = _mm_add_epi32(sum, tmp);
  tmp = _mm_shuffle_epi32(sum, _MM_SHUFFLE(2,3,0,1));
  sum = _mm_add_epi32(sum, tmp);

  int32_t result = _mm_cvtsi128_si32(sum);

  // Scalar tail (0–7 inputs).
  for (; offset < n; ++offset) {
    result += static_cast<int32_t>(u16[offset]) *
              static_cast<int32_t>(w8[offset]);
  }
  return result;
}

// Computes part of matrix.vector v = Wu. Computes 1 result.
static inline void PartialMatrixDotVector1_u16(const int8_t* wi,
                                               const TFloat* scale,
                                               const int16_t* u16,
                                               int rounded_num_in,
                                               TFloat* v) {
  int32_t total = IntDotProductSSE_u16(u16, wi, rounded_num_in);
  int32_t bias  = static_cast<int32_t>(wi[rounded_num_in]) * INT8_MAX;
  *v = (total + bias) * *scale;
}

static void matrixDotVector(int dim1, int dim2,
                            const int8_t* wi,
                            const TFloat* scales,
                            const int8_t* u,
                            TFloat* v) {
  const int num_out = dim1;
  const int num_in = dim2 - 1;

  const int rounded_num_in =
      IntSimdMatrix::Roundup(num_in, kNumInputsPerGroup);
  const int row_stride = rounded_num_in + 1;

  // --- Pre-expand u to int16 once ---
  std::vector<int16_t> u16;
  u16.resize(rounded_num_in, 0);

  int i = 0;
  // Vectorized expand in chunks of 16 bytes.
  for (; i + 16 <= num_in; i += 16) {
    __m128i u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(u + i));
    __m128i lo = _mm_cvtepi8_epi16(u8);                     // first 8
    __m128i hi = _mm_cvtepi8_epi16(_mm_srli_si128(u8, 8));  // next 8
    _mm_storeu_si128(reinterpret_cast<__m128i*>(u16.data() + i), lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(u16.data() + i + 8), hi);
  }
  // Tail for remaining inputs.
  for (; i < num_in; ++i) {
    u16[i] = static_cast<int16_t>(u[i]);
  }
  // Any padded elements [num_in .. rounded_num_in) stay 0.

  int out = 0;
  
  // ---- 4-row fusion ----
  for (; out + 3 < num_out; out += 4) {
    int32_t s0, s1, s2, s3;
    IntDotProduct4xSSE_u16_w8(u16.data(),
                              wi + 0*row_stride,
                              wi + 1*row_stride,
                              wi + 2*row_stride,
                              wi + 3*row_stride,
                              rounded_num_in,
                              s0, s1, s2, s3);

    int32_t b0 = wi[0*row_stride + rounded_num_in] * INT8_MAX;
    int32_t b1 = wi[1*row_stride + rounded_num_in] * INT8_MAX;
    int32_t b2 = wi[2*row_stride + rounded_num_in] * INT8_MAX;
    int32_t b3 = wi[3*row_stride + rounded_num_in] * INT8_MAX;

    v[0] = (s0 + b0) * scales[0];
    v[1] = (s1 + b1) * scales[1];
    v[2] = (s2 + b2) * scales[2];
    v[3] = (s3 + b3) * scales[3];

    wi     += 4 * row_stride;
    scales += 4;
    v      += 4;
  }

  // tail (0~3 outputs)
  for (; out < num_out; ++out) {
    PartialMatrixDotVector1_u16(wi, scales, u16.data(), rounded_num_in, v);
    wi += row_stride; ++scales; ++v;
  }

}

const IntSimdMatrix IntSimdMatrix::intSimdMatrixSSE = {
    matrixDotVector,
    // Number of 32 bit outputs held in each register.
    kNumOutputsPerRegister,
    // Maximum number of registers that we will use to hold outputs.
    kMaxOutputRegisters,
    // Number of 8 bit inputs in the inputs register.
    kNumInputsPerRegister,
    // Number of inputs in each weight group.
    kNumInputsPerGroup
};

} // namespace tesseract.

#endif
