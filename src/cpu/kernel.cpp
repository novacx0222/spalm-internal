#include "cpu/kernel.h"

#include <immintrin.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#define prepare_creg(n0, n1)                                                   \
    __m512d c##n0##n1;                                                         \
    c##n0##n1 = _mm512_setzero_pd();

#define inter_fmadd(nu, nm)                                                    \
    a_val0 = _mm512_castps_pd(                                                 \
        _mm512_broadcast_f32x4(_mm_castpd_ps(_mm_loadu_pd(R_ptr + nu)))        \
    );                                                                         \
    c##nu##0 = _mm512_fmadd_pd(a_val0, b_vals0, c##nu##0);                     \
    c##nu##1 = _mm512_fmadd_pd(a_val0, b_vals1, c##nu##1);                     \
    c##nm##0 = _mm512_fmadd_pd(a_val0, b_vals2, c##nm##0);                     \
    c##nm##1 = _mm512_fmadd_pd(a_val0, b_vals3, c##nm##1);

#define prefetch(off, dis)                                                     \
    asm volatile("" ::: "memory");                                             \
    _mm_prefetch(&S_dense[S_dense_idx + (off * 16) + dis], _MM_HINT_T0);       \
    asm volatile("" ::: "memory");

#define iter()                                                                 \
    b_vals0 = _mm512_movedup_pd(_mm512_loadu_pd(S_ptr));                       \
    b_vals1 = _mm512_movedup_pd(_mm512_loadu_pd(S_ptr + 1));                   \
    b_vals2 = _mm512_movedup_pd(_mm512_loadu_pd(S_ptr + 8));                   \
    b_vals3 = _mm512_movedup_pd(_mm512_loadu_pd(S_ptr + 9));                   \
    inter_fmadd(0, 1);                                                         \
    inter_fmadd(2, 3);                                                         \
    inter_fmadd(4, 5);                                                         \
    inter_fmadd(6, 7);                                                         \
    inter_fmadd(8, 9);                                                         \
    inter_fmadd(10, 11);

#define inter_fmadd_cont(nu, nm, t)                                            \
    a_val0 = _mm512_castps_pd(_mm512_broadcast_f32x4(                          \
        _mm_castpd_ps(_mm_loadu_pd(&R_ptr[(i + t) * 12 + nu]))                 \
    ));                                                                        \
    c##nu##0 = _mm512_fmadd_pd(a_val0, b_vals0, c##nu##0);                     \
    c##nu##1 = _mm512_fmadd_pd(a_val0, b_vals1, c##nu##1);                     \
    c##nm##0 = _mm512_fmadd_pd(a_val0, b_vals2, c##nm##0);                     \
    c##nm##1 = _mm512_fmadd_pd(a_val0, b_vals3, c##nm##1);

#define iter_cont(t)                                                           \
    b_vals0 = _mm512_movedup_pd(_mm512_loadu_pd(&S_ptr[(i + t) * 16]));        \
    b_vals1 = _mm512_movedup_pd(_mm512_loadu_pd(&S_ptr[(i + t) * 16 + 1]));    \
    b_vals2 = _mm512_movedup_pd(_mm512_loadu_pd(&S_ptr[(i + t) * 16 + 8]));    \
    b_vals3 = _mm512_movedup_pd(_mm512_loadu_pd(&S_ptr[(i + t) * 16 + 9]));    \
    inter_fmadd_cont(0, 1, t);                                                 \
    inter_fmadd_cont(2, 3, t);                                                 \
    inter_fmadd_cont(4, 5, t);                                                 \
    inter_fmadd_cont(6, 7, t);                                                 \
    inter_fmadd_cont(8, 9, t);                                                 \
    inter_fmadd_cont(10, 11, t);

#define iter_R_cont(t)                                                         \
    b_vals0 = _mm512_movedup_pd(_mm512_loadu_pd(&S_ptr[(i + t) * 16]));        \
    b_vals1 = _mm512_movedup_pd(_mm512_loadu_pd(&S_ptr[(i + t) * 16 + 1]));    \
    b_vals2 = _mm512_movedup_pd(_mm512_loadu_pd(&S_ptr[(i + t) * 16 + 8]));    \
    b_vals3 = _mm512_movedup_pd(_mm512_loadu_pd(&S_ptr[(i + t) * 16 + 9]));    \
    inter_fmadd(0, 1);                                                         \
    inter_fmadd(2, 3);                                                         \
    inter_fmadd(4, 5);                                                         \
    inter_fmadd(6, 7);                                                         \
    inter_fmadd(8, 9);                                                         \
    inter_fmadd(10, 11);

#define iter_S_cont(t)                                                         \
    b_vals0 = _mm512_movedup_pd(_mm512_loadu_pd(S_ptr));                       \
    b_vals1 = _mm512_movedup_pd(_mm512_loadu_pd(S_ptr + 1));                   \
    b_vals2 = _mm512_movedup_pd(_mm512_loadu_pd(S_ptr + 8));                   \
    b_vals3 = _mm512_movedup_pd(_mm512_loadu_pd(S_ptr + 9));                   \
    inter_fmadd_cont(0, 1, t);                                                 \
    inter_fmadd_cont(2, 3, t);                                                 \
    inter_fmadd_cont(4, 5, t);                                                 \
    inter_fmadd_cont(6, 7, t);                                                 \
    inter_fmadd_cont(8, 9, t);                                                 \
    inter_fmadd_cont(10, 11, t);

#define interleave_store(n0, n1)                                               \
    b_vals0 = _mm512_unpacklo_pd(c##n0##0, c##n0##1);                          \
    b_vals1 = _mm512_unpacklo_pd(c##n1##0, c##n1##1);                          \
    b_vals2 = _mm512_unpackhi_pd(c##n0##0, c##n0##1);                          \
    b_vals3 = _mm512_unpackhi_pd(c##n1##0, c##n1##1);                          \
    _mm512_storeu_pd(                                                          \
        &result_p[n0 * N],                                                     \
        _mm512_add_pd(b_vals0, _mm512_loadu_pd(&result_p[n0 * N]))             \
    );                                                                         \
    _mm512_storeu_pd(                                                          \
        &result_p[n0 * N + 8],                                                 \
        _mm512_add_pd(b_vals1, _mm512_loadu_pd(&result_p[n0 * N + 8]))         \
    );                                                                         \
    _mm512_storeu_pd(                                                          \
        &result_p[n1 * N],                                                     \
        _mm512_add_pd(b_vals2, _mm512_loadu_pd(&result_p[n1 * N]))             \
    );                                                                         \
    _mm512_storeu_pd(                                                          \
        &result_p[n1 * N + 8],                                                 \
        _mm512_add_pd(b_vals3, _mm512_loadu_pd(&result_p[n1 * N + 8]))         \
    );

#define interleave_store_lo(n0, n1)                                            \
    b_vals0 = _mm512_unpacklo_pd(c##n0##0, c##n0##1);                          \
    b_vals1 = _mm512_unpacklo_pd(c##n1##0, c##n1##1);                          \
    _mm512_mask_storeu_pd(                                                     \
        &result_p[n0 * N], k0,                                                 \
        _mm512_add_pd(b_vals0, _mm512_maskz_loadu_pd(k0, &result_p[n0 * N]))   \
    );                                                                         \
    _mm512_mask_storeu_pd(                                                     \
        &result_p[n0 * N + 8], k1,                                             \
        _mm512_add_pd(                                                         \
            b_vals1, _mm512_maskz_loadu_pd(k1, &result_p[n0 * N + 8])          \
        )                                                                      \
    );

#define interleave_store_hi(n0, n1)                                            \
    b_vals0 = _mm512_unpackhi_pd(c##n0##0, c##n0##1);                          \
    b_vals1 = _mm512_unpackhi_pd(c##n1##0, c##n1##1);                          \
    _mm512_mask_storeu_pd(                                                     \
        &result_p[n1 * N], k0,                                                 \
        _mm512_add_pd(b_vals0, _mm512_maskz_loadu_pd(k0, &result_p[n1 * N]))   \
    );                                                                         \
    _mm512_mask_storeu_pd(                                                     \
        &result_p[n1 * N + 8], k1,                                             \
        _mm512_add_pd(                                                         \
            b_vals1, _mm512_maskz_loadu_pd(k1, &result_p[n1 * N + 8])          \
        )                                                                      \
    );

#define interleave_store_hi_nocont(n0, n1)                                     \
    b_vals0 = _mm512_unpackhi_pd(c##n0##0, c##n0##1);                          \
    b_vals1 = _mm512_unpackhi_pd(c##n1##0, c##n1##1);                          \
    _mm512_mask_i32scatter_pd(                                                 \
        write_locations[n1], k0, scatter_idx0,                                 \
        _mm512_add_pd(                                                         \
            b_vals0, _mm512_mask_i32gather_pd(                                 \
                         b_vals0, k0, scatter_idx0, write_locations[n1], 8     \
                     )                                                         \
        ),                                                                     \
        8                                                                      \
    );                                                                         \
    _mm512_mask_i32scatter_pd(                                                 \
        write_locations[n1], k1, scatter_idx1,                                 \
        _mm512_add_pd(                                                         \
            b_vals1, _mm512_mask_i32gather_pd(                                 \
                         b_vals1, k1, scatter_idx1, write_locations[n1], 8     \
                     )                                                         \
        ),                                                                     \
        8                                                                      \
    );

#define interleave_store_lo_nocont(n0, n1)                                     \
    b_vals0 = _mm512_unpacklo_pd(c##n0##0, c##n0##1);                          \
    b_vals1 = _mm512_unpacklo_pd(c##n1##0, c##n1##1);                          \
    _mm512_mask_i32scatter_pd(                                                 \
        write_locations[n0], k0, scatter_idx0,                                 \
        _mm512_add_pd(                                                         \
            b_vals0, _mm512_mask_i32gather_pd(                                 \
                         b_vals0, k0, scatter_idx0, write_locations[n0], 8     \
                     )                                                         \
        ),                                                                     \
        8                                                                      \
    );                                                                         \
    _mm512_mask_i32scatter_pd(                                                 \
        write_locations[n0], k1, scatter_idx1,                                 \
        _mm512_add_pd(                                                         \
            b_vals1, _mm512_mask_i32gather_pd(                                 \
                         b_vals1, k1, scatter_idx1, write_locations[n0], 8     \
                     )                                                         \
        ),                                                                     \
        8                                                                      \
    );

#define interleave_store_lo_sparse(n0, n1)                                     \
    b_vals0 = _mm512_unpacklo_pd(c##n0##0, c##n0##1);                          \
    b_vals1 = _mm512_unpacklo_pd(c##n1##0, c##n1##1);                          \
    _mm512_mask_storeu_pd(                                                     \
        result_p, k0, _mm512_add_pd(b_vals0, _mm512_loadu_pd(result_p))        \
    );                                                                         \
    _mm512_mask_storeu_pd(                                                     \
        &result_p[8], k1,                                                      \
        _mm512_add_pd(b_vals1, _mm512_loadu_pd(&result_p[8]))                  \
    );

#define interleave_store_hi_sparse(n0, n1)                                     \
    b_vals0 = _mm512_unpackhi_pd(c##n0##0, c##n0##1);                          \
    b_vals1 = _mm512_unpackhi_pd(c##n1##0, c##n1##1);                          \
    _mm512_mask_storeu_pd(                                                     \
        result_p, k0, _mm512_add_pd(b_vals0, _mm512_loadu_pd(result_p))        \
    );                                                                         \
    _mm512_mask_storeu_pd(                                                     \
        &result_p[8], k1,                                                      \
        _mm512_add_pd(b_vals1, _mm512_loadu_pd(&result_p[8]))                  \
    );

int compute_offsets(
    const uint64_t R_bm, const uint64_t S_bm, char offsets[],
    uint16_t offsets16[], __m512i _8bit_indexes
) {
    int total_1s = 0;

    const int R_slice_size = 12;
    const int S_slice_size = 16;

    const uint64_t and_bits = R_bm & S_bm;

    __m512i r_idxs = _mm512_maskz_expand_epi8(R_bm, _8bit_indexes);
    r_idxs = _mm512_maskz_compress_epi8(and_bits, r_idxs);
    _mm512_storeu_si512((__m128i *) offsets, r_idxs);
    __m512i slice_size_reg = _mm512_set1_epi16(R_slice_size);
    __m512i idxs = _mm512_mullo_epi16(
        slice_size_reg,
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *) &offsets[0]))
    );
    _mm512_storeu_si512((__m512i *) &offsets16[0], idxs);
    idxs = _mm512_mullo_epi16(
        slice_size_reg,
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *) &offsets[32]))
    );
    _mm512_storeu_si512((__m512i *) &offsets16[32], idxs);

    __m512i s_idxs = _mm512_maskz_expand_epi8(S_bm, _8bit_indexes);
    s_idxs = _mm512_maskz_compress_epi8(and_bits, s_idxs);
    _mm512_storeu_si512((__m128i *) offsets, s_idxs);
    slice_size_reg = _mm512_set1_epi16(S_slice_size);
    idxs = _mm512_mullo_epi16(
        slice_size_reg,
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *) &offsets[0]))
    );
    _mm512_storeu_si512((__m512i *) &offsets16[64], idxs);
    idxs = _mm512_mullo_epi16(
        slice_size_reg,
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *) &offsets[32]))
    );
    _mm512_storeu_si512((__m512i *) &offsets16[96], idxs);

    total_1s = __builtin_popcountll(and_bits);

    return total_1s;
}

int compute_offsets_R(
    const uint64_t S_bm, char offsets[], uint16_t offsets16[],
    __m512i _8bit_indexes
) {
    const int R_slice_size = 12;
    __m512i r_idxs = _mm512_maskz_compress_epi8(S_bm, _8bit_indexes);
    _mm512_storeu_si512((__m128i *) offsets, r_idxs);
    __m512i slice_size_reg = _mm512_set1_epi16(R_slice_size);
    __m512i idxs = _mm512_mullo_epi16(
        slice_size_reg,
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *) &offsets[0]))
    );
    _mm512_storeu_si512((__m512i *) &offsets16[0], idxs);
    idxs = _mm512_mullo_epi16(
        slice_size_reg,
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *) &offsets[32]))
    );
    _mm512_storeu_si512((__m512i *) &offsets16[32], idxs);

    int total_1s = __builtin_popcountll(S_bm);
    return total_1s;
}

int compute_offsets_S(
    const uint64_t R_bm, char offsets[], uint16_t offsets16[],
    __m512i _8bit_indexes
) {
    const int S_slice_size = 16;
    __m512i s_idxs = _mm512_maskz_compress_epi8(R_bm, _8bit_indexes);
    _mm512_storeu_si512((__m128i *) offsets, s_idxs);
    __m512i slice_size_reg = _mm512_set1_epi16(S_slice_size);
    __m512i idxs = _mm512_mullo_epi16(
        slice_size_reg,
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *) &offsets[0]))
    );
    _mm512_storeu_si512((__m512i *) &offsets16[0], idxs);
    idxs = _mm512_mullo_epi16(
        slice_size_reg,
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *) &offsets[32]))
    );
    _mm512_storeu_si512((__m512i *) &offsets16[32], idxs);

    return __builtin_popcountll(R_bm);
}

void compute_offsets_and(
    const uint64_t and_bm, uint8_t *offsets, __m512i _8bit_indexes
) {
    __m512i indexes = _mm512_maskz_compress_epi8(and_bm, _8bit_indexes);
    _mm512_storeu_si512((__m512i *) offsets, indexes);
    // int total_1s = __builtin_popcountll(and_bm);
    // return total_1s;
}

void __attribute__((noinline)) inner_kernel_bm(
    double *__restrict__ R, BIT_DTYPE *rbm, double *__restrict__ S,
    BIT_DTYPE *sbm, const int panel_size, const int R_sh, const int S_sh,
    const slice &R_slice_info, const int *R_row_idxs, const slice &S_slice_info,
    const int *S_row_idxs, const int M, const int N, double *__restrict__ ret,
    bool sparse_result, int *csr_agg_idxs, const int row_offset
) {
    const int R_slice_size = 12;
    const int S_slice_size = 16;

    __m512d a_val0;

    __m512d b_vals0;
    __m512d b_vals1;
    __m512d b_vals2;
    __m512d b_vals3;

    __m512i _8bit_indexes = _mm512_set_epi32(
        0x3f3e3d3c, 0x3b3a3938, 0x37363534, 0x33323130, 0x2f2e2d2c, 0x2b2a2928,
        0x27262524, 0x23222120, 0x1f1e1d1c, 0x1b1a1918, 0x17161514, 0x13121110,
        0x0f0e0d0c, 0x0b0a0908, 0x07060504, 0x03020100
    );

    register __m512d c00 asm("zmm24");
    register __m512d c01 asm("zmm25");
    register __m512d c10 asm("zmm2");
    register __m512d c11 asm("zmm3");
    register __m512d c20 asm("zmm4");
    register __m512d c21 asm("zmm5");
    register __m512d c30 asm("zmm6");
    register __m512d c31 asm("zmm7");
    register __m512d c40 asm("zmm8");
    register __m512d c41 asm("zmm9");
    register __m512d c50 asm("zmm10");
    register __m512d c51 asm("zmm11");
    register __m512d c60 asm("zmm12");
    register __m512d c61 asm("zmm13");
    register __m512d c70 asm("zmm14");
    register __m512d c71 asm("zmm15");
    register __m512d c80 asm("zmm16");
    register __m512d c81 asm("zmm17");
    register __m512d c90 asm("zmm18");
    register __m512d c91 asm("zmm19");
    register __m512d c100 asm("zmm20");
    register __m512d c101 asm("zmm21");
    register __m512d c110 asm("zmm22");
    register __m512d c111 asm("zmm23");

    c00 = _mm512_setzero_pd();
    c01 = _mm512_setzero_pd();
    c10 = _mm512_setzero_pd();
    c11 = _mm512_setzero_pd();
    c20 = _mm512_setzero_pd();
    c21 = _mm512_setzero_pd();
    c30 = _mm512_setzero_pd();
    c31 = _mm512_setzero_pd();
    c40 = _mm512_setzero_pd();
    c41 = _mm512_setzero_pd();
    c50 = _mm512_setzero_pd();
    c51 = _mm512_setzero_pd();
    c60 = _mm512_setzero_pd();
    c61 = _mm512_setzero_pd();
    c70 = _mm512_setzero_pd();
    c71 = _mm512_setzero_pd();
    c80 = _mm512_setzero_pd();
    c81 = _mm512_setzero_pd();
    c90 = _mm512_setzero_pd();
    c91 = _mm512_setzero_pd();
    c100 = _mm512_setzero_pd();
    c101 = _mm512_setzero_pd();
    c110 = _mm512_setzero_pd();
    c111 = _mm512_setzero_pd();

    const int unroll_factor = 4;

    uint8_t offsets[64];
    for (int i = 0; i < (panel_size + BITS - 1) / BITS; i++) {
        BIT_DTYPE rbm64 = rbm[i];
        BIT_DTYPE sbm64 = sbm[i];

        BIT_DTYPE and_bm = rbm64 & sbm64;

        int cnt = __builtin_popcountll(and_bm);

        if (!cnt) {
            continue;
        }

        double *R_ptr_start = &R[64 * i * R_slice_size];
        double *S_ptr_start = &S[64 * i * S_slice_size];

        if (cnt != BITS) {
            compute_offsets_and(and_bm, offsets, _8bit_indexes);
            const int stop = cnt - (cnt % unroll_factor);
            uint8_t *offset_ptr = offsets;
            for (int i = 0; i < stop; i += unroll_factor) {
                uint8_t next = *(offset_ptr++);
                double *R_ptr = R_ptr_start + next * R_slice_size;
                double *S_ptr = S_ptr_start + next * S_slice_size;
                iter();
                next = *(offset_ptr++);
                R_ptr = R_ptr_start + next * R_slice_size;
                S_ptr = S_ptr_start + next * S_slice_size;
                iter();
                next = *(offset_ptr++);
                R_ptr = R_ptr_start + next * R_slice_size;
                S_ptr = S_ptr_start + next * S_slice_size;
                iter();
                next = *(offset_ptr++);
                R_ptr = R_ptr_start + next * R_slice_size;
                S_ptr = S_ptr_start + next * S_slice_size;
                iter();
            }
            for (int i = stop; i < cnt; i++) {
                uint8_t next = *(offset_ptr++);
                double *R_ptr = R_ptr_start + next * R_slice_size;
                double *S_ptr = S_ptr_start + next * S_slice_size;
                iter();
            }
        } else {
            double *R_ptr = R_ptr_start;
            double *S_ptr = S_ptr_start;
            for (int i = 0; i < BITS; i += unroll_factor) {
                /*
        iter_cont(0);
        iter_cont(1);
        iter_cont(2);
        iter_cont(3);
        */
                iter();
                R_ptr += R_slice_size;
                S_ptr += S_slice_size;
                iter();
                R_ptr += R_slice_size;
                S_ptr += S_slice_size;
                iter();
                R_ptr += R_slice_size;
                S_ptr += S_slice_size;
                iter();
                R_ptr += R_slice_size;
                S_ptr += S_slice_size;
                /*
        double* R_ptr = R_ptr_start + i * R_slice_size;
        double* S_ptr = S_ptr_start + i * S_slice_size;
        iter();
        R_ptr = R_ptr_start + (i+1) * R_slice_size;
        S_ptr = S_ptr_start + (i+1) * S_slice_size;
        iter();
        R_ptr = R_ptr_start + (i+2) * R_slice_size;
        S_ptr = S_ptr_start + (i+2) * S_slice_size;
        iter();
        R_ptr = R_ptr_start + (i+3) * R_slice_size;
        S_ptr = S_ptr_start + (i+3) * S_slice_size;
        iter();
        */
            }
        }
    }

    if (!sparse_result) {
        if (R_slice_info.continuous && S_slice_info.continuous) {
            // if (false) {
            const int I = R_row_idxs[0];
            const int J = S_row_idxs[0];
            const int write_height = M - I;
            const int write_length = N - J;

            __mmask8 k0 = 0b11111111;
            __mmask8 k1 = 0b11111111;
            int shift = write_length >= 8 ? 0 : 8 - write_length;
            k0 >>= shift;
            shift = write_length >= 16 ? 0 : 16 - write_length;
            k1 >>= shift;

            double *result_p = &ret[I * N + J];

            if (R_slice_size <= write_height) {
                interleave_store_lo(0, 1);
                interleave_store_hi(0, 1);
                interleave_store_lo(2, 3);
                interleave_store_hi(2, 3);
                interleave_store_lo(4, 5);
                interleave_store_hi(4, 5);
                interleave_store_lo(6, 7);
                interleave_store_hi(6, 7);
                interleave_store_lo(8, 9);
                interleave_store_hi(8, 9);
                interleave_store_lo(10, 11);
                interleave_store_hi(10, 11);
            } else {
                switch (write_height) {
                case 12:
                    interleave_store_hi(10, 11);
                case 11:
                    interleave_store_lo(10, 11);
                case 10:
                    interleave_store_hi(8, 9);
                case 9:
                    interleave_store_lo(8, 9);
                case 8:
                    interleave_store_hi(6, 7);
                case 7:
                    interleave_store_lo(6, 7);
                case 6:
                    interleave_store_hi(4, 5);
                case 5:
                    interleave_store_lo(4, 5);
                case 4:
                    interleave_store_hi(2, 3);
                case 3:
                    interleave_store_lo(2, 3);
                case 2:
                    interleave_store_hi(0, 1);
                case 1:
                    interleave_store_lo(0, 1);
                }
            }
        } else {
            const int I = R_row_idxs[0];
            const int J = S_row_idxs[0];
            const int last_write_R = M - R_row_idxs[R_slice_info.len - 1] < R_sh
                                         ? M - R_row_idxs[R_slice_info.len - 1]
                                         : R_sh;
            const int write_height =
                R_sh * (R_slice_info.len - 1) + last_write_R;
            const int last_write_S = N - S_row_idxs[S_slice_info.len - 1] < S_sh
                                         ? N - S_row_idxs[S_slice_info.len - 1]
                                         : S_sh;
            const int write_length =
                S_sh * (S_slice_info.len - 1) + last_write_S;

            double *write_locations[12];
            for (int i = 0; i < R_slice_info.len; i++) {
                for (int j = 0; j < R_sh; j++) {
                    write_locations[i * R_sh + j] =
                        &ret[(R_row_idxs[i] + j) * N];
                }
            }

            int scatter_locations[16];
            for (int i = 0; i < S_slice_info.len; i++) {
                for (int j = 0; j < S_sh; j++) {
                    scatter_locations[i * S_sh + j] = S_row_idxs[i] + j;
                }
            }

            __m256i scatter_idx0 =
                _mm256_loadu_si256((__m256i *) scatter_locations);
            __m256i scatter_idx1 =
                _mm256_loadu_si256((__m256i *) (scatter_locations + 8));

            __mmask8 k0 = 0b11111111;
            __mmask8 k1 = 0b11111111;
            int shift = write_length >= 8 ? 0 : 8 - write_length;
            k0 >>= shift;
            shift = write_length >= 16 ? 0 : 16 - write_length;
            k1 >>= shift;

            switch (write_height) {
            case 12:
                interleave_store_hi_nocont(10, 11);
            case 11:
                interleave_store_lo_nocont(10, 11);
            case 10:
                interleave_store_hi_nocont(8, 9);
            case 9:
                interleave_store_lo_nocont(8, 9);
            case 8:
                interleave_store_hi_nocont(6, 7);
            case 7:
                interleave_store_lo_nocont(6, 7);
            case 6:
                interleave_store_hi_nocont(4, 5);
            case 5:
                interleave_store_lo_nocont(4, 5);
            case 4:
                interleave_store_hi_nocont(2, 3);
            case 3:
                interleave_store_lo_nocont(2, 3);
            case 2:
                interleave_store_hi_nocont(0, 1);
            case 1:
                interleave_store_lo_nocont(0, 1);
            }
        }
    } else {
    }
}

#define inter_fmadd_fp32(nu, nm)                                               \
    a_val0 = _mm512_broadcast_f32x2(_mm_loadu_ps(R_ptr + nu));                 \
    c##nu##0 = _mm512_fmadd_ps(a_val0, b_vals0, c##nu##0);                     \
    c##nu##1 = _mm512_fmadd_ps(a_val0, b_vals1, c##nu##1);                     \
    c##nm##0 = _mm512_fmadd_ps(a_val0, b_vals2, c##nm##0);                     \
    c##nm##1 = _mm512_fmadd_ps(a_val0, b_vals3, c##nm##1);

#define iter_fp32()                                                            \
    b_vals0 = _mm512_moveldup_ps(_mm512_loadu_ps(S_ptr));                      \
    b_vals1 = _mm512_moveldup_ps(_mm512_loadu_ps(S_ptr + 1));                  \
    b_vals2 = _mm512_moveldup_ps(_mm512_loadu_ps(S_ptr + 16));                 \
    b_vals3 = _mm512_moveldup_ps(_mm512_loadu_ps(S_ptr + 17));                 \
    inter_fmadd_fp32(0, 1);                                                    \
    inter_fmadd_fp32(2, 3);                                                    \
    inter_fmadd_fp32(4, 5);                                                    \
    inter_fmadd_fp32(6, 7);                                                    \
    inter_fmadd_fp32(8, 9);                                                    \
    inter_fmadd_fp32(10, 11);

/*
#define interleave_store_lo_fp32(n0, n1) \
  b_vals0 = _mm512_unpacklo_ps(c##n0##0, c##n0##1);                          \
  b_vals1 = _mm512_unpacklo_ps(c##n1##0, c##n1##1);                          \
  _mm512_mask_storeu_ps(                                                     \
      &result_p[n0 * N], k0,                                                 \
      _mm512_add_ps(b_vals0, _mm512_maskz_loadu_ps(k0, &result_p[n0 * N]))); \
  _mm512_mask_storeu_ps(                                                     \
      &result_p[n0 * N + 8], k1,                                             \
      _mm512_add_ps(b_vals1,                                                 \
                    _mm512_maskz_loadu_ps(k1, &result_p[n0 * N + 8])));
*/

#define interleave_store_lo_fp32(n0, n1)                                       \
    b_vals0 = _mm512_mask_mov_ps(                                              \
        c##n0##0, 0b1010101010101010,                                          \
        (__m512) _mm512_bslli_epi128((__m512i) c##n0##1, 4)                    \
    );                                                                         \
    b_vals1 = _mm512_mask_mov_ps(                                              \
        c##n1##0, 0b1010101010101010,                                          \
        (__m512) _mm512_bslli_epi128((__m512i) c##n1##1, 4)                    \
    );                                                                         \
    _mm512_mask_storeu_ps(                                                     \
        &result_p[n0 * N], k0,                                                 \
        _mm512_add_ps(b_vals0, _mm512_maskz_loadu_ps(k0, &result_p[n0 * N]))   \
    );                                                                         \
    _mm512_mask_storeu_ps(                                                     \
        &result_p[n0 * N + 16], k1,                                            \
        _mm512_add_ps(                                                         \
            b_vals1, _mm512_maskz_loadu_ps(k1, &result_p[n0 * N + 16])         \
        )                                                                      \
    );

/*
#define interleave_store_hi_fp32(n0, n1) \
  b_vals0 = _mm512_unpackhi_ps(c##n0##0, c##n0##1);                          \
  b_vals1 = _mm512_unpackhi_ps(c##n1##0, c##n1##1);                          \
  _mm512_mask_storeu_ps(                                                     \
      &result_p[n1 * N], k0,                                                 \
      _mm512_add_ps(b_vals0, _mm512_maskz_loadu_ps(k0, &result_p[n1 * N]))); \
  _mm512_mask_storeu_ps(                                                     \
      &result_p[n1 * N + 8], k1,                                             \
      _mm512_add_ps(b_vals1,                                                 \
                    _mm512_maskz_loadu_ps(k1, &result_p[n1 * N + 8])));
*/

#define interleave_store_hi_fp32(n0, n1)                                       \
    b_vals0 = _mm512_mask_mov_ps(                                              \
        (__m512) _mm512_bsrli_epi128((__m512i) c##n0##0, 4),                   \
        0b1010101010101010, c##n0##1                                           \
    );                                                                         \
    b_vals1 = _mm512_mask_mov_ps(                                              \
        (__m512) _mm512_bsrli_epi128((__m512i) c##n1##0, 4),                   \
        0b1010101010101010, c##n1##1                                           \
    );                                                                         \
    _mm512_mask_storeu_ps(                                                     \
        &result_p[n1 * N], k0,                                                 \
        _mm512_add_ps(b_vals0, _mm512_maskz_loadu_ps(k0, &result_p[n1 * N]))   \
    );                                                                         \
    _mm512_mask_storeu_ps(                                                     \
        &result_p[n1 * N + 16], k1,                                            \
        _mm512_add_ps(                                                         \
            b_vals1, _mm512_maskz_loadu_ps(k1, &result_p[n1 * N + 16])         \
        )                                                                      \
    );

#define interleave_store_hi_nocont_fp32(n0, n1)                                \
    b_vals0 = _mm512_mask_mov_ps(                                              \
        (__m512) _mm512_bsrli_epi128((__m512i) c##n0##0, 4),                   \
        0b1010101010101010, c##n0##1                                           \
    );                                                                         \
    b_vals1 = _mm512_mask_mov_ps(                                              \
        (__m512) _mm512_bsrli_epi128((__m512i) c##n1##0, 4),                   \
        0b1010101010101010, c##n1##1                                           \
    );                                                                         \
    _mm512_mask_i32scatter_ps(                                                 \
        write_locations[n1], k0, scatter_idx0,                                 \
        _mm512_add_ps(                                                         \
            b_vals0, _mm512_mask_i32gather_ps(                                 \
                         b_vals0, k0, scatter_idx0, write_locations[n1], 4     \
                     )                                                         \
        ),                                                                     \
        4                                                                      \
    );                                                                         \
    _mm512_mask_i32scatter_ps(                                                 \
        write_locations[n1], k1, scatter_idx1,                                 \
        _mm512_add_ps(                                                         \
            b_vals1, _mm512_mask_i32gather_ps(                                 \
                         b_vals1, k1, scatter_idx1, write_locations[n1], 4     \
                     )                                                         \
        ),                                                                     \
        4                                                                      \
    );

#define interleave_store_lo_nocont_fp32(n0, n1)                                \
    b_vals0 = _mm512_mask_mov_ps(                                              \
        c##n0##0, 0b1010101010101010,                                          \
        (__m512) _mm512_bslli_epi128((__m512i) c##n0##1, 4)                    \
    );                                                                         \
    b_vals1 = _mm512_mask_mov_ps(                                              \
        c##n1##0, 0b1010101010101010,                                          \
        (__m512) _mm512_bslli_epi128((__m512i) c##n1##1, 4)                    \
    );                                                                         \
    _mm512_mask_i32scatter_ps(                                                 \
        write_locations[n0], k0, scatter_idx0,                                 \
        _mm512_add_ps(                                                         \
            b_vals0, _mm512_mask_i32gather_ps(                                 \
                         b_vals0, k0, scatter_idx0, write_locations[n0], 4     \
                     )                                                         \
        ),                                                                     \
        4                                                                      \
    );                                                                         \
    _mm512_mask_i32scatter_ps(                                                 \
        write_locations[n0], k1, scatter_idx1,                                 \
        _mm512_add_ps(                                                         \
            b_vals1, _mm512_mask_i32gather_ps(                                 \
                         b_vals1, k1, scatter_idx1, write_locations[n0], 4     \
                     )                                                         \
        ),                                                                     \
        4                                                                      \
    );

void __attribute__((noinline)) inner_kernel_bm(
    float *__restrict__ R, BIT_DTYPE *rbm, float *__restrict__ S,
    BIT_DTYPE *sbm, const int panel_size, const int R_sh, const int S_sh,
    const slice &R_slice_info, const int *R_row_idxs, const slice &S_slice_info,
    const int *S_row_idxs, const int M, const int N, float *__restrict__ ret,
    bool sparse_result, int *csr_agg_idxs, const int row_offset
) {
    const int R_slice_size = 12;
    const int S_slice_size = 32;

    __m512 a_val0;

    __m512 b_vals0;
    __m512 b_vals1;
    __m512 b_vals2;
    __m512 b_vals3;

    __m512i _8bit_indexes = _mm512_set_epi32(
        0x3f3e3d3c, 0x3b3a3938, 0x37363534, 0x33323130, 0x2f2e2d2c, 0x2b2a2928,
        0x27262524, 0x23222120, 0x1f1e1d1c, 0x1b1a1918, 0x17161514, 0x13121110,
        0x0f0e0d0c, 0x0b0a0908, 0x07060504, 0x03020100
    );

    register __m512 c00 asm("zmm24");
    register __m512 c01 asm("zmm25");
    register __m512 c10 asm("zmm2");
    register __m512 c11 asm("zmm3");
    register __m512 c20 asm("zmm4");
    register __m512 c21 asm("zmm5");
    register __m512 c30 asm("zmm6");
    register __m512 c31 asm("zmm7");
    register __m512 c40 asm("zmm8");
    register __m512 c41 asm("zmm9");
    register __m512 c50 asm("zmm10");
    register __m512 c51 asm("zmm11");
    register __m512 c60 asm("zmm12");
    register __m512 c61 asm("zmm13");
    register __m512 c70 asm("zmm14");
    register __m512 c71 asm("zmm15");
    register __m512 c80 asm("zmm16");
    register __m512 c81 asm("zmm17");
    register __m512 c90 asm("zmm18");
    register __m512 c91 asm("zmm19");
    register __m512 c100 asm("zmm20");
    register __m512 c101 asm("zmm21");
    register __m512 c110 asm("zmm22");
    register __m512 c111 asm("zmm23");

    c00 = _mm512_setzero_ps();
    c01 = _mm512_setzero_ps();
    c10 = _mm512_setzero_ps();
    c11 = _mm512_setzero_ps();
    c20 = _mm512_setzero_ps();
    c21 = _mm512_setzero_ps();
    c30 = _mm512_setzero_ps();
    c31 = _mm512_setzero_ps();
    c40 = _mm512_setzero_ps();
    c41 = _mm512_setzero_ps();
    c50 = _mm512_setzero_ps();
    c51 = _mm512_setzero_ps();
    c60 = _mm512_setzero_ps();
    c61 = _mm512_setzero_ps();
    c70 = _mm512_setzero_ps();
    c71 = _mm512_setzero_ps();
    c80 = _mm512_setzero_ps();
    c81 = _mm512_setzero_ps();
    c90 = _mm512_setzero_ps();
    c91 = _mm512_setzero_ps();
    c100 = _mm512_setzero_ps();
    c101 = _mm512_setzero_ps();
    c110 = _mm512_setzero_ps();
    c111 = _mm512_setzero_ps();

    const int unroll_factor = 4;

    uint8_t offsets[64];
    for (int i = 0; i < (panel_size + BITS - 1) / BITS; i++) {
        BIT_DTYPE rbm64 = rbm[i];
        BIT_DTYPE sbm64 = sbm[i];

        BIT_DTYPE and_bm = rbm64 & sbm64;

        int cnt = __builtin_popcountll(and_bm);

        if (!cnt) {
            continue;
        }

        float *R_ptr_start = &R[64 * i * R_slice_size];
        float *S_ptr_start = &S[64 * i * S_slice_size];

        if (cnt != BITS) {
            compute_offsets_and(and_bm, offsets, _8bit_indexes);
            const int stop = cnt - (cnt % unroll_factor);
            uint8_t *offset_ptr = offsets;
            for (int i = 0; i < stop; i += unroll_factor) {
                uint8_t next = *(offset_ptr++);
                float *R_ptr = R_ptr_start + next * R_slice_size;
                float *S_ptr = S_ptr_start + next * S_slice_size;
                iter_fp32();
                next = *(offset_ptr++);
                R_ptr = R_ptr_start + next * R_slice_size;
                S_ptr = S_ptr_start + next * S_slice_size;
                iter_fp32();
                next = *(offset_ptr++);
                R_ptr = R_ptr_start + next * R_slice_size;
                S_ptr = S_ptr_start + next * S_slice_size;
                iter_fp32();
                next = *(offset_ptr++);
                R_ptr = R_ptr_start + next * R_slice_size;
                S_ptr = S_ptr_start + next * S_slice_size;
                iter_fp32();
            }
            for (int i = stop; i < cnt; i++) {
                uint8_t next = *(offset_ptr++);
                float *R_ptr = R_ptr_start + next * R_slice_size;
                float *S_ptr = S_ptr_start + next * S_slice_size;
                iter_fp32();
            }
        } else {
            float *R_ptr = R_ptr_start;
            float *S_ptr = S_ptr_start;
            for (int i = 0; i < BITS; i += unroll_factor) {
                /*
        iter_cont(0);
        iter_cont(1);
        iter_cont(2);
        iter_cont(3);
        */
                iter_fp32();
                R_ptr += R_slice_size;
                S_ptr += S_slice_size;
                iter_fp32();
                R_ptr += R_slice_size;
                S_ptr += S_slice_size;
                iter_fp32();
                R_ptr += R_slice_size;
                S_ptr += S_slice_size;
                iter_fp32();
                R_ptr += R_slice_size;
                S_ptr += S_slice_size;
                /*
        double* R_ptr = R_ptr_start + i * R_slice_size;
        double* S_ptr = S_ptr_start + i * S_slice_size;
        iter();
        R_ptr = R_ptr_start + (i+1) * R_slice_size;
        S_ptr = S_ptr_start + (i+1) * S_slice_size;
        iter();
        R_ptr = R_ptr_start + (i+2) * R_slice_size;
        S_ptr = S_ptr_start + (i+2) * S_slice_size;
        iter();
        R_ptr = R_ptr_start + (i+3) * R_slice_size;
        S_ptr = S_ptr_start + (i+3) * S_slice_size;
        iter();
        */
            }
        }
    }

    if (!sparse_result) {
        if (R_slice_info.continuous && S_slice_info.continuous) {
            // if (false) {
            const int I = R_row_idxs[0];
            const int J = S_row_idxs[0];
            const int write_height = M - I;
            const int write_length = N - J;

            __mmask16 k0 = 0b1111111111111111;
            __mmask16 k1 = 0b1111111111111111;
            int shift = write_length >= 16 ? 0 : 16 - write_length;
            k0 >>= shift;
            shift = write_length >= 32 ? 0 : 32 - write_length;
            k1 >>= shift;

            float *result_p = &ret[I * N + J];

            if (R_slice_size <= write_height) {
                interleave_store_lo_fp32(0, 1);
                interleave_store_hi_fp32(0, 1);
                interleave_store_lo_fp32(2, 3);
                interleave_store_hi_fp32(2, 3);
                interleave_store_lo_fp32(4, 5);
                interleave_store_hi_fp32(4, 5);
                interleave_store_lo_fp32(6, 7);
                interleave_store_hi_fp32(6, 7);
                interleave_store_lo_fp32(8, 9);
                interleave_store_hi_fp32(8, 9);
                interleave_store_lo_fp32(10, 11);
                interleave_store_hi_fp32(10, 11);
            } else {
                switch (write_height) {
                case 12:
                    interleave_store_hi_fp32(10, 11);
                case 11:
                    interleave_store_lo_fp32(10, 11);
                case 10:
                    interleave_store_hi_fp32(8, 9);
                case 9:
                    interleave_store_lo_fp32(8, 9);
                case 8:
                    interleave_store_hi_fp32(6, 7);
                case 7:
                    interleave_store_lo_fp32(6, 7);
                case 6:
                    interleave_store_hi_fp32(4, 5);
                case 5:
                    interleave_store_lo_fp32(4, 5);
                case 4:
                    interleave_store_hi_fp32(2, 3);
                case 3:
                    interleave_store_lo_fp32(2, 3);
                case 2:
                    interleave_store_hi_fp32(0, 1);
                case 1:
                    interleave_store_lo_fp32(0, 1);
                }
            }
        } else {
            const int I = R_row_idxs[0];
            const int J = S_row_idxs[0];
            const int last_write_R = M - R_row_idxs[R_slice_info.len - 1] < R_sh
                                         ? M - R_row_idxs[R_slice_info.len - 1]
                                         : R_sh;
            const int write_height =
                R_sh * (R_slice_info.len - 1) + last_write_R;
            const int last_write_S = N - S_row_idxs[S_slice_info.len - 1] < S_sh
                                         ? N - S_row_idxs[S_slice_info.len - 1]
                                         : S_sh;
            const int write_length =
                S_sh * (S_slice_info.len - 1) + last_write_S;

            float *write_locations[12];
            for (int i = 0; i < R_slice_info.len; i++) {
                for (int j = 0; j < R_sh; j++) {
                    write_locations[i * R_sh + j] =
                        &ret[(R_row_idxs[i] + j) * N];
                }
            }

            int scatter_locations[32];
            for (int i = 0; i < S_slice_info.len; i++) {
                for (int j = 0; j < S_sh; j++) {
                    scatter_locations[i * S_sh + j] = S_row_idxs[i] + j;
                }
            }

            __m512i scatter_idx0 =
                _mm512_loadu_si512((__m512i *) scatter_locations);
            __m512i scatter_idx1 =
                _mm512_loadu_si512((__m512i *) (scatter_locations + 16));

            __mmask16 k0 = 0b1111111111111111;
            __mmask16 k1 = 0b1111111111111111;
            int shift = write_length >= 16 ? 0 : 16 - write_length;
            k0 >>= shift;
            shift = write_length >= 32 ? 0 : 32 - write_length;
            k1 >>= shift;

            switch (write_height) {
            case 12:
                interleave_store_hi_nocont_fp32(10, 11);
            case 11:
                interleave_store_lo_nocont_fp32(10, 11);
            case 10:
                interleave_store_hi_nocont_fp32(8, 9);
            case 9:
                interleave_store_lo_nocont_fp32(8, 9);
            case 8:
                interleave_store_hi_nocont_fp32(6, 7);
            case 7:
                interleave_store_lo_nocont_fp32(6, 7);
            case 6:
                interleave_store_hi_nocont_fp32(4, 5);
            case 5:
                interleave_store_lo_nocont_fp32(4, 5);
            case 4:
                interleave_store_hi_nocont_fp32(2, 3);
            case 3:
                interleave_store_lo_nocont_fp32(2, 3);
            case 2:
                interleave_store_hi_nocont_fp32(0, 1);
            case 1:
                interleave_store_lo_nocont_fp32(0, 1);
            }
        }
    } else {
    }
}
