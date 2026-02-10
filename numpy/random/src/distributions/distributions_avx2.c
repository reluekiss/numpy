/*
 * AVX2-optimized distribution generation for NumPy
 *
 * Uses SIMD-parallel xorshift256+ (4 lanes) for maximum throughput.
 * Processes 4 doubles at a time for uniform, normal, exponential, gamma.
 *
 * Copyright (c) 2025
 */

#include "numpy/random/distributions.h"
#include <immintrin.h>
#include <string.h>

#ifdef __AVX2__

/* ============================================================================
 * Types and Constants
 * ============================================================================ */

typedef __m256d d256;
typedef __m256i i256;
typedef __m128i i128;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* SIMD-parallel xorshift256+ state: 4 independent generators */
typedef struct {
    i256 s0, s1, s2, s3;
} rng256_state;

/* Thread-local AVX2 RNG state */
static __thread rng256_state avx2_rng_state;
static __thread int avx2_rng_initialized = 0;

/* ============================================================================
 * RNG State Management
 * ============================================================================ */

static inline uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static inline rng256_state rng256_init(uint64_t seed) {
    uint64_t sm = seed ? seed : 1;

    __attribute__((aligned(64))) uint64_t a[4], b[4], c[4], d[4];
    for (int i = 0; i < 4; ++i) a[i] = splitmix64_next(&sm);
    for (int i = 0; i < 4; ++i) b[i] = splitmix64_next(&sm);
    for (int i = 0; i < 4; ++i) c[i] = splitmix64_next(&sm);
    for (int i = 0; i < 4; ++i) d[i] = splitmix64_next(&sm);

    rng256_state st;
    st.s0 = _mm256_load_si256((const i256 *)a);
    st.s1 = _mm256_load_si256((const i256 *)b);
    st.s2 = _mm256_load_si256((const i256 *)c);
    st.s3 = _mm256_load_si256((const i256 *)d);
    return st;
}

/* Initialize from NumPy's bitgen state */
static void ensure_avx2_state(bitgen_t *bitgen_state) {
    if (!avx2_rng_initialized) {
        /* Seed from the bitgen's current state */
        uint64_t seed = bitgen_state->next_uint64(bitgen_state->state);
        avx2_rng_state = rng256_init(seed);
        avx2_rng_initialized = 1;
    }
}

/* Reseed AVX2 state (call when NumPy generator is reseeded) */
void random_avx2_reseed(bitgen_t *bitgen_state) {
    uint64_t seed = bitgen_state->next_uint64(bitgen_state->state);
    avx2_rng_state = rng256_init(seed);
    avx2_rng_initialized = 1;
}

/* ============================================================================
 * Core SIMD RNG - xorshift256+
 * ============================================================================ */

static inline i256 rng256_u64x4(rng256_state *st) {
    i256 result = _mm256_add_epi64(st->s0, st->s3);

    i256 t = _mm256_slli_epi64(st->s1, 17);

    st->s2 = _mm256_xor_si256(st->s2, st->s0);
    st->s3 = _mm256_xor_si256(st->s3, st->s1);
    st->s1 = _mm256_xor_si256(st->s1, st->s2);
    st->s0 = _mm256_xor_si256(st->s0, st->s3);

    st->s2 = _mm256_xor_si256(st->s2, t);

    st->s3 = _mm256_or_si256(
        _mm256_slli_epi64(st->s3, 45),
        _mm256_srli_epi64(st->s3, 19)
    );

    return result;
}

/* Convert 4 u64 to 4 doubles in [0, 1) */
static inline d256 rng256_uniform_d(rng256_state *st) {
    i256 r = rng256_u64x4(st);
    const i256 mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFull);
    const i256 one_exp = _mm256_set1_epi64x(0x3FF0000000000000ull);
    i256 mantissa = _mm256_and_si256(r, mantissa_mask);
    i256 as_double_bits = _mm256_or_si256(mantissa, one_exp);
    d256 result = _mm256_castsi256_pd(as_double_bits);
    return _mm256_sub_pd(result, _mm256_set1_pd(1.0));
}

/* ============================================================================
 * SIMD Math Functions
 * ============================================================================ */

/* Extract exponent helper */
static inline i128 extract_high32(i256 v) {
    __m128 lo = _mm_castsi128_ps(_mm256_castsi256_si128(v));
    __m128 hi = _mm_castsi128_ps(_mm256_extractf128_si256(v, 1));
    __m128 result_lo = _mm_shuffle_ps(lo, _mm_setzero_ps(), 0x0d);
    __m128 result_hi = _mm_shuffle_ps(_mm_setzero_ps(), hi, 0xd0);
    return _mm_castps_si128(_mm_or_ps(result_lo, result_hi));
}

static inline i128 vilogb2k(d256 d) {
    i256 di = _mm256_castpd_si256(d);
    i128 q = extract_high32(di);
    q = _mm_srli_epi32(q, 20);
    q = _mm_and_si128(q, _mm_set1_epi32(0x7ff));
    q = _mm_sub_epi32(q, _mm_set1_epi32(0x3ff));
    return q;
}

static inline d256 vldexp3(d256 d, i128 e) {
    i128 exp = _mm_add_epi32(e, _mm_set1_epi32(0x3ff));
    exp = _mm_slli_epi32(exp, 20);
    i256 exp64 = _mm256_cvtepi32_epi64(exp);
    exp64 = _mm256_slli_epi64(exp64, 32);
    return _mm256_mul_pd(d, _mm256_castsi256_pd(exp64));
}

/* Fast log (3.5 ulp accuracy) */
__attribute__((always_inline))
static inline d256 rng256_log(d256 d) {
    const d256 scale = _mm256_set1_pd(1.0 / 0.75);
    i128 e = vilogb2k(_mm256_mul_pd(d, scale));

    d256 m = vldexp3(d, _mm_sub_epi32(_mm_setzero_si128(), e));

    d256 x = _mm256_div_pd(
        _mm256_sub_pd(m, _mm256_set1_pd(1.0)),
        _mm256_add_pd(m, _mm256_set1_pd(1.0))
    );
    d256 x2 = _mm256_mul_pd(x, x);
    d256 x4 = _mm256_mul_pd(x2, x2);
    d256 x8 = _mm256_mul_pd(x4, x4);
    d256 x3 = _mm256_mul_pd(x, x2);

    d256 t;
    {
        d256 c0 = _mm256_set1_pd(0.153487338491425068243146);
        d256 c1 = _mm256_set1_pd(0.152519917006351951593857);
        d256 c2 = _mm256_set1_pd(0.181863266251982985677316);
        d256 c3 = _mm256_set1_pd(0.222221366518767365905163);
        d256 c4 = _mm256_set1_pd(0.285714294746548025383248);
        d256 c5 = _mm256_set1_pd(0.399999999950799600689777);
        d256 c6 = _mm256_set1_pd(0.6666666666667778740063);

        d256 t01 = _mm256_fmadd_pd(c0, x2, c1);
        d256 t23 = _mm256_fmadd_pd(c2, x2, c3);
        d256 t45 = _mm256_fmadd_pd(c4, x2, c5);
        d256 t0123 = _mm256_fmadd_pd(t01, x4, t23);
        d256 t4567 = _mm256_fmadd_pd(t45, x4, c6);
        t = _mm256_fmadd_pd(t0123, x8, t4567);
    }

    d256 e_dbl = _mm256_cvtepi32_pd(e);
    const d256 ln2 = _mm256_set1_pd(0.693147180559945286226764);
    d256 result = _mm256_fmadd_pd(x, _mm256_set1_pd(2.0), _mm256_mul_pd(ln2, e_dbl));
    result = _mm256_fmadd_pd(x3, t, result);

    return result;
}

/* Fast cos for |d| < ~15 */
static inline d256 rng256_cos(d256 d) {
    const d256 M_1_PI_v = _mm256_set1_pd(0.31830988618379067154);
    const double PI_A2 = 3.141592653589793116;
    const double PI_B2 = 1.2246467991473532072e-16;

    d256 dql = _mm256_fmadd_pd(
        _mm256_set1_pd(2.0),
        _mm256_round_pd(
            _mm256_fmadd_pd(d, M_1_PI_v, _mm256_set1_pd(-0.5)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
        ),
        _mm256_set1_pd(1.0)
    );

    i128 ql = _mm256_cvtpd_epi32(dql);

    d = _mm256_fmadd_pd(dql, _mm256_set1_pd(-PI_A2 * 0.5), d);
    d = _mm256_fmadd_pd(dql, _mm256_set1_pd(-PI_B2 * 0.5), d);

    d256 s = _mm256_mul_pd(d, d);

    i128 q_and_2 = _mm_and_si128(ql, _mm_set1_epi32(2));
    i128 sign_mask_lo = _mm_cmpeq_epi32(q_and_2, _mm_setzero_si128());
    i256 sign_expand = _mm256_cvtepi32_epi64(sign_mask_lo);
    d256 sign_bit = _mm256_and_pd(_mm256_castsi256_pd(sign_expand), _mm256_set1_pd(-0.0));
    d = _mm256_xor_pd(d, sign_bit);

    d256 s2 = _mm256_mul_pd(s, s);
    d256 s4 = _mm256_mul_pd(s2, s2);

    d256 c0 = _mm256_set1_pd(-7.97255955009037868891952e-18);
    d256 c1 = _mm256_set1_pd(2.81009972710863200091251e-15);
    d256 c2 = _mm256_set1_pd(-7.64712219118158833288484e-13);
    d256 c3 = _mm256_set1_pd(1.60590430605664501629054e-10);
    d256 c4 = _mm256_set1_pd(-2.50521083763502045810755e-08);
    d256 c5 = _mm256_set1_pd(2.75573192239198747630416e-06);
    d256 c6 = _mm256_set1_pd(-0.000198412698412696162806809);
    d256 c7 = _mm256_set1_pd(0.00833333333333332974823815);

    d256 t01 = _mm256_fmadd_pd(c0, s, c1);
    d256 t23 = _mm256_fmadd_pd(c2, s, c3);
    d256 t45 = _mm256_fmadd_pd(c4, s, c5);
    d256 t67 = _mm256_fmadd_pd(c6, s, c7);
    d256 t0123 = _mm256_fmadd_pd(t01, s2, t23);
    d256 t4567 = _mm256_fmadd_pd(t45, s2, t67);
    d256 u = _mm256_fmadd_pd(t0123, s4, t4567);

    u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(-0.166666666666666657414808));
    u = _mm256_fmadd_pd(_mm256_mul_pd(s, u), d, d);

    return u;
}

static inline d256 rng256_sin(d256 x) {
    return rng256_cos(_mm256_sub_pd(x, _mm256_set1_pd(M_PI / 2.0)));
}

/* ============================================================================
 * Distribution Implementations
 * ============================================================================ */

/* Exponential: -log(U) / lambda */
static inline d256 rng256_exponential_d(rng256_state *st, d256 lambda) {
    d256 u = rng256_uniform_d(st);
    const d256 umin = _mm256_set1_pd(0x1.0p-53);
    u = _mm256_max_pd(u, umin);

    d256 neg_log_u = _mm256_sub_pd(_mm256_setzero_pd(), rng256_log(u));
    return _mm256_div_pd(neg_log_u, lambda);
}

/* Box-Muller normal: produces 8 normals (2 vectors of 4) */
__attribute__((always_inline))
static inline void rng256_normal2_d(rng256_state *st, d256 mean, d256 stddev,
                                     d256 *out0, d256 *out1) {
    d256 u1 = rng256_uniform_d(st);
    d256 u2 = rng256_uniform_d(st);

    const d256 umin = _mm256_set1_pd(0x1.0p-53);
    u1 = _mm256_max_pd(u1, umin);

    d256 r = _mm256_sqrt_pd(
        _mm256_mul_pd(_mm256_set1_pd(-2.0), rng256_log(u1))
    );

    d256 theta = _mm256_mul_pd(_mm256_set1_pd(2.0 * M_PI), u2);

    d256 z0 = _mm256_mul_pd(r, rng256_cos(theta));
    d256 z1 = _mm256_mul_pd(r, rng256_sin(theta));

    *out0 = _mm256_fmadd_pd(z0, stddev, mean);
    *out1 = _mm256_fmadd_pd(z1, stddev, mean);
}

/* Gamma via sum of exponentials for integer shape */
static inline d256 rng256_gamma_int_d(rng256_state *st, int k, d256 scale) {
    d256 sum = _mm256_setzero_pd();
    d256 one = _mm256_set1_pd(1.0);
    for (int i = 0; i < k; i++) {
        sum = _mm256_add_pd(sum, rng256_exponential_d(st, one));
    }
    return _mm256_mul_pd(sum, scale);
}

/* Wilson-Hilferty approximation for large shape (α > 10) */
static inline d256 rng256_gamma_wilson_d(rng256_state *st, d256 shape, d256 scale) {
    d256 zero = _mm256_setzero_pd();
    d256 one = _mm256_set1_pd(1.0);
    d256 nine = _mm256_set1_pd(9.0);

    d256 z0, z1;
    rng256_normal2_d(st, zero, one, &z0, &z1);

    d256 nine_alpha = _mm256_mul_pd(nine, shape);
    d256 inv_9a = _mm256_div_pd(one, nine_alpha);
    d256 inv_sqrt_9a = _mm256_sqrt_pd(inv_9a);

    d256 base = _mm256_sub_pd(one, inv_9a);
    base = _mm256_fmadd_pd(z0, inv_sqrt_9a, base);
    base = _mm256_max_pd(base, _mm256_set1_pd(1e-10));

    d256 base3 = _mm256_mul_pd(_mm256_mul_pd(base, base), base);

    return _mm256_mul_pd(_mm256_mul_pd(shape, base3), scale);
}

/* Marsaglia-Tsang for general case */
static inline d256 rng256_gamma_mt_d(rng256_state *st, d256 shape, d256 scale) {
    d256 one = _mm256_set1_pd(1.0);
    d256 zero = _mm256_setzero_pd();

    d256 boost_mask = _mm256_cmp_pd(shape, one, _CMP_LT_OQ);
    d256 adjusted_shape = _mm256_blendv_pd(shape, _mm256_add_pd(shape, one), boost_mask);

    d256 one_third = _mm256_set1_pd(1.0 / 3.0);
    d256 d_param = _mm256_sub_pd(adjusted_shape, one_third);
    d256 c_param = _mm256_div_pd(one, _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(9.0), d_param)));

    d256 result = zero;
    d256 done = zero;

    d256 const_0_0331 = _mm256_set1_pd(0.0331);
    d256 const_0_5 = _mm256_set1_pd(0.5);

    for (int iter = 0; iter < 100 && _mm256_movemask_pd(done) != 0xF; iter++) {
        d256 n0, n1;
        rng256_normal2_d(st, zero, one, &n0, &n1);
        d256 x = n0;
        d256 u = rng256_uniform_d(st);

        d256 v = _mm256_fmadd_pd(c_param, x, one);
        d256 v_valid = _mm256_cmp_pd(v, zero, _CMP_GT_OQ);
        d256 v3 = _mm256_mul_pd(_mm256_mul_pd(v, v), v);
        d256 x2 = _mm256_mul_pd(x, x);
        d256 x4 = _mm256_mul_pd(x2, x2);

        d256 quick_thresh = _mm256_fnmadd_pd(const_0_0331, x4, one);
        d256 quick_accept = _mm256_cmp_pd(u, quick_thresh, _CMP_LT_OQ);

        d256 log_u = rng256_log(u);
        d256 log_v = rng256_log(v3);
        d256 slow_rhs = _mm256_fmadd_pd(const_0_5, x2,
                        _mm256_mul_pd(d_param,
                        _mm256_add_pd(_mm256_sub_pd(one, v3), log_v)));
        d256 slow_accept = _mm256_cmp_pd(log_u, slow_rhs, _CMP_LT_OQ);

        d256 accept = _mm256_and_pd(v_valid, _mm256_or_pd(quick_accept, slow_accept));
        d256 new_accept = _mm256_andnot_pd(done, accept);

        d256 candidate = _mm256_mul_pd(d_param, v3);
        result = _mm256_blendv_pd(result, candidate, new_accept);
        done = _mm256_or_pd(done, new_accept);
    }

    /* Handle shape < 1 boost */
    if (_mm256_movemask_pd(boost_mask)) {
        d256 u_boost = rng256_uniform_d(st);
        const d256 umin = _mm256_set1_pd(0x1.0p-53);
        u_boost = _mm256_max_pd(u_boost, umin);

        /* pow(u, 1/shape) - scalar fallback for simplicity */
        __attribute__((aligned(32))) double u_arr[4], shape_arr[4], result_arr[4];
        _mm256_store_pd(u_arr, u_boost);
        _mm256_store_pd(shape_arr, shape);
        _mm256_store_pd(result_arr, result);

        __attribute__((aligned(32))) double mask_arr[4];
        _mm256_store_pd(mask_arr, boost_mask);

        for (int i = 0; i < 4; i++) {
            if (((uint64_t*)mask_arr)[i]) {
                result_arr[i] *= pow(u_arr[i], 1.0 / shape_arr[i]);
            }
        }
        result = _mm256_load_pd(result_arr);
    }

    return _mm256_mul_pd(result, scale);
}

/* ============================================================================
 * Public API - NumPy Compatible Fill Functions
 * ============================================================================ */

void random_standard_uniform_fill_avx2(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    npy_intp i = 0;
    for (; i + 4 <= cnt; i += 4) {
        d256 v = rng256_uniform_d(st);
        _mm256_storeu_pd(out + i, v);
    }

    /* Handle remainder */
    if (i < cnt) {
        __attribute__((aligned(32))) double tmp[4];
        d256 v = rng256_uniform_d(st);
        _mm256_store_pd(tmp, v);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_standard_normal_fill_avx2(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;
    d256 zero = _mm256_setzero_pd();
    d256 one = _mm256_set1_pd(1.0);

    npy_intp i = 0;
    for (; i + 8 <= cnt; i += 8) {
        d256 z0, z1;
        rng256_normal2_d(st, zero, one, &z0, &z1);
        _mm256_storeu_pd(out + i, z0);
        _mm256_storeu_pd(out + i + 4, z1);
    }

    /* Handle remainder */
    if (i < cnt) {
        __attribute__((aligned(32))) double tmp[8];
        d256 z0, z1;
        rng256_normal2_d(st, zero, one, &z0, &z1);
        _mm256_store_pd(tmp, z0);
        _mm256_store_pd(tmp + 4, z1);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_standard_exponential_fill_avx2(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;
    d256 one = _mm256_set1_pd(1.0);

    npy_intp i = 0;
    for (; i + 4 <= cnt; i += 4) {
        d256 v = rng256_exponential_d(st, one);
        _mm256_storeu_pd(out + i, v);
    }

    /* Handle remainder */
    if (i < cnt) {
        __attribute__((aligned(32))) double tmp[4];
        d256 v = rng256_exponential_d(st, one);
        _mm256_store_pd(tmp, v);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_standard_gamma_fill_avx2(bitgen_t *bitgen_state, double shape,
                                      npy_intp cnt, double *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    d256 shape_v = _mm256_set1_pd(shape);
    d256 one = _mm256_set1_pd(1.0);

    /* Choose algorithm based on shape */
    int is_small_int = (shape == (double)(int)shape) && (shape >= 1.0) && (shape <= 10.0);
    int is_large = (shape > 10.0);

    npy_intp i = 0;

    if (is_small_int) {
        int k = (int)shape;
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_int_d(st, k, one);
            _mm256_storeu_pd(out + i, v);
        }
    } else if (is_large) {
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_wilson_d(st, shape_v, one);
            _mm256_storeu_pd(out + i, v);
        }
    } else {
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_mt_d(st, shape_v, one);
            _mm256_storeu_pd(out + i, v);
        }
    }

    /* Handle remainder with scalar fallback */
    for (; i < cnt; i++) {
        out[i] = random_standard_gamma(bitgen_state, shape);
    }
}

/* ============================================================================
 * Scaled Distribution Fills (double)
 * ============================================================================ */

void random_uniform_fill_avx2(bitgen_t *bitgen_state, double low, double range,
                               npy_intp cnt, double *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    d256 low_v = _mm256_set1_pd(low);
    d256 range_v = _mm256_set1_pd(range);

    npy_intp i = 0;
    for (; i + 4 <= cnt; i += 4) {
        d256 u = rng256_uniform_d(st);
        d256 v = _mm256_fmadd_pd(u, range_v, low_v);
        _mm256_storeu_pd(out + i, v);
    }

    if (i < cnt) {
        __attribute__((aligned(32))) double tmp[4];
        d256 u = rng256_uniform_d(st);
        d256 v = _mm256_fmadd_pd(u, range_v, low_v);
        _mm256_store_pd(tmp, v);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_normal_fill_avx2(bitgen_t *bitgen_state, double loc, double scale,
                              npy_intp cnt, double *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    d256 loc_v = _mm256_set1_pd(loc);
    d256 scale_v = _mm256_set1_pd(scale);

    npy_intp i = 0;
    for (; i + 8 <= cnt; i += 8) {
        d256 z0, z1;
        rng256_normal2_d(st, loc_v, scale_v, &z0, &z1);
        _mm256_storeu_pd(out + i, z0);
        _mm256_storeu_pd(out + i + 4, z1);
    }

    if (i < cnt) {
        __attribute__((aligned(32))) double tmp[8];
        d256 z0, z1;
        rng256_normal2_d(st, loc_v, scale_v, &z0, &z1);
        _mm256_store_pd(tmp, z0);
        _mm256_store_pd(tmp + 4, z1);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_exponential_fill_avx2(bitgen_t *bitgen_state, double scale,
                                   npy_intp cnt, double *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    d256 scale_v = _mm256_set1_pd(scale);
    d256 one = _mm256_set1_pd(1.0);

    npy_intp i = 0;
    for (; i + 4 <= cnt; i += 4) {
        d256 e = rng256_exponential_d(st, one);
        d256 v = _mm256_mul_pd(e, scale_v);
        _mm256_storeu_pd(out + i, v);
    }

    if (i < cnt) {
        __attribute__((aligned(32))) double tmp[4];
        d256 e = rng256_exponential_d(st, one);
        d256 v = _mm256_mul_pd(e, scale_v);
        _mm256_store_pd(tmp, v);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_gamma_fill_avx2(bitgen_t *bitgen_state, double shape, double scale,
                             npy_intp cnt, double *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    d256 shape_v = _mm256_set1_pd(shape);
    d256 scale_v = _mm256_set1_pd(scale);

    int is_small_int = (shape == (double)(int)shape) && (shape >= 1.0) && (shape <= 10.0);
    int is_large = (shape > 10.0);

    npy_intp i = 0;

    if (is_small_int) {
        int k = (int)shape;
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_int_d(st, k, scale_v);
            _mm256_storeu_pd(out + i, v);
        }
    } else if (is_large) {
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_wilson_d(st, shape_v, scale_v);
            _mm256_storeu_pd(out + i, v);
        }
    } else {
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_mt_d(st, shape_v, scale_v);
            _mm256_storeu_pd(out + i, v);
        }
    }

    for (; i < cnt; i++) {
        out[i] = scale * random_standard_gamma(bitgen_state, shape);
    }
}

/* ============================================================================
 * Float32 Primitives (8 floats per AVX2 register)
 * ============================================================================ */

typedef __m256 f256;

/* Convert 8 u64 to 8 floats in [0, 1) using same technique as double */
static inline f256 rng256_uniform_f(rng256_state *st) {
    /* Generate 8 u64s (two calls) */
    i256 r0 = rng256_u64x4(st);
    i256 r1 = rng256_u64x4(st);

    /* For float: use bits [40:63] (top 24 bits) as mantissa
     * Float format: 1 sign + 8 exp + 23 mantissa
     * We set exponent to 127 (1.0) and OR in 23 mantissa bits, then subtract 1.0 */
    const i256 mantissa_mask = _mm256_set1_epi64x(0x007FFFFF00000000ull);
    const i256 one_exp = _mm256_set1_epi64x(0x3F80000000000000ull);

    i256 m0 = _mm256_and_si256(r0, mantissa_mask);
    i256 m1 = _mm256_and_si256(r1, mantissa_mask);
    m0 = _mm256_or_si256(m0, one_exp);
    m1 = _mm256_or_si256(m1, one_exp);

    /* Extract high 32 bits of each 64-bit lane to get the floats */
    /* r0 has 4 floats in high dwords, r1 has 4 floats in high dwords */
    /* Shuffle to pack: [a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, d_hi, d_lo] -> [a_hi, b_hi, c_hi, d_hi] */
    const int shuffle_hi = 0xDD; /* 11 01 11 01 = pick dwords 3,1,3,1 from each 128-bit lane */
    __m128 f0_lo = _mm_castsi128_ps(_mm_shuffle_epi32(_mm256_castsi256_si128(m0), shuffle_hi));
    __m128 f0_hi = _mm_castsi128_ps(_mm_shuffle_epi32(_mm256_extracti128_si256(m0, 1), shuffle_hi));
    __m128 f1_lo = _mm_castsi128_ps(_mm_shuffle_epi32(_mm256_castsi256_si128(m1), shuffle_hi));
    __m128 f1_hi = _mm_castsi128_ps(_mm_shuffle_epi32(_mm256_extracti128_si256(m1, 1), shuffle_hi));

    /* Combine: f0_lo has 2 valid floats in low positions, f0_hi has 2, etc */
    __m128 combined0 = _mm_shuffle_ps(f0_lo, f0_hi, 0x44); /* 01 00 01 00 */
    __m128 combined1 = _mm_shuffle_ps(f1_lo, f1_hi, 0x44);

    f256 result = _mm256_setr_m128(combined0, combined1);

    /* Subtract 1.0 to get [0, 1) */
    return _mm256_sub_ps(result, _mm256_set1_ps(1.0f));
}

/* Fast float log (for exponential) - adapted from double version */
static inline f256 rng256_log_f(f256 d) {
    /* Same algorithm as double but with float precision */
    i256 di = _mm256_castps_si256(d);

    /* Extract exponent: shift right 23, mask, subtract bias */
    i256 exp_bits = _mm256_srli_epi32(di, 23);
    exp_bits = _mm256_and_si256(exp_bits, _mm256_set1_epi32(0xFF));
    exp_bits = _mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127));
    f256 e = _mm256_cvtepi32_ps(exp_bits);

    /* Extract mantissa, set exponent to 127 (1.0) */
    i256 mantissa = _mm256_and_si256(di, _mm256_set1_epi32(0x007FFFFF));
    mantissa = _mm256_or_si256(mantissa, _mm256_set1_epi32(0x3F800000));
    f256 m = _mm256_castsi256_ps(mantissa);

    /* log(m) for m in [1, 2) using x = (m-1)/(m+1), log(m) = 2*x + 2*x^3/3 + ... */
    f256 x = _mm256_div_ps(
        _mm256_sub_ps(m, _mm256_set1_ps(1.0f)),
        _mm256_add_ps(m, _mm256_set1_ps(1.0f))
    );
    f256 x2 = _mm256_mul_ps(x, x);

    /* Polynomial: 2 + 2/3*x^2 + 2/5*x^4 + 2/7*x^6 */
    f256 t = _mm256_set1_ps(0.2857143f);  /* 2/7 */
    t = _mm256_fmadd_ps(t, x2, _mm256_set1_ps(0.4f));  /* 2/5 */
    t = _mm256_fmadd_ps(t, x2, _mm256_set1_ps(0.6666667f));  /* 2/3 */
    t = _mm256_fmadd_ps(t, x2, _mm256_set1_ps(2.0f));

    f256 log_m = _mm256_mul_ps(x, t);

    /* log(d) = log(m) + e * ln(2) */
    const f256 ln2 = _mm256_set1_ps(0.6931472f);
    return _mm256_fmadd_ps(e, ln2, log_m);
}

/* Float exponential: -log(U) */
static inline f256 rng256_exponential_f(rng256_state *st) {
    f256 u = rng256_uniform_f(st);
    const f256 umin = _mm256_set1_ps(1.0f / 16777216.0f);
    u = _mm256_max_ps(u, umin);

    f256 neg_log_u = _mm256_sub_ps(_mm256_setzero_ps(), rng256_log_f(u));
    return neg_log_u;
}

/* Float cos - simple Taylor series, input should be in reasonable range */
static inline f256 rng256_cos_f(f256 x) {
    /* For Box-Muller we only need cos(2*pi*u) where u in [0,1)
     * So x is in [0, 2*pi). We use cos Taylor series directly. */
    const f256 twopi = _mm256_set1_ps(6.2831853f);

    /* Reduce to [-pi, pi] */
    x = _mm256_sub_ps(x, _mm256_mul_ps(
        _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.15915494f)),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        twopi));

    /* cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ... */
    f256 x2 = _mm256_mul_ps(x, x);
    f256 x4 = _mm256_mul_ps(x2, x2);

    /* Coefficients: 1, -1/2, 1/24, -1/720, 1/40320 */
    f256 c2 = _mm256_set1_ps(-0.5f);
    f256 c4 = _mm256_set1_ps(0.041666667f);   /* 1/24 */
    f256 c6 = _mm256_set1_ps(-0.001388889f);  /* -1/720 */
    f256 c8 = _mm256_set1_ps(0.0000248016f);  /* 1/40320 */

    f256 result = _mm256_set1_ps(1.0f);
    result = _mm256_fmadd_ps(c2, x2, result);
    result = _mm256_fmadd_ps(c4, x4, result);
    result = _mm256_fmadd_ps(c6, _mm256_mul_ps(x4, x2), result);
    result = _mm256_fmadd_ps(c8, _mm256_mul_ps(x4, x4), result);

    return result;
}

static inline f256 rng256_sin_f(f256 x) {
    /* sin(x) = cos(x - pi/2) */
    return rng256_cos_f(_mm256_sub_ps(x, _mm256_set1_ps(1.5707963f)));
}

/* Float Box-Muller: produces 16 normals (2 vectors of 8) */
static inline void rng256_normal2_f(rng256_state *st, f256 mean, f256 stddev,
                                     f256 *out0, f256 *out1) {
    f256 u1 = rng256_uniform_f(st);
    f256 u2 = rng256_uniform_f(st);

    const f256 umin = _mm256_set1_ps(1.0f / 16777216.0f);
    u1 = _mm256_max_ps(u1, umin);

    f256 r = _mm256_sqrt_ps(
        _mm256_mul_ps(_mm256_set1_ps(-2.0f), rng256_log_f(u1))
    );

    f256 theta = _mm256_mul_ps(_mm256_set1_ps(2.0f * (float)M_PI), u2);

    f256 z0 = _mm256_mul_ps(r, rng256_cos_f(theta));
    f256 z1 = _mm256_mul_ps(r, rng256_sin_f(theta));

    *out0 = _mm256_fmadd_ps(z0, stddev, mean);
    *out1 = _mm256_fmadd_ps(z1, stddev, mean);
}

/* ============================================================================
 * Float32 Public API
 * ============================================================================ */

void random_standard_uniform_fill_avx2_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    npy_intp i = 0;
    for (; i + 8 <= cnt; i += 8) {
        f256 v = rng256_uniform_f(st);
        _mm256_storeu_ps(out + i, v);
    }

    if (i < cnt) {
        __attribute__((aligned(32))) float tmp[8];
        f256 v = rng256_uniform_f(st);
        _mm256_store_ps(tmp, v);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_standard_normal_fill_avx2_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;
    f256 zero = _mm256_setzero_ps();
    f256 one = _mm256_set1_ps(1.0f);

    npy_intp i = 0;
    for (; i + 16 <= cnt; i += 16) {
        f256 z0, z1;
        rng256_normal2_f(st, zero, one, &z0, &z1);
        _mm256_storeu_ps(out + i, z0);
        _mm256_storeu_ps(out + i + 8, z1);
    }

    if (i < cnt) {
        __attribute__((aligned(32))) float tmp[16];
        f256 z0, z1;
        rng256_normal2_f(st, zero, one, &z0, &z1);
        _mm256_store_ps(tmp, z0);
        _mm256_store_ps(tmp + 8, z1);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_standard_exponential_fill_avx2_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) {
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    npy_intp i = 0;
    for (; i + 8 <= cnt; i += 8) {
        f256 v = rng256_exponential_f(st);
        _mm256_storeu_ps(out + i, v);
    }

    if (i < cnt) {
        __attribute__((aligned(32))) float tmp[8];
        f256 v = rng256_exponential_f(st);
        _mm256_store_ps(tmp, v);
        for (npy_intp j = 0; i < cnt; ++i, ++j) {
            out[i] = tmp[j];
        }
    }
}

void random_standard_gamma_fill_avx2_f(bitgen_t *bitgen_state, float shape,
                                        npy_intp cnt, float *out) {
    /* Float gamma: convert to double, compute, convert back */
    /* This is simpler and gamma is already compute-bound */
    ensure_avx2_state(bitgen_state);
    rng256_state *st = &avx2_rng_state;

    double dshape = (double)shape;
    d256 shape_v = _mm256_set1_pd(dshape);
    d256 one = _mm256_set1_pd(1.0);

    int is_small_int = (dshape == (double)(int)dshape) && (dshape >= 1.0) && (dshape <= 10.0);
    int is_large = (dshape > 10.0);

    npy_intp i = 0;

    /* Process 4 at a time, convert to float */
    __attribute__((aligned(32))) double dtmp[4];

    if (is_small_int) {
        int k = (int)dshape;
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_int_d(st, k, one);
            _mm256_store_pd(dtmp, v);
            out[i] = (float)dtmp[0];
            out[i+1] = (float)dtmp[1];
            out[i+2] = (float)dtmp[2];
            out[i+3] = (float)dtmp[3];
        }
    } else if (is_large) {
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_wilson_d(st, shape_v, one);
            _mm256_store_pd(dtmp, v);
            out[i] = (float)dtmp[0];
            out[i+1] = (float)dtmp[1];
            out[i+2] = (float)dtmp[2];
            out[i+3] = (float)dtmp[3];
        }
    } else {
        for (; i + 4 <= cnt; i += 4) {
            d256 v = rng256_gamma_mt_d(st, shape_v, one);
            _mm256_store_pd(dtmp, v);
            out[i] = (float)dtmp[0];
            out[i+1] = (float)dtmp[1];
            out[i+2] = (float)dtmp[2];
            out[i+3] = (float)dtmp[3];
        }
    }

    for (; i < cnt; i++) {
        out[i] = random_standard_gamma_f(bitgen_state, shape);
    }
}

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static void cpuid_ex(int out[4], int leaf, int subleaf) {
#if defined(_MSC_VER)
    __cpuidex(out, leaf, subleaf);
#elif defined(__GNUC__) || defined(__clang__)
    int a, b, c, d;
    __asm__ volatile("cpuid"
                     : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                     : "a"(leaf), "c"(subleaf));
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = d;
#else
    out[0] = out[1] = out[2] = out[3] = 0;
#endif
}

static uint64_t xgetbv0(void) {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#else
    return 0;
#endif
}

int random_avx2_available(void) {
#if !defined(__x86_64__) && !defined(__i386__) && !defined(_M_X64) && \
    !defined(_M_IX86)
    return 0;
#else
    int r[4];

    /* Check OSXSAVE + AVX in CPUID.(EAX=1):ECX */
    cpuid_ex(r, 1, 0);
    const int osxsave = (r[2] >> 27) & 1;
    const int avx = (r[2] >> 28) & 1;
    if (!osxsave || !avx) return 0;

    /* Check XCR0 enables XMM (bit 1) and YMM (bit 2) state */
    const uint64_t xcr0 = xgetbv0();
    if ((xcr0 & 0x6) != 0x6) return 0;

    /* Check AVX2 in CPUID.(EAX=7,ECX=0):EBX bit 5 */
    cpuid_ex(r, 7, 0);
    const int avx2 = (r[1] >> 5) & 1;
    return avx2 ? 1 : 0;
#endif
}

#else /* !__AVX2__ */

/* Stub implementations when AVX2 is not available */
void random_standard_uniform_fill_avx2(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
    random_standard_uniform_fill(bitgen_state, cnt, out);
}

void random_standard_normal_fill_avx2(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
    random_standard_normal_fill(bitgen_state, cnt, out);
}

void random_standard_exponential_fill_avx2(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
    random_standard_exponential_fill(bitgen_state, cnt, out);
}

void random_standard_gamma_fill_avx2(bitgen_t *bitgen_state, double shape,
                                      npy_intp cnt, double *out) {
    for (npy_intp i = 0; i < cnt; i++) {
        out[i] = random_standard_gamma(bitgen_state, shape);
    }
}

void random_uniform_fill_avx2(bitgen_t *bitgen_state, double low, double range,
                               npy_intp cnt, double *out) {
    for (npy_intp i = 0; i < cnt; i++) {
        out[i] = random_uniform(bitgen_state, low, range);
    }
}

void random_normal_fill_avx2(bitgen_t *bitgen_state, double loc, double scale,
                              npy_intp cnt, double *out) {
    for (npy_intp i = 0; i < cnt; i++) {
        out[i] = random_normal(bitgen_state, loc, scale);
    }
}

void random_exponential_fill_avx2(bitgen_t *bitgen_state, double scale,
                                   npy_intp cnt, double *out) {
    for (npy_intp i = 0; i < cnt; i++) {
        out[i] = random_exponential(bitgen_state, scale);
    }
}

void random_gamma_fill_avx2(bitgen_t *bitgen_state, double shape, double scale,
                             npy_intp cnt, double *out) {
    for (npy_intp i = 0; i < cnt; i++) {
        out[i] = random_gamma(bitgen_state, shape, scale);
    }
}

void random_standard_uniform_fill_avx2_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) {
    random_standard_uniform_fill_f(bitgen_state, cnt, out);
}

void random_standard_normal_fill_avx2_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) {
    random_standard_normal_fill_f(bitgen_state, cnt, out);
}

void random_standard_exponential_fill_avx2_f(bitgen_t *bitgen_state, npy_intp cnt, float *out) {
    random_standard_exponential_fill_f(bitgen_state, cnt, out);
}

void random_standard_gamma_fill_avx2_f(bitgen_t *bitgen_state, float shape,
                                        npy_intp cnt, float *out) {
    for (npy_intp i = 0; i < cnt; i++) {
        out[i] = random_standard_gamma_f(bitgen_state, shape);
    }
}

void random_avx2_reseed(bitgen_t *bitgen_state) {
    (void)bitgen_state;
}

int random_avx2_available(void) {
    return 0;
}

#endif /* __AVX2__ */
