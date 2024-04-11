#include <vector>
#include <type_traits>

#include "ggml-tmac.h"
#include "ggml-quants.h"

#include "t-mac/tmac_gemm_wrapper.h"

#define GGML_TMAC_MAX_NODES 8192

static bool initialized = false;

static TMAC::TMACGeMMWrapper<float> * wrapper = nullptr;

static tmac_tensor_extra * tmac_tensor_extras = nullptr;

static size_t tmac_tensor_extras_index = 0;

static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, TMAC::kAllocAlignment);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, TMAC::kAllocAlignment, size);
    return ptr;
#endif
}

static void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void ggml_tmac_init(void) {
    LOG(INFO) << "ggml_tmac_init";

    if (initialized) {
        return;
    }
    initialized = true;

    if (wrapper == nullptr) {
        wrapper = new TMAC::TMACGeMMWrapper<float>();
    }
    if (tmac_tensor_extras == nullptr) {
        tmac_tensor_extras = new tmac_tensor_extra[GGML_TMAC_MAX_NODES];
    }
    tmac_tensor_extras_index = 0;
}

void ggml_tmac_free(void) {
    LOG(INFO) << "ggml_tmac_free";

    if (!initialized) {
        return;
    }
    initialized = false;

    delete wrapper;
    wrapper = nullptr;
    for (size_t i = 0; i < tmac_tensor_extras_index; i++) {
        aligned_free(tmac_tensor_extras[i].qweights);
        aligned_free(tmac_tensor_extras[i].scales);
    }
    delete[] tmac_tensor_extras;
    tmac_tensor_extras = nullptr;
}

static bool is_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ3_XXS) {
        return true;
    } else {
        return false;
    }
}

template <int Bits, int SIMD_LEN = 16>
struct BlockQTypeAccessor {
    using block_t = std::conditional_t<Bits == 2, block_iq2_xxs, std::conditional_t<Bits == 3, block_iq3_xxs, block_q4_0>>;

    static constexpr int group_size = (sizeof(block_t) - sizeof(ggml_fp16_t)) * 8 / Bits;

    static constexpr int simd_n_elem = SIMD_LEN * 8 / Bits;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs = (const uint8_t *) ((((const block_t *) data)[idx / group_size]).qs);
        int internal_idx = idx % group_size;
        const uint8_t * simd_qs = qs + internal_idx / simd_n_elem * SIMD_LEN;
        int simd_idx = internal_idx % simd_n_elem;
        return simd_qs[simd_idx % SIMD_LEN] >> (simd_idx / SIMD_LEN * Bits);
    }

    static float get_scale(const void * data, int idx) {
        return ggml_fp16_to_fp32(((const block_t *) data)[idx / group_size].d);
    }
};

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_CPU) {
        return true;
    }
    return false;
}

size_t ggml_tmac_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    const int bits = ggml_tmac_get_type_bits(src0->type);

    TMAC::TMACGeMMConfig kcfg = wrapper->get_kcfg(ne01, ne10, 1, bits);

    return ne10 * ne11 * 4 * sizeof(int8_t) + kcfg.lut_scales_size * ne11 * 2 * sizeof(float);
}

// m = batch_size
// n = output_dim
void ggml_tmac_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k, int m, int bits) {
    // t-mac llama.cpp n and m swapped
    wrapper->llama_cpp_init(src1, qlut, lut_scales, lut_biases, n, k, m, bits);
}

void ggml_tmac_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits) {
    wrapper->llama_cpp_compute(src0, scales, qlut, lut_scales, lut_biases, dst, n, k, m, bits);
}

void ggml_tmac_transform_tensor(struct ggml_tensor * tensor) {
    if (!is_type_supported(tensor->type)) {
        return;
    }

    const int bits = ggml_tmac_get_type_bits(tensor->type);
    const int g = 4;
    const int ngroups_per_elem = 2;

    int k = tensor->ne[0];
    int m = tensor->ne[1];  // `n` in llama.cpp

    TMAC::TMACGeMMConfig kcfg = wrapper->get_kcfg(m, k, 1, bits);
    const int bm              = kcfg.bm;
    const int simd_n_in       = kcfg.simd_n_in;
    const int simd_n_out      = kcfg.simd_n_out;
    const int kfactor         = kcfg.kfactor;
    const int group_size      = kcfg.group_size;  // could be different from block size in llama.cpp
    const int lut_scales_size = kcfg.lut_scales_size;
    const int scales_size     = kcfg.scales_size;
    const int n_tile_num      = kcfg.n_tile_num;
    LOG(INFO) << "Transforming tensor: " << tensor->name << " (m: " << m << ", k: " << k << ", bits: " << bits << ")";
    LOG(INFO) << "kcfg (bm=" << bm << ", simd_n_in=" << simd_n_in << ", simd_n_out=" << simd_n_out << ", kfactor=" << kfactor
              << ", group_size=" << group_size << ", lut_scales_size=" << lut_scales_size << ", scales_size=" << scales_size << ", n_tile_num=" << n_tile_num << ")";

    const int mgroup = ngroups_per_elem * simd_n_in;
    m = m * bits;

    uint8_t * qweights = (uint8_t *) aligned_malloc(k * m / 8);
    float * scales = (float *) aligned_malloc(scales_size * sizeof(float));
    tensor->extra = tmac_tensor_extras + tmac_tensor_extras_index;
    tmac_tensor_extras[tmac_tensor_extras_index++] = {
        /* .lut_scales_size = */ lut_scales_size,
        /* .scales_size     = */ scales_size,
        /* .n_tile_num      = */ n_tile_num,
        /* .qweights        = */ qweights,
        /* .scales          = */ scales
    };

// for fast testing
#define TMAC_EMPTY_WEIGHTS
#ifndef TMAC_EMPTY_WEIGHTS
    // TODO: optimize to accelerate weights loading
    uint8_t * buf1 = (uint8_t *) malloc(m * k);
    uint8_t * buf2 = (uint8_t *) malloc(m * k / g);

    // # (M // bits, K, bits)
    // w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    for (int im = 0; im < m / bits; im++) {
        for (int ik = 0; ik < k; ik++) {
            for (int ib = 0; ib < bits; ib++) {
                uint8_t v;
                if (bits == 2) {
                    v = BlockQTypeAccessor<2>::get_q(tensor->data, im * k + ik);
                } else if (bits == 3) {
                    v = BlockQTypeAccessor<3>::get_q(tensor->data, im * k + ik);
                } else if (bits == 4) {
                    v = BlockQTypeAccessor<4>::get_q(tensor->data, im * k + ik);
                }
                buf1[im * k * bits + ik * bits + bits] = (v >> ib) & 1;
            }
        }
    }

    // # (M // bits, K, bits) -> (M // bits, bits, K) -> (M // bits, bits, K // g, g) -> (M // bits, bits, K // g)
    // w = w.transpose(0, 2, 1).reshape(M // bits, bits, K // g, g)
    // w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
    memset(buf2, 0, m * k / g);
    for (int im = 0; im < m / bits; im++) {
        for (int ik = 0; ik < k; ik++) {
            for (int ib = 0; ib < bits; ib++) {
                int new_im = im;
                int new_ib = ib;
                int new_ik = ik / g;
                int new_ig = ik % g;
                buf2[new_im * bits * k / g + new_ib * k / g + new_ik] += buf1[im * k * bits + ik * bits + bits] << new_ig;
            }
        }
    }

    // # 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
    // # for bits=3
    // # bit0: [0, 8), bit1: [8, 16), bit2: [16, 24), bit0: [24, 32)
    // # (M // bits // simd_n_float16, bits, simd_n_float16, K // g)
    // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
    // mgroup = ngroups_per_elem * simd_n_in
    // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
    // #             0        1             2             3                 4                  5
    // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
    // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
    memset(qweights, 0, m * k / g / ngroups_per_elem);
    for (int im = 0; im < m / bits; im++) {
        for (int ib = 0; ib < bits; ib++) {
            for (int ik = 0; ik < k / g; ik++) {
                int new_im = im / simd_n_out;
                int new_isno = im % simd_n_out;
                int new_ib = ib;
                int new_ik = ik;
                // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
                int new_idx = new_im * bits * simd_n_out * k / g + new_ib * simd_n_out * k / g + new_isno * k / g + new_ik;
                // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
                int nb2 = k / g;
                int nb1 = simd_n_in * nb2;
                int nb0 = ngroups_per_elem * nb1;
                new_im = new_idx / nb0;
                int new_ing = (new_idx % nb0) / nb1;
                int new_isni = (new_idx % nb1) / nb2;
                new_ik = (new_idx % nb2);
                new_idx = new_im * ngroups_per_elem * simd_n_in * k / g + new_isni * ngroups_per_elem * k / g + new_ing * k / g + new_ik;
                // #             0        1             2             3                 4                  5
                // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
                int nb4 = kfactor;
                int nb3 = k / g / kfactor * nb4;
                nb2 = ngroups_per_elem * nb3;
                nb1 = simd_n_in * nb2;
                nb0 = bm / mgroup * nb1;
                new_im = new_idx / nb0;
                int new_ibm = (new_idx % nb0) / nb1;
                new_isni = (new_idx % nb1) / nb2;
                new_ing = (new_idx % nb2) / nb3;
                new_ik = (new_idx % nb3) / nb4;
                int new_ikf = (new_idx % nb4);
                new_idx = new_im * k / g / kfactor * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
                          new_ik * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
                          new_ibm * kfactor * simd_n_in * ngroups_per_elem +
                          new_ikf * simd_n_in * ngroups_per_elem +
                          new_isni * ngroups_per_elem +
                          new_ing;
                new_idx = new_idx / ngroups_per_elem;
                // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
                qweights[new_idx] += buf2[im * bits * k / g + ib * k / g + ik] << (new_ing * g);
            }
        }
    }

    int m_group_size, k_group_size;
    if (scales_size < m / bits) {  // BitNet-like scale (m_groups,)
        m_group_size = m / bits / scales_size;
        k_group_size = k;
    } else {  // GPTQ-like scale (m / bits, k / group_size)
        GGML_ASSERT(scales_size == m / bits * k / group_size);
        m_group_size = 1;
        k_group_size = group_size;
    }

    // scales = scales.reshape(M // bm, bm // bits, K // group_size).transpose(0, 2, 1)
    for (int im = 0; im < m / bits; im += m_group_size) {
        for (int ik = 0; ik < k; ik += k_group_size) {
            float scale;
            int idx = im * k + ik;
            if (bits == 2) {
                scale = BlockQTypeAccessor<2>::get_scale(tensor->data, idx);
            } else if (bits == 3) {
                scale = BlockQTypeAccessor<3>::get_scale(tensor->data, idx);
            } else if (bits == 4) {
                scale = BlockQTypeAccessor<4>::get_scale(tensor->data, idx);
            }
            int new_idx;
            if (scales_size < m / bits) {
                new_idx = im / m_group_size;
            } else {
                idx = idx / group_size;
                int new_im = idx / (bm / bits * k / group_size);
                int new_ibm = (idx % (bm / bits * k / group_size)) / (k / group_size);
                int new_ik = (idx % (k / group_size));
                new_idx = new_im * k / group_size * bm / bits + new_ik * bm / bits + new_ibm;
            }
            scales[new_idx] = scale;
        }
    }

    free(buf1);
    free(buf2);
#else
    memset(qweights, 0, k * m / 8);
    memset(scales, 0, scales_size * sizeof(float));
#endif
}

int ggml_tmac_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
            return 2;
        case GGML_TYPE_Q2_K:
            return 2;
        case GGML_TYPE_IQ3_XXS:
            return 3;
        case GGML_TYPE_Q3_K:
            return 3;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}
