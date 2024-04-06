#include "ggml-tmac.h"
#include <vector>

static bool initialized = false;

static TMAC::TMACGeMMWrapper<float> * wrapper;

static std::vector<struct tmac_tensor_extra> tmac_tensor_extras;

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
    if (initialized) {
        return;
    }
    initialized = true;

    wrapper = new TMAC::TMACGeMMWrapper<float>();
    tmac_tensor_extras.clear();
}

void ggml_tmac_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete wrapper;
    for (auto & tmac_tensor_extra : tmac_tensor_extras) {
        aligned_free(tmac_tensor_extra.qweights);
        aligned_free(tmac_tensor_extra.scales);
    }
    tmac_tensor_extras.clear();
}

static bool is_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_0) {
        return true;
    } else {
        return false;
    }
}

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_CPU) {
        return true;
    }
    return false;
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
    // TODO: load from config
    const int bits = ggml_tmac_get_type_bits(tensor->type);
    const int g = 4;

    const uint8_t * w = tensor->data;

    const int k = tensor->ne[0];
    const int m = tensor->ne[1] * bits;

    TMAC::TMACGeMMConfig kcfg = wrapper->get_kcfg(1, k, m, bits);
    const int bm              = kcfg.bm;
    const int simd_n_in       = kcfg.simd_n_in;
    const int simd_n_out      = kcfg.simd_n_out;
    const int kfactor         = kcfg.kfactor;
    const int group_size      = kcfg.group_size;  // could be different from block size in llama.cpp
    const int lut_scales_size = kcfg.lut_scales_size;
    const int scales_size     = kcfg.scales_size;
    const int n_tile_num      = kcfg.n_tile_num;

    uint8_t * qweights = (uint8_t *) aligned_malloc(k * m / 8);
    float * scales = (float *) aligned_malloc(scales_size);
    tmac_tensor_extras.push_back((struct tmac_tensor_extra) {
        /* .lut_scales_size = */ lut_scales_size,
        /* .scales_size     = */ scales_size,
        /* .n_tile_num      = */ n_tile_num,
        /* .qweights        = */ qweights,
        /* .scales          = */ scales
    });
    tensor->extra = &tmac_tensor_extras[tmac_tensor_extras.size() - 1];
}

int ggml_tmac_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
            return 2;
        case GGML_TYPE_Q3_K:
            return 3;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}
