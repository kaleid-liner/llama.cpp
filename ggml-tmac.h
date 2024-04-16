#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct tmac_tensor_extra {
    int lut_scales_size;
    int scales_size;
    int n_tile_num;
    uint8_t * qweights;
    float * scales;
};

GGML_API void ggml_tmac_init(void);
GGML_API void ggml_tmac_free(void);
// src0->type == Q4_0/IQ2_XXS/IQ3_XXS
// T-MAC currently only supports BitNet quantization or GPTQ-like quantization (only scales, without zeros)
// If use i-quantization gguf models, the results will be wrong
// TODO: add customized block types Q2_0/Q3_0
GGML_API bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API size_t ggml_tmac_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API void ggml_tmac_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k, int m, int bits);
GGML_API void ggml_tmac_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits);
GGML_API void ggml_tmac_transform_tensor(struct ggml_tensor * tensor);
GGML_API int ggml_tmac_get_type_bits(enum ggml_type type);
GGML_API void ggml_tmac_set_n_threads(int n_threads);

#ifdef  __cplusplus
}
#endif
