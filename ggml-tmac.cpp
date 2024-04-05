#include "ggml-tmac.h"

static TMAC::TMACGeMMWrapper<float> * wrapper;

void ggml_tmac_init(void) {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;

    wrapper = new TMAC::TMACGeMMWrapper<float>();
}

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    return true;
}

// m = batch_size
// n = output_dim
void ggml_tmac_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k, int m, int bits) {
    // t-mac llama.cpp n and m swapped
    wrapper->llama_cpp_init(src1, qlut, lut_scales, lut_biases, n, k, m, bits);
}

bool ggml_tmac_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits) {
    wrapper->llama_cpp_compute(src0, scales, qlut, lut_scales, lut_biases, dst, n, k, m, bits);
}
