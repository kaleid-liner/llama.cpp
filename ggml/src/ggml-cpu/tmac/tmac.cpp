
#include <algorithm>
#include <string>

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-traits.h"
#include "lut_mul_mat.h"
#include "tmac.h"

#define GGML_USE_TMAC
#if defined(GGML_USE_TMAC)
namespace ggml::cpu::tmac {

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}


class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // auto is_contiguous = [](const struct ggml_tensor * t) {
        //     return ggml_is_contiguous(t);
        // };

        const struct ggml_tensor * src0 = op->src[0];
        const struct ggml_tensor * src1 = op->src[1];
        if (op->op == GGML_OP_MUL_MAT &&
            // ggml_is_contiguous(src0) &&         // src0 must be contiguous
            // ggml_is_contiguous(src1) &&         // src1 must be contiguous
            // op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_tmac_buffer_type() &&
            ggml_tmac_can_mul_mat(src0, src1, op)) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {    // src1 must be host buffer
                return false;
            }
            return true;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_tmac_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;
    }
};

}  // namespace ggml::cpu::tmac


bool ggml_tmac_init(const char * fname) {
    tmac_init();

    std::string tmac_meta_fname(fname);
    std::string new_fname_part("tmac_meta.json");
    std::replace(tmac_meta_fname.begin(), tmac_meta_fname.end(), '\\', '/');
    size_t lastSlashPos = tmac_meta_fname.find_last_of('/');
    if (lastSlashPos == std::string::npos) {
        tmac_meta_fname = new_fname_part;  // Only the new file name, no directory to append
    } else {
        tmac_meta_fname = tmac_meta_fname.substr(0, lastSlashPos).append("/" + new_fname_part);
    }

    GGML_LOG_INFO("%s: loading TMAC meta data from %s\n", __func__, tmac_meta_fname.c_str());
    if (!load_and_parse_tmac_meta(tmac_meta_fname.c_str())) {
        GGML_LOG_WARN("%s: failed to load TMAC meta data from %s. This has no effect on non-TMAC model running, but will lead to errors on T-MAC models.\n", __func__, tmac_meta_fname.c_str());
        return false;
    }
    return true;
}


static void ggml_backend_tmac_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, buffer->size);
}

static void * ggml_backend_tmac_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static enum ggml_status ggml_backend_tmac_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::tmac::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_tmac_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_tmac_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    if (is_type_supported(tensor->type)) {
        GGML_LOG_DEBUG("%s: tmac repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_tmac_convert_weight(tensor, data, offset, size);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

static void ggml_backend_tmac_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}


static ggml_backend_buffer_i ggml_backend_tmac_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_tmac_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_tmac_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_tmac_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_tmac_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_tmac_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_tmac_buffer_clear,
    /* .reset           = */ nullptr,
};


// T-MAC backend buffer type
static const char * ggml_backend_tmac_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_tmac_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_tmac_buffer_interface, data, size);
}

static size_t ggml_backend_tmac_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_tmac_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_tmac_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    // T-MAC version of ggml_nbytes
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    size_t nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    if(tensor->type == GGML_TYPE_I1 ||
       tensor->type == GGML_TYPE_I2 ||
       tensor->type == GGML_TYPE_I3 ||
       tensor->type == GGML_TYPE_I4){
        nbytes = ggml_tmac_get_nbytes(tensor);
    }
    return nbytes;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_tmac_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_tmac = {
        /* .iface = */ {
                        /* .get_name         = */ ggml_backend_tmac_buffer_type_get_name,
                        /* .alloc_buffer     = */ ggml_backend_tmac_buffer_type_alloc_buffer,
                        /* .get_alignment    = */ ggml_backend_tmac_buffer_type_get_alignment,
                        /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size   = */ ggml_backend_tmac_buffer_type_get_alloc_size,
                        /* .is_host          = */ ggml_backend_tmac_buffer_type_is_host,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::tmac::extra_buffer_type(),
    };

    return &ggml_backend_buffer_type_tmac;
}

#endif // GGML_USE_TMAC