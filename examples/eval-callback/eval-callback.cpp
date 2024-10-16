#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <random>
#include <string>
#include <tuple>
#include <vector>

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
//  */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) data + i);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) data + i;
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) data + i;
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) data + i;
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) data + i;
                    } else if (type == GGML_TYPE_I2) {
                        v = (float) *(int8_t *) data + i / 4;
                    } else {
                        GGML_ASSERT(false);
                    }
                    printf("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("                                      ],\n");
        }
        printf("                                     ]\n");
        printf("                                     sum = %f\n", sum);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    // const struct ggml_tensor * src0 = t->src[0];
    // const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }
    // char* temp = "result_output";
    // if (strcmp(t->name, temp) == 0){
    // char src1_str[128] = {0};
    // if (src1) {
    //     sprintf(src1_str, "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    // }

    // printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
    //        t->name, ggml_type_name(t->type), ggml_op_desc(t),
    //        src0->name, ggml_ne_string(src0).c_str(),
    //        src1 ? src1_str : "",
    //        ggml_ne_string(t).c_str());


    // // copy the data from the GPU memory if needed
    // const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    // if (!is_host) {
    //     auto n_bytes = ggml_nbytes(t);
    //     cb_data->data.resize(n_bytes);
    //     ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    // }

    // if (!ggml_is_quantized(t->type)) {
    //     uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
    //     ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
    // }  
                        const struct ggml_tensor * src0 = t->src[0];
                        const struct ggml_tensor * src1 = t->src[1];
                        printf("compute\n");
                        printf("%s\n", t->name);
                        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                            printf("%d ", t->ne[i]);
                            if (i + 1 < GGML_MAX_DIMS) {
                                printf(",");
                            }
                        }    
                        printf("\n");        
                        uint8_t * node_data = (uint8_t *) t->data;
                        ggml_print_tensor(node_data, t->type, t->ne, t->nb, 3);
                        // printf("%s: %24s = (%s) %10s\n", __func__,
                        //     node->name, ggml_type_name(node->type), ggml_op_desc(node));    
                        if (src0){
                            printf("has src0\n");
                            printf("%s\n", src0->name);    
                            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                                printf("%d ", src0->ne[i]);
                                if (i + 1 < GGML_MAX_DIMS) {
                                printf(",");
                                }
                            }                         
                            printf("\n");    
                            uint8_t * src0_data = (uint8_t *) src0->data;
                            ggml_print_tensor(src0_data, src0->type, src0->ne, src0->nb, 3);   
                        }else{
                            printf("no src0\n");
                        }

                        if(src1){
                            printf("has src1\n");
                            printf("%s\n", src1->name);    
                            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                                printf("%d ", src1->ne[i]);
                                if (i + 1 < GGML_MAX_DIMS) {
                                    printf(",");
                                }
                            }   
                            printf("\n");             
                            uint8_t * src1_data = (uint8_t *) src1->data;
                            ggml_print_tensor(src1_data, src1->type, src1->ne, src1->nb, 3);                  
                        }else{
                            printf("no src1\n");
                        }
      
    // }



    return true;
}

static bool run(llama_context * ctx, const gpt_params & params) {
    const bool add_bos = true;
    if (add_bos){
        printf("add\n");
    }
    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);
    printf("tokenize_old\n");
    for (int i=0; i<tokens.size(); i++){
        printf("%d ", tokens[i]);
    }
    printf("\n");
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}
// struct callback_data {
//     std::vector<uint8_t> data;
// };

// static std::string ggml_ne_string(const ggml_tensor * t) {
//     std::string str;
//     for (int i = 0; i < GGML_MAX_DIMS; ++i) {
//         str += std::to_string(t->ne[i]);
//         if (i + 1 < GGML_MAX_DIMS) {
//             str += ", ";
//         }
//     }
//     return str;
// }

// static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
//     GGML_ASSERT(n > 0);
//     float sum = 0;
//     for (int64_t i3 = 0; i3 < ne[3]; i3++) {
//         printf("                                     [\n");
//         for (int64_t i2 = 0; i2 < ne[2]; i2++) {
//             if (i2 == n && ne[2] > 2*n) {
//                 printf("                                      ..., \n");
//                 i2 = ne[2] - n;
//             }
//             printf("                                      [\n");
//             for (int64_t i1 = 0; i1 < ne[1]; i1++) {
//                 if (i1 == n && ne[1] > 2*n) {
//                     printf("                                       ..., \n");
//                     i1 = ne[1] - n;
//                 }
//                 printf("                                       [");
//                 for (int64_t i0 = 0; i0 < ne[0]; i0++) {
//                     if (i0 == n && ne[0] > 2*n) {
//                         printf("..., ");
//                         i0 = ne[0] - n;
//                     }
//                     size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
//                     float v;
//                     if (type == GGML_TYPE_F16) {
//                         v = ggml_fp16_to_fp32(*(ggml_fp16_t *) data + i);
//                     } else if (type == GGML_TYPE_F32) {
//                         v = *(float *) data + i;
//                     } else if (type == GGML_TYPE_I32) {
//                         v = (float) *(int32_t *) data + i;
//                     } else if (type == GGML_TYPE_I16) {
//                         v = (float) *(int16_t *) data + i;
//                     } else if (type == GGML_TYPE_I8) {
//                         v = (float) *(int8_t *) data + i;
//                     } else {
//                         GGML_ASSERT(false);
//                     }
//                     printf("%12.4f", v);
//                     sum += v;
//                     if (i0 < ne[0] - 1) printf(", ");
//                 }
//                 printf("],\n");
//             }
//             printf("                                      ],\n");
//         }
//         printf("                                     ]\n");
//         printf("                                     sum = %f\n", sum);
//     }
// }

// static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
//     auto * cb_data = (callback_data *) user_data;

//     const struct ggml_tensor * src0 = t->src[0];
//     const struct ggml_tensor * src1 = t->src[1];

//     if (ask) {
//         return true; // Always retrieve data
//     }

//     char src1_str[128] = {0};
//     if (src1) {
//         sprintf(src1_str, "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
//     }

//     printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
//            t->name, ggml_type_name(t->type), ggml_op_desc(t),
//            src0->name, ggml_ne_string(src0).c_str(),
//            src1 ? src1_str : "",
//            ggml_ne_string(t).c_str());


//     // copy the data from the GPU memory if needed
//     const bool is_host = ggml_backend_buffer_is_host(t->buffer);

//     if (!is_host) {
//         auto n_bytes = ggml_nbytes(t);
//         cb_data->data.resize(n_bytes);
//         ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
//     }

//     if (!ggml_is_quantized(t->type)) {
//         uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
//         ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
//     }

//     return true;
// }

// static bool run(llama_context * ctx, const gpt_params & params) {
//     const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

//     std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);

//     if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
//         fprintf(stderr, "%s : failed to eval\n", __func__);
//         return false;
//     }

//     return true;
// }


int main(int argc, char ** argv) {

    callback_data cb_data;

    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    print_build_info();

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    llama_model * model;
    llama_context * ctx;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", get_system_info(params).c_str());
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
