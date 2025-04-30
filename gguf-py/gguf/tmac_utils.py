import json
import numpy as np
import os
from pathlib import Path
import sys
from typing import Optional, Tuple


if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf


def is_tmac_w2_ftype(ftype: gguf.LlamaFileType):
    return ftype == gguf.LlamaFileType.MOSTLY_TMAC_BN_0 or \
           ftype == gguf.LlamaFileType.MOSTLY_TMAC_W2G64_0 or \
           ftype == gguf.LlamaFileType.MOSTLY_TMAC_W2G64_1 or \
           ftype == gguf.LlamaFileType.MOSTLY_TMAC_W2G128_0 or \
           ftype == gguf.LlamaFileType.MOSTLY_TMAC_W2G128_1

def is_tmac_w4_ftype(ftype: gguf.LlamaFileType):
    return ftype == gguf.LlamaFileType.MOSTLY_TMAC_W4G64_0 or \
           ftype == gguf.LlamaFileType.MOSTLY_TMAC_W4G64_1 or \
           ftype == gguf.LlamaFileType.MOSTLY_TMAC_W4G128_0 or \
           ftype == gguf.LlamaFileType.MOSTLY_TMAC_W4G128_1

def is_tmac_ftype(ftype: gguf.LlamaFileType):
    return is_tmac_w2_ftype(ftype) or is_tmac_w4_ftype(ftype)

def is_tmac_w2_dtype(dtype: gguf.GGMLQuantizationType):
    return dtype == gguf.GGMLQuantizationType.TMAC_BN_0 or \
           dtype == gguf.GGMLQuantizationType.TMAC_W2G64_0 or \
           dtype == gguf.GGMLQuantizationType.TMAC_W2G64_1 or \
           dtype == gguf.GGMLQuantizationType.TMAC_W2G128_0 or \
           dtype == gguf.GGMLQuantizationType.TMAC_W2G128_1

def is_tmac_w4_dtype(dtype: gguf.GGMLQuantizationType):
    return dtype == gguf.GGMLQuantizationType.TMAC_W4G64_0 or \
           dtype == gguf.GGMLQuantizationType.TMAC_W4G64_1 or \
           dtype == gguf.GGMLQuantizationType.TMAC_W4G128_0 or \
           dtype == gguf.GGMLQuantizationType.TMAC_W4G128_1

def is_tmac_dtype(dtype: gguf.GGMLQuantizationType):
    return is_tmac_w2_dtype(dtype) or is_tmac_w4_dtype(dtype)


def parse_gptqv2(qweight: np.ndarray, scales: np.ndarray, qzeros: np.ndarray) -> Tuple:
    bits = 32 // (scales.shape[1] // qzeros.shape[1])
    K = qweight.shape[0] * (32 // bits)
    M = qweight.shape[1]
    group_size = K // scales.shape[0]

    return K, M, bits, group_size


def unpack_gptqv2(qweight: np.ndarray, scales: np.ndarray, qzeros: np.ndarray, gptq_v2: bool = True):
    """
    Unpack GPTQv2
    Return T-MAC biased uint8 weight [0, 2 ** bits), fp16 scales, biased fp16 zeros, bits, group_size
    """
    assert qweight.dtype == "int32"
    assert qzeros.dtype == "int32"

    K, M, bits, group_size = parse_gptqv2(qweight, scales, qzeros)

    # Unpack qweight
    qweights = [(qweight >> bit_offset) & ((1 << bits) - 1) for bit_offset in range(0, 32, bits)]
    w = np.stack(qweights, axis=1).reshape(K, M).T.astype("uint8")

    scales = scales.T

    # Unpack qzeros
    zeros = [(qzeros >> bit_offset) & ((1 << bits) - 1) for bit_offset in range(0, 32, bits)]
    zeros = np.stack(zeros, axis=-1).reshape(K // group_size, M).T.astype(scales.dtype)
    if not gptq_v2:
        # `zeros = zeros - 1` in AutoGPTQ
        # Not in GPTQModel
        zeros += 1
    zeros = (zeros - (2 ** (bits - 1))) * scales

    return w, scales, zeros, bits, group_size


def get_quantization_config(model_dir: str) -> dict:
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        hparams = json.load(f)

    # GPTQ
    quantization_config = hparams.get("quantization_config", {})
    desc_act = quantization_config.get("desc_act", False)
    assert not desc_act, "desc_act=True currently unsupported by T-MAC"
    quantizer = quantization_config.get("meta", {}).get("quantizer", "")
    group_size = quantization_config.get("group_size", 0)
    bits = quantization_config.get("bits", 0)
    sym = quantization_config.get("sym", False)
    quant_method = quantization_config.get("quant_method", "")
    # BitNet
    weight_bits = hparams.get("weight_bits", 0)

    return {
        "quantizer": quantizer,
        "group_size": group_size,
        "bits": bits,
        "sym": sym,
        "quant_method": quant_method,
        "weight_bits": weight_bits,
    }


def tighten_bit_array(
    w: np.ndarray,
    bits: int
) -> np.ndarray:
    mask = (1 << bits) - 1
    tightened_array = w & mask
    flattened_bits = np.unpackbits(tightened_array.astype(np.uint8)).reshape(-1, 8)[:, -bits:]
    tightened_compact = np.packbits(flattened_bits)
    return tightened_compact


def preprocess_for_t_mac(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits: int = 2,
    g: int = 4,
) -> np.ndarray:

    w_packed = tighten_bit_array(w, bits)

    if zeros is not None:
        return np.concatenate([w_packed, scales.astype(np.float32).copy().view(np.uint8).flatten(), zeros.astype(np.float32).copy().view(np.uint8).flatten()])
    else:
        return np.concatenate([w_packed, scales.astype(np.float32).copy().view(np.uint8).flatten()])
