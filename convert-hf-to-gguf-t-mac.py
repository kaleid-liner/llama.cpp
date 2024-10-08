#!/usr/bin/env python3

from __future__ import annotations

import logging
import argparse
import contextlib
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterator, Sequence, TypeVar, cast, Optional, Tuple
import configparser

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from convert import LlamaHfVocab, permute

logger = logging.getLogger("hf-to-gguf")


###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class LlamaFType(IntEnum):
    F32 = 0
    MOSTLY_F16 = 1
    MOSTLY_IN = 32


AnyModel = TypeVar("AnyModel", bound="type[Model]")


class Model(ABC):
    _model_classes: dict[str, type[Model]] = {}

    def __init__(self, dir_model: Path, ftype: int, fname_out: Path, is_big_endian: bool, use_temp_file: bool):
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.is_safetensors = self._is_model_safetensors()
        self.num_parts = Model.count_model_parts(self.dir_model, ".safetensors" if self.is_safetensors else ".bin")
        self.part_names = self._get_part_names()
        self.hparams = Model.load_hparams(self.dir_model)
        self.gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file)
        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @property
    @abstractmethod
    def model_arch(self) -> gguf.MODEL_ARCH:
        pass

    def find_hparam(self, keys: Sequence[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                for name in model_part.keys():
                    data = model_part.get_tensor(name) if self.is_safetensors else model_part[name]
                    yield name, data

    def match_model_tensor_name(self, name: str, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        self.gguf_writer.add_embedding_length(n_embd)
        logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_head_count(n_head)
        logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        quant_dict = {}
        # Store scales and qzeros to dict to be later preprocessed
        # Save memory by not storing qweight
        for name, data_torch in self.get_tensors():
            if name.endswith(".scales") or name.endswith(".qzeros"):
                data = data_torch.numpy()
                quant_dict[name] = data
        if len(quant_dict) > 0:
            from t_mac.model_utils import get_quantization_config
            quantization_config = get_quantization_config(self.dir_model)

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            # should be converted with qweight together
            if name.endswith(".scales") or name.endswith(".qzeros") or name.endswith(".g_idx"):
                continue

            if name.endswith(".qweight"):
                qweight = data_torch.numpy()
                scales = quant_dict[name.replace(".qweight", ".scales")]
                qzeros = quant_dict[name.replace(".qweight", ".qzeros")]
                from t_mac.model_utils import unpack_gptqv2
                w, scales, zeros, bits, group_size = unpack_gptqv2(qweight, scales, qzeros, "gptqmodel" in quantization_config["quantizer"])
                if bits != quantization_config["bits"] or group_size != quantization_config["group_size"]:
                    logger.warning("Error while parsing weights for quantization_config: {}".format(quantization_config))
                data_shape = w.shape
                new_name = tensor_map.get_name(name.replace(".qweight", ".weight"), try_suffixes=(".weight", ".bias"))

                if self.ftype == LlamaFType.MOSTLY_IN:
                    if bits == 1:
                        to_dtype = gguf.GGMLQuantizationType.I1
                    elif bits == 2:
                        to_dtype = gguf.GGMLQuantizationType.I2
                    elif bits == 3:
                        to_dtype = gguf.GGMLQuantizationType.I3
                    elif bits == 4:
                        to_dtype = gguf.GGMLQuantizationType.I4
                    if quantization_config["sym"]:
                        if not np.allclose(zeros, np.zeros_like(zeros)):
                            logger.warning("Although the quantized model claimed to be symmetric, the weights are asymmetric")
                        else:
                            zeros = None
                    data = preprocess_for_t_mac(w, scales, zeros, bits=bits)
                else:
                    to_dtype = gguf.GGMLQuantizationType.F32
                    w = w.astype("float32").reshape(-1, group_size)
                    scales = scales.astype("float32").reshape(-1, 1)
                    zeros = zeros.astype("float32").reshape(-1, 1)
                    data = (w - (zeros / scales + (2 ** (bits - 1)))) * scales
                    if self.ftype == LlamaFType.MOSTLY_F16:
                        to_dtype = gguf.GGMLQuantizationType.F16
                        data = data.astype("float16")

                logger.info(f"{new_name}, n_dims = {data_torch.ndim}, {data_torch.dtype} --> {to_dtype.name}")
                self.gguf_writer.add_tensor(new_name, data, raw_shape=data_shape, raw_dtype=to_dtype)
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            data = data_torch.squeeze().numpy()
            data_shape = data.shape

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            extra_f32 = any(self.match_model_tensor_name(new_name, key, bid) for key in (
                gguf.MODEL_TENSOR.FFN_GATE_INP,
                gguf.MODEL_TENSOR.POS_EMBD,
                gguf.MODEL_TENSOR.TOKEN_TYPES,
            ))

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            extra_f16 = any(cond for cond in (
                (name.endswith(".weight") and n_dims >= 2),
            ))

            to_dtype = gguf.GGMLQuantizationType.F32

            if self.ftype != LlamaFType.F32 and extra_f16 and not extra_f32:
                to_dtype = gguf.GGMLQuantizationType.F16

            if to_dtype == gguf.GGMLQuantizationType.F32:
                data = data.astype(np.float32)
            elif to_dtype == gguf.GGMLQuantizationType.F16:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {to_dtype.name}")

            self.gguf_writer.add_tensor(new_name, data, raw_shape=data_shape, raw_dtype=to_dtype)

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write_vocab(self):
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def count_model_parts(dir_model: Path, prefix: str) -> int:
        num_parts = 0
        not_included = ["training_args.bin"]
        for filename in os.listdir(dir_model):
            if filename.endswith(prefix) and filename not in not_included:
                num_parts += 1

        return num_parts

    @staticmethod
    def load_hparams(dir_model):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: type[Model]):
            for name in names:
                cls._model_classes[name] = modelcls
            return modelcls
        return func

    @classmethod
    def from_model_architecture(cls, arch):
        try:
            return cls._model_classes[arch]
        except KeyError:
            raise NotImplementedError(f'Architecture {arch!r} not supported!') from None

    def _is_model_safetensors(self) -> bool:
        return Model.count_model_parts(self.dir_model, ".safetensors") > 0

    def _get_part_names(self):
        if self.is_safetensors:
            if self.num_parts == 1:  # there's only one .safetensors file
                return ("model.safetensors",)
            return tuple(f"model-{n:05}-of-{self.num_parts:05}.safetensors" for n in range(1, self.num_parts + 1))

        if self.num_parts == 1:  # there's only one .bin file
            return ("pytorch_model.bin",)
        return tuple(f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin" for n in range(1, self.num_parts + 1))

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert-hf-to-gguf-update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggerganov/llama.cpp/pull/6920
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert-hf-to-gguf-update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3":
            # ref: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b
            res = "stablelm2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"
        if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
            # ref: https://huggingface.co/Qwen/Qwen1.5-7B
            res = "qwen2"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/allenai/OLMo-1.7-7B-hf
            res = "olmo"
        if chkhsh == "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e":
            # ref: https://huggingface.co/databricks/dbrx-base
            res = "dbrx"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-en
            res = "jina-v2-en"
        if chkhsh == "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-es
            res = "jina-v2-es"
        if chkhsh == "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-de
            res = "jina-v2-de"
        if chkhsh == "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d":
            # ref: https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct
            res = "smaug-bpe"
        if chkhsh == "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360":
            # ref: https://huggingface.co/LumiOpen/Poro-34B-chat
            res = "poro-chat"
        if chkhsh == "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
            res = "jina-v2-code"
        if chkhsh == "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee":
            # ref: https://huggingface.co/LumiOpen/Viking-7B
            res = "viking"
        if chkhsh == "b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901":
            # ref: https://huggingface.co/core42/jais-13b
            res = "jais"

        if res is None:
            logger.warning("\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert-hf-to-gguf-update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert-hf-to-gguf-update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggerganov/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning("**************************************************************************************")
            logger.warning("\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_qwen(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.special_tokens
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in (vocab | added_vocab).items()}

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        # only add special tokens when they were not already loaded from config.json
        if len(special_vocab.special_token_ids) == 0:
            special_vocab._set_special_token("bos", tokenizer.special_tokens["<|endoftext|>"])
            special_vocab._set_special_token("eos", tokenizer.special_tokens["<|endoftext|>"])
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.special_tokens["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor(str(tokenizer_path))
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    key = key.encode("utf-8")
                    if key not in tokens:
                        tokens.append(key)
                        scores.append(-1000.0)
                        toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(f"[PAD{i}]")
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        assert len(tokens) == vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_llama_hf(self):
        vocab = LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


@Model.register("GPTNeoXForCausalLM")
class GPTNeoXModel(Model):
    model_arch = gguf.MODEL_ARCH.GPTNEOX

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(
            int(self.hparams["rotary_pct"] * (self.hparams["hidden_size"] // self.hparams["num_attention_heads"])),
        )
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_parallel_residual(self.hparams.get("use_parallel_residual", True))
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_eps"])


@Model.register("BloomForCausalLM")
class BloomModel(Model):
    model_arch = gguf.MODEL_ARCH.BLOOM

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("Bloom")
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(4 * n_embed)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams["n_layer"]
        tensors = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        has_lm_head = True
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))

        for name, data_torch in tensors.items():
            if "lm_head.weight" not in tensors.keys() and "output.weight" not in tensors.keys():
                has_lm_head = False

            name = re.sub(r'transformer\.', '', name)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            if re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
                # Map bloom-style qkv_linear to gpt-style qkv_linear
                # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
                # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
                qkv_weights = data.reshape((n_head, 3, n_embed // n_head, n_embed))
                data = np.concatenate(
                    (
                        qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                        qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                        qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                    ),
                    axis=0,
                )
                logger.info("re-format attention.linear_qkv.weight")
            elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
                qkv_bias = data.reshape((n_head, 3, n_embed // n_head))
                data = np.concatenate(
                    (
                        qkv_bias[:, 0, :].reshape((n_embed,)),
                        qkv_bias[:, 1, :].reshape((n_embed,)),
                        qkv_bias[:, 2, :].reshape((n_embed,)),
                    ),
                    axis=0,
                )
                logger.info("re-format attention.linear_qkv.bias")

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"=> {new_name}, shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

            if not has_lm_head and name == "word_embeddings.weight":
                self.gguf_writer.add_tensor("output.weight", data)
                logger.info(name, f"=> output.weight, shape = {data.shape}, {old_dtype} --> {data.dtype}")


@Model.register("MPTForCausalLM")
class MPTModel(Model):
    model_arch = gguf.MODEL_ARCH.MPT

    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
        except Exception:
            # Fallback for SEA-LION model
            self._set_vocab_sentencepiece()
            self.gguf_writer.add_add_bos_token(False)
            self.gguf_writer.add_pad_token_id(3)
            self.gguf_writer.add_eos_token_id(1)
            self.gguf_writer.add_unk_token_id(0)

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layers"]
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["d_model"])
        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        if kv_n_heads := self.hparams["attn_config"].get("kv_n_heads"):
            self.gguf_writer.add_head_count_kv(kv_n_heads)
        self.gguf_writer.add_layer_norm_eps(1e-5)
        if self.hparams["attn_config"]["clip_qkv"] is not None:
            self.gguf_writer.add_clamp_kqv(self.hparams["attn_config"]["clip_qkv"])
        if self.hparams["attn_config"]["alibi"]:
            self.gguf_writer.add_max_alibi_bias(self.hparams["attn_config"]["alibi_bias_max"])
        else:
            self.gguf_writer.add_max_alibi_bias(0.0)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            if "scales" in name:
                new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias", ".scales"))
                if new_name is not None:
                    new_name = new_name.replace("scales", "act.scales")
            else:
                new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("OrionForCausalLM")
class OrionModel(Model):
    model_arch = gguf.MODEL_ARCH.ORION

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        # note: config provides rms norm but it is actually layer norm
        # ref:  https://huggingface.co/OrionStarAI/Orion-14B-Chat/blob/276a17221ce42beb45f66fac657a41540e71f4f5/modeling_orion.py#L570-L571
        self.gguf_writer.add_layer_norm_eps(self.hparams["rms_norm_eps"])

    def write_tensors(self):
        # Collect tensors from generator object
        model_kv = dict(self.get_tensors())
        block_count = self.hparams["num_hidden_layers"]
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{name} -> {new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)


@Model.register("BaichuanForCausalLM", "BaiChuanForCausalLM")
class BaichuanModel(Model):
    model_arch = gguf.MODEL_ARCH.BAICHUAN

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def write_tensors(self):
        # Collect tensors from generator object
        model_kv = dict(self.get_tensors())
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        for i in range(block_count):
            if (w := model_kv.get(f"model.layers.{i}.self_attn.W_pack.weight")) is not None:
                logger.info(f"Unpacking and permuting layer {i}")
                model_kv[f"model.layers.{i}.self_attn.q_proj.weight"] = \
                    self._reverse_hf_permute_part(w, 0, head_count, head_count)
                model_kv[f"model.layers.{i}.self_attn.k_proj.weight"] = \
                    self._reverse_hf_permute_part(w, 1, head_count, head_count_kv)
                model_kv[f"model.layers.{i}.self_attn.v_proj.weight"] = \
                    self._reverse_hf_part(w, 2)
                del model_kv[f"model.layers.{i}.self_attn.W_pack.weight"]

        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{name} -> {new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def _reverse_hf_permute_part(
        self, weights: Tensor, n_part: int, n_head: int, n_head_kv: int | None = None,
    ) -> Tensor:
        r = weights.shape[0] // 3
        return self._reverse_hf_permute(weights[r * n_part:r * n_part + r, ...], n_head, n_head_kv)

    def _reverse_hf_part(self, weights: Tensor, n_part: int) -> Tensor:
        r = weights.shape[0] // 3
        return weights[r * n_part:r * n_part + r, ...]


@Model.register("XverseForCausalLM")
class XverseModel(Model):
    model_arch = gguf.MODEL_ARCH.XVERSE

    def set_vocab(self):
        assert (self.dir_model / "tokenizer.json").is_file()
        dir_model = self.dir_model
        hparams = self.hparams

        tokens: list[bytearray] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model)
        vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for token_id in range(vocab_size):
            token_text = reverse_vocab[token_id].encode('utf-8')
            # replace "\x00" to string with length > 0
            if token_text == b"\x00":
                toktype = gguf.TokenType.BYTE  # special
                token_text = f"<{token_text}>".encode('utf-8')
            elif re.fullmatch(br"<0x[0-9A-Fa-f]{2}>", token_text):
                toktype = gguf.TokenType.BYTE  # special
            elif reverse_vocab[token_id] in added_vocab:
                if tokenizer.added_tokens_decoder[token_id].special:
                    toktype = gguf.TokenType.CONTROL
                else:
                    toktype = gguf.TokenType.USER_DEFINED
            else:
                toktype = gguf.TokenType.NORMAL

            tokens.append(token_text)
            toktypes.append(toktype)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def write_tensors(self):
        # Collect tensors from generator object
        model_kv = dict(self.get_tensors())
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # HF models permute some of the tensors, so we need to undo that
            if name.endswith(("q_proj.weight")):
                data_torch = self._reverse_hf_permute(data_torch, head_count, head_count)
            if name.endswith(("k_proj.weight")):
                data_torch = self._reverse_hf_permute(data_torch, head_count, head_count_kv)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{name} -> {new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )


@Model.register("FalconForCausalLM", "RWForCausalLM")
class FalconModel(Model):
    model_arch = gguf.MODEL_ARCH.FALCON

    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        self.gguf_writer.add_name("Falcon")
        self.gguf_writer.add_context_length(2048)  # not in config.json
        self.gguf_writer.add_tensor_data_layout("jploski")  # qkv tensor transform
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        head_dim = self.hparams["hidden_size"] // n_head
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # QKV tensor transform
            # The original query_key_value tensor contains n_head_kv "kv groups",
            # each consisting of n_head/n_head_kv query weights followed by one key
            # and one value weight (shared by all query heads in the kv group).
            # This layout makes it a big pain to work with in GGML.
            # So we rearrange them here,, so that we have n_head query weights
            # followed by n_head_kv key weights followed by n_head_kv value weights,
            # in contiguous fashion.
            # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

            if "query_key_value" in name:
                qkv = data_torch.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
                q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
                k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
                v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
                data_torch = torch.cat((q, k, v)).reshape_as(data_torch)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("GPTBigCodeForCausalLM")
class StarCoderModel(Model):
    model_arch = gguf.MODEL_ARCH.STARCODER

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("StarCoder")
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@Model.register("GPTRefactForCausalLM")
class RefactModel(Model):
    model_arch = gguf.MODEL_ARCH.REFACT

    def set_gguf_parameters(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("Refact")
        # refact uses Alibi. So this is from config.json which might be used by training.
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])

        self.gguf_writer.add_feed_forward_length(ff_dim)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        n_head = self.hparams["n_head"]
        n_head_kv = 1
        head_dim = self.hparams["n_embd"] // n_head
        block_count = self.hparams["n_layer"]

        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        tensors = dict(self.get_tensors())
        for i in range(block_count):
            if (w := tensors.get(f"transformer.h.{i}.attn.kv.weight")) is not None:
                tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = w[:n_head_kv * head_dim]
                tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = w[n_head_kv * head_dim:]
                del tensors[f"transformer.h.{i}.attn.kv.weight"]
            if (w := tensors.get(f"transformer.h.{i}.attn.q.weight")) is not None:
                tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = w
                del tensors[f"transformer.h.{i}.attn.q.weight"]
            if (w := tensors.get(f"transformer.h.{i}.mlp.gate_up_proj.weight")) is not None:
                tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = w[:ff_dim]
                tensors[f"model.layers.{i}.mlp.up_proj.weight"] = w[ff_dim:]
                del tensors[f"transformer.h.{i}.mlp.gate_up_proj.weight"]

        for name, data_torch in tensors.items():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight",))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("PersimmonForCausalLM")
class PersimmonModel(Model):
    model_arch = gguf.MODEL_ARCH.PERSIMMON

    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = head_count
        hidden_size = self.hparams["hidden_size"]

        self.gguf_writer.add_name('persimmon-8b-chat')
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hidden_size)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])

        # NOTE: not sure about this change - why does the model not have a rope dimension count when it is smaller
        #       than the head size?
        #       ref: https://github.com/ggerganov/llama.cpp/pull/4889
        # self.gguf_writer.add_rope_dimension_count(hidden_size // head_count)
        self.gguf_writer.add_rope_dimension_count(hidden_size // head_count // 2)

        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_eps"])

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        # self.gguf_writer.add_bos_token_id(71013)
        # self.gguf_writer.add_eos_token_id(71013)

    def write_tensors(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            if name.endswith(".self_attention.rotary_emb.inv_freq"):
                continue
            old_dtype = data_torch.dtype
            # TODO: FP16 conversion produces garbage outputs. (Q8_0 does not, so..?)
            data = data_torch.to(torch.float32).squeeze().numpy()
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")
            n_dims = len(data.shape)
            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)


@Model.register("StableLmForCausalLM", "StableLMEpochForCausalLM", "LlavaStableLMEpochForCausalLM")
class StableLMModel(Model):
    model_arch = gguf.MODEL_ARCH.STABLELM

    def set_vocab(self):
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            # StableLM 2 1.6B uses a vocab in a similar format to Qwen's vocab
            self._set_vocab_qwen()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        rotary_factor = self.find_hparam(["partial_rotary_factor", "rope_pct"])
        self.gguf_writer.add_rope_dimension_count(int(rotary_factor * (hparams["hidden_size"] // hparams["num_attention_heads"])))
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])
        self.gguf_writer.add_parallel_residual(hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_eps", "norm_eps"]))

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        n_head = self.hparams.get("num_attention_heads")
        n_kv_head = self.hparams.get("num_key_value_heads")
        q_norms = dict()
        k_norms = dict()
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()
            n_dims = len(data.shape)
            if name.find("q_layernorm.norms") != -1:
                q_norms[name] = data
                if len(q_norms) >= (block_count * n_head):
                    self._stack_qk_norm(block_count, name, tensor_map, n_head, q_norms, n_dims, layer_name="q_layernorm")
                continue
            if name.find("k_layernorm.norms") != -1:
                k_norms[name] = data
                if len(k_norms) >= (block_count * n_kv_head):
                    self._stack_qk_norm(block_count, name, tensor_map, n_kv_head, k_norms, n_dims, layer_name="k_layernorm")
                continue

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and (n_dims == 1 or new_name.endswith("_norm.weight")):
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and not new_name.endswith("_norm.weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.debug(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

    def _stack_qk_norm(self, block_count, name, tensor_map, n_head, norms, n_dims, layer_name="q_layernorm"):
        for bid in range(block_count):
            datas = []
            for xid in range(n_head):
                ename = f"model.layers.{bid}.self_attn.{layer_name}.norms.{xid}.weight"
                datas.append(norms[ename])
                del norms[ename]
            data = np.stack(datas, axis=0)
            data_dtype = data.dtype
            merged_name = f"model.layers.{bid}.self_attn.{layer_name}.weight"
            new_name = tensor_map.get_name(merged_name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")
            if self.ftype == 1 and data_dtype == np.float16 and (n_dims == 1 or new_name.endswith("_norm.weight")):
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and not new_name.endswith("_norm.weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.debug(f"{new_name}, n_dims = {len(data.shape)}, shape = {data.shape} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


# Modified version of BitDistiller pseudo_quantize_tensor
# core quantization method (simulated quantization)
def real_quantize_tensor(w, n_bit=8, zero_point=True, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)

    w = w.reshape(org_w_shape).numpy()
    scales = scales.numpy().reshape(w.shape[0], -1)
    zeros = zeros.numpy().reshape(w.shape[0], -1) if zero_point else None

    if zero_point:
        w = w.astype(np.uint8)
        zeros = (zeros - (2 ** (n_bit - 1))) * scales
        return w, scales, zeros
    else:
        w = (w - min_int).astype(np.uint8)
        return w, scales, zeros


@Model.register("LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM")
class LlamaModel(Model):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_vocab(self):
        try:
            self. _set_vocab_sentencepiece()
        except FileNotFoundError:
            try:
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                # Llama 3
                self._set_vocab_gpt2()

        # Apply to CodeLlama only (and ignore for Llama 3 with a vocab size of 128256)
        if self.hparams.get("vocab_size", 32000) == 32016:
            special_vocab = gguf.SpecialVocab(
                self.dir_model, load_merges=False,
                special_token_types = ['prefix', 'suffix', 'middle', 'eot']
            )
            special_vocab._set_special_token("prefix", 32007)
            special_vocab._set_special_token("suffix", 32008)
            special_vocab._set_special_token("middle", 32009)
            special_vocab._set_special_token("eot",    32010)
            special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["hidden_size"] // hparams["num_attention_heads"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    # Same as super class, but permuting q_proj, k_proj
    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        n_head = self.hparams.get("num_attention_heads")
        n_kv_head = self.hparams.get("num_key_value_heads")
        n_experts = self.hparams.get("num_local_experts")
        experts = dict()

        quant_dict = {}
        # Store scales and qzeros to dict to be later preprocessed
        # Save memory by not storing qweight
        for name, data_torch in self.get_tensors():
            if name.endswith(".scales") or name.endswith(".qzeros"):
                data = data_torch.numpy()
                quant_dict[name] = data
        if len(quant_dict) > 0:
            from t_mac.model_utils import get_quantization_config
            quantization_config = get_quantization_config(self.dir_model)

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            # should be converted with qweight together
            if name.endswith(".scales") or name.endswith(".qzeros") or name.endswith(".g_idx"):
                continue

            if name.endswith(".qweight"):
                qweight = data_torch.numpy()
                scales = quant_dict[name.replace(".qweight", ".scales")]
                qzeros = quant_dict[name.replace(".qweight", ".qzeros")]
                from t_mac.model_utils import unpack_gptqv2
                w, scales, zeros, bits, group_size = unpack_gptqv2(qweight, scales, qzeros, "gptqmodel" in quantization_config["quantizer"])
                if bits != quantization_config["bits"] or group_size != quantization_config["group_size"]:
                    logger.warning("Error while parsing weights for quantization_config: {}".format(quantization_config))
                if name.endswith("q_proj.qweight"):
                    w = permute(w, n_head, n_head)
                    scales = permute(scales, n_head, n_head)
                    zeros = permute(zeros, n_head, n_head)
                if name.endswith("k_proj.qweight"):
                    w = permute(w, n_head, n_kv_head)
                    scales = permute(scales, n_head, n_kv_head)
                    zeros = permute(zeros, n_head, n_kv_head)
                data_shape = w.shape
                new_name = tensor_map.get_name(name.replace(".qweight", ".weight"), try_suffixes=(".weight", ".bias"))

                if self.ftype == LlamaFType.MOSTLY_IN:
                    if bits == 1:
                        to_dtype = gguf.GGMLQuantizationType.I1
                    elif bits == 2:
                        to_dtype = gguf.GGMLQuantizationType.I2
                    elif bits == 3:
                        to_dtype = gguf.GGMLQuantizationType.I3
                    elif bits == 4:
                        to_dtype = gguf.GGMLQuantizationType.I4
                    if quantization_config["sym"]:
                        if not np.allclose(zeros, np.zeros_like(zeros)):
                            logger.warning("Although the GPTQ model claimed to be symmetric, the weights are asymmetric according to qzeros")
                        else:
                            zeros = None
                    data = preprocess_for_t_mac(w, scales, zeros, bits=bits)
                else:
                    to_dtype = gguf.GGMLQuantizationType.F32
                    w = w.astype("float32").reshape(-1, group_size)
                    scales = scales.astype("float32").reshape(-1, 1)
                    zeros = zeros.astype("float32").reshape(-1, 1)
                    data = (w - (zeros / scales + (2 ** (bits - 1)))) * scales
                    if self.ftype == LlamaFType.MOSTLY_F16:
                        to_dtype = gguf.GGMLQuantizationType.F16
                        data = data.astype("float16")

                logger.info(f"{new_name}, n_dims = {data_torch.ndim}, {data_torch.dtype} --> {to_dtype.name}")
                self.gguf_writer.add_tensor(new_name, data, raw_shape=data_shape, raw_dtype=to_dtype)
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            data = data_torch.numpy()
            data_shape = data.shape

            if name.endswith("q_proj.weight"):
                data = permute(data, n_head, n_head)
            if name.endswith("k_proj.weight"):
                data = permute(data, n_head, n_kv_head)

            data = data.squeeze()

            # process the experts separately
            if name.find("block_sparse_moe.experts") != -1:
                experts[name] = data
                if len(experts) >= n_experts:
                    # merge the experts into a single 3d tensor
                    for bid in range(block_count):
                        for wid in range(1, 4):
                            full = True
                            for xid in range(n_experts):
                                ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.w{wid}.weight"
                                if ename not in experts:
                                    full = False
                                    break
                            if not full:
                                continue

                            datas = []
                            for xid in range(n_experts):
                                ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.w{wid}.weight"
                                datas.append(experts[ename])
                                del experts[ename]

                            data = np.stack(datas, axis=0)
                            data_dtype = data.dtype

                            if self.ftype == LlamaFType.F32 and data_dtype == np.float16:
                                data = data.astype(np.float32)

                            if self.ftype == LlamaFType.MOSTLY_F16 and data_dtype == np.float32:
                                data = data.astype(np.float16)

                            merged_name = f"layers.{bid}.feed_forward.experts.w{wid}.weight"

                            new_name = tensor_map.get_name(merged_name, try_suffixes=(".weight", ".bias"))
                            if new_name is None:
                                raise ValueError(f"Can not map tensor {name!r}")

                            logger.info(f"{new_name}, n_dims = {len(data.shape)}, shape = {data.shape} --> {data.dtype}")

                            self.gguf_writer.add_tensor(new_name, data)
                continue

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            extra_f32 = any(self.match_model_tensor_name(new_name, key, bid) for key in (
                gguf.MODEL_TENSOR.FFN_GATE_INP,
                gguf.MODEL_TENSOR.POS_EMBD,
                gguf.MODEL_TENSOR.TOKEN_TYPES,
            ))

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            extra_f16 = any(cond for cond in (
                (name.endswith(".weight") and n_dims >= 2),
            ))

            to_dtype = gguf.GGMLQuantizationType.F32

            if self.ftype != LlamaFType.F32 and extra_f16 and not extra_f32:
                if self.ftype == LlamaFType.MOSTLY_IN and any(self.match_model_tensor_name(new_name, key, bid) for key in [
                    gguf.MODEL_TENSOR.ATTN_Q,
                    gguf.MODEL_TENSOR.ATTN_K,
                    gguf.MODEL_TENSOR.ATTN_V,
                    gguf.MODEL_TENSOR.ATTN_OUT,
                    gguf.MODEL_TENSOR.FFN_UP,
                    gguf.MODEL_TENSOR.FFN_DOWN,
                    gguf.MODEL_TENSOR.FFN_GATE,
                ]):
                    to_dtype = gguf.GGMLQuantizationType.I2
                else:
                    to_dtype = gguf.GGMLQuantizationType.F16

            scales = None
            if to_dtype == gguf.GGMLQuantizationType.F32:
                data = data.astype(np.float32)
            elif to_dtype == gguf.GGMLQuantizationType.F16:
                data = data.astype(np.float16)
            elif to_dtype == gguf.GGMLQuantizationType.I2:
                bits = 2
                w, scales, zeros = real_quantize_tensor(
                    data_torch,
                    n_bit=bits,
                    zero_point=True,
                    q_group_size=args.group_size,
                )
                data = preprocess_for_t_mac(w, scales, zeros, bits=bits)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {to_dtype.name}")

            self.gguf_writer.add_tensor(new_name, data, raw_shape=data_shape, raw_dtype=to_dtype)

        if len(experts) > 0:
            raise ValueError(f"Unprocessed experts: {experts.keys()}")


@Model.register("GrokForCausalLM")
class GrokModel(Model):
    model_arch = gguf.MODEL_ARCH.GROK

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_name("Grok")

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        n_experts = self.hparams.get("num_local_experts")
        experts = dict()
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # process the experts separately
            if name.find(".moe.") != -1:
                experts[name] = data
                if len(experts) >= n_experts:
                    # merge the experts into a single 3d tensor
                    for bid in range(block_count):
                        for wid in ["linear", "linear_1", "linear_v"]:
                            full = True
                            for xid in range(n_experts):
                                ename = f"transformer.decoder_layer.{bid}.moe.{xid}.{wid}.weight"
                                if ename not in experts:
                                    full = False
                                    break
                            if not full:
                                continue

                            datas = []
                            for xid in range(n_experts):
                                ename = f"transformer.decoder_layer.{bid}.moe.{xid}.{wid}.weight"
                                datas.append(experts[ename])
                                del experts[ename]

                            data = np.stack(datas, axis=0)
                            data_dtype = data.dtype

                            if self.ftype == LlamaFType.F32 and data_dtype == np.float16:
                                data = data.astype(np.float32)

                            if self.ftype == LlamaFType.MOSTLY_F16 and data_dtype == np.float32:
                                data = data.astype(np.float16)

                            merged_name = f"transformer.decoder_layer.{bid}.moe.{wid}.weight"

                            new_name = tensor_map.get_name(merged_name, try_suffixes=(".weight", ".bias"))
                            if new_name is None:
                                raise ValueError(f"Can not map tensor {name!r}")

                            logger.info(f"{new_name}, n_dims = {len(data.shape)}, shape = {data.shape} --> {data.dtype}")

                            self.gguf_writer.add_tensor(new_name, data)
                continue

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("DbrxForCausalLM")
class DbrxModel(Model):
    model_arch = gguf.MODEL_ARCH.DBRX

    def set_gguf_parameters(self):
        ffn_config = self.hparams["ffn_config"]
        attn_config = self.hparams["attn_config"]
        self.gguf_writer.add_name(self.hparams["model_type"])
        self.gguf_writer.add_block_count(self.hparams["n_layers"])

        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(ffn_config["ffn_hidden_size"])

        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        self.gguf_writer.add_head_count_kv(attn_config["kv_n_heads"])

        self.gguf_writer.add_rope_freq_base(attn_config["rope_theta"])

        self.gguf_writer.add_clamp_kqv(attn_config["clip_qkv"])
        self.gguf_writer.add_file_type(self.ftype)

        self.gguf_writer.add_expert_count(ffn_config["moe_num_experts"])
        self.gguf_writer.add_expert_used_count(ffn_config["moe_top_k"])

        self.gguf_writer.add_layer_norm_eps(1e-5)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def write_tensors(self):
        block_count = self.hparams.get("n_layers")
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in self.get_tensors():
            n_expert = self.hparams["ffn_config"]["moe_num_experts"]
            n_ff = self.hparams["ffn_config"]["ffn_hidden_size"]
            n_embd = self.hparams["d_model"]

            # Specific behavior for experts tensors: suffix .weight, view as 3D and transpose
            # original implementation expects (n_expert, n_ff, n_embd) for all experts weights
            # But llama.cpp moe graph works differently
            # AND the dimensions in ggml are typically in the reverse order of the pytorch dimensions
            # so (n_expert, n_ff, n_embd) in pytorch is {n_embd, n_ff, n_expert} in ggml_tensor
            exp_tensor_names = {"ffn.experts.mlp.w1": None,       # LLM_TENSOR_FFN_GATE_EXPS ggml_tensor->ne{n_embd, n_ff,   n_expert}
                                "ffn.experts.mlp.w2": (0, 2, 1),  # LLM_TENSOR_FFN_DOWN_EXPS ggml_tensor->ne{n_ff,   n_embd, n_expert}
                                "ffn.experts.mlp.v1": None}       # LLM_TENSOR_FFN_UP_EXPS   ggml_tensor->ne{n_embd, n_ff,   n_expert}
            experts = False
            for exp_tensor_name in exp_tensor_names.keys():
                if name.find(exp_tensor_name) != -1 and name.find(".weight") == -1:
                    experts = True
                    data_torch = data_torch.view(n_expert, n_ff, n_embd)
                    if (permute_tensor := exp_tensor_names[exp_tensor_name]) is not None:
                        data_torch = data_torch.permute(*permute_tensor)
                    break

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            # In MoE models the ffn tensors are typically most of the model weights,
            # and need to be quantizable. Quantize expects tensor names to be suffixed by .weight.
            # Every other model has the weight names ending in .weight,
            # let's assume that is the convention which is not the case for dbrx:
            # https://huggingface.co/databricks/dbrx-instruct/blob/main/model.safetensors.index.json#L15
            new_name = tensor_map.get_name(name if not experts else name + ".weight", try_suffixes=(".weight",))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # Most of the codebase that takes in 1D tensors only handles F32 tensors
            # and most of the outputs tensors are F32.
            if data_dtype != np.float32 and n_dims == 1:
                raise ValueError(f"Can not map tensor {name!r}: all 1D tensors must be F32")

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and n_dims > 1:
                data = data.astype(np.float16)

            logger.debug(f"{new_name}, n_dims = {n_dims}, shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("MiniCPMForCausalLM")
class MiniCPMModel(Model):
    model_arch = gguf.MODEL_ARCH.MINICPM

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        self.gguf_writer.add_name("MiniCPM")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_llama_hf()

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        n_head = self.hparams.get("num_attention_heads")
        n_kv_head = self.hparams.get("num_key_value_heads")
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # HF models permute some of the tensors, so we need to undo that
            if name.endswith(("q_proj.weight")):
                data_torch = self._reverse_hf_permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight")):
                data_torch = self._reverse_hf_permute(data_torch, n_head, n_kv_head)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("QWenLMHeadModel")
class QwenModel(Model):
    model_arch = gguf.MODEL_ARCH.QWEN

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_vocab(self):
        self._set_vocab_qwen()

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("Qwen")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])

    def write_tensors(self):
        block_count = self.hparams["num_hidden_layers"]
        model_kv = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_writer.add_tensor(new_name, data)


@Model.register("Qwen2ForCausalLM")
class Qwen2Model(Model):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()


@Model.register("Qwen2MoeForCausalLM")
class Qwen2MoeModel(Model):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        n_experts = self.hparams.get("num_experts")
        experts = dict()
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # process the experts separately
            if name.find("experts") != -1:
                experts[name] = data
                if len(experts) >= n_experts * 3:
                    # merge the experts into a single 3d tensor
                    for bid in range(block_count):
                        for w_name in ["down_proj", "gate_proj", "up_proj"]:
                            full = True
                            for xid in range(n_experts):
                                ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                                if ename not in experts:
                                    full = False
                                    break
                            if not full:
                                continue

                            datas = []
                            for xid in range(n_experts):
                                ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                                datas.append(experts[ename])
                                del experts[ename]

                            data = np.stack(datas, axis=0)
                            data_dtype = data.dtype

                            if self.ftype == 0 and data_dtype == np.float16:
                                data = data.astype(np.float32)

                            if self.ftype == 1 and data_dtype == np.float32:
                                data = data.astype(np.float16)

                            merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                            new_name = tensor_map.get_name(merged_name, try_suffixes=(".weight", ".bias"))
                            if new_name is None:
                                raise ValueError(f"Can not map tensor {name!r}")

                            logger.debug(f"{new_name}, n_dims = {len(data.shape)}, shape = {data.shape} --> {data.dtype}")

                            self.gguf_writer.add_tensor(new_name, data)
                continue

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and (n_dims == 1 or new_name.endswith("_norm.weight")):
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.debug(f"{new_name}, n_dims = {n_dims}, shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

        if len(experts) > 0:
            raise ValueError(f"Unprocessed experts: {experts.keys()}")


@Model.register("GPT2LMHeadModel")
class GPT2Model(Model):
    model_arch = gguf.MODEL_ARCH.GPT2

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq", ".attn.bias", ".attn.masked_bias")):
                continue

            if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")):
                data_torch = data_torch.transpose(1, 0)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

            # note: GPT2 output is tied to (same as) wte in original model
            if new_name == "token_embd.weight":
                logger.info(f"output.weight, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
                self.gguf_writer.add_tensor("output.weight", data)


@Model.register("PhiForCausalLM")
class Phi2Model(Model):
    model_arch = gguf.MODEL_ARCH.PHI2

    def set_gguf_parameters(self):
        block_count = self.find_hparam(["num_hidden_layers", "n_layer"])

        rot_pct = self.find_hparam(["partial_rotary_factor"])
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])

        self.gguf_writer.add_name("Phi2")
        self.gguf_writer.add_context_length(self.find_hparam(["n_positions", "max_position_embeddings"]))

        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(4 * n_embd)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_epsilon", "layer_norm_eps"]))
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * n_embd) // n_head)
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_add_bos_token(False)


@Model.register("Phi3ForCausalLM")
class Phi3MiniModel(Model):
    model_arch = gguf.MODEL_ARCH.PHI3

    def set_vocab(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            raise ValueError(f'Error: Missing {tokenizer_path}')

        tokenizer = SentencePieceProcessor(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNKNOWN] * vocab_size

        for token_id in range(tokenizer.vocab_size()):

            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if (token_id >= vocab_size):
                        logger.debug(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        block_count = self.find_hparam(["num_hidden_layers", "n_layer"])

        rot_pct = 1.0
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        rms_eps = self.find_hparam(["rms_norm_eps"])

        self.gguf_writer.add_name("Phi3")
        self.gguf_writer.add_context_length(self.find_hparam(["n_positions", "max_position_embeddings"]))

        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(8192)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_rms_eps(rms_eps)
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * n_embd) // n_head)
        self.gguf_writer.add_file_type(self.ftype)


@Model.register("PlamoForCausalLM")
class PlamoModel(Model):
    model_arch = gguf.MODEL_ARCH.PLAMO

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_name("PLaMo")
        self.gguf_writer.add_context_length(4096)  # not in config.json
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(5)  # hparams["num_key_value_heads"]) is wrong
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

    def shuffle_attn_q_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(8, 5, 128, 5120)
        data_torch = torch.permute(data_torch, (1, 0, 2, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def shuffle_attn_output_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(5120, 8, 5, 128)
        data_torch = torch.permute(data_torch, (0, 2, 1, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def write_tensors(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            if "self_attn.rotary_emb.inv_freq" in name:
                continue

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            # shuffle for broadcasting of gqa in ggml_mul_mat
            if new_name.endswith("attn_q.weight"):
                data_torch = self.shuffle_attn_q_weight(data_torch)
            elif new_name.endswith("attn_output.weight"):
                data_torch = self.shuffle_attn_output_weight(data_torch)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("CodeShellForCausalLM")
class CodeShellModel(Model):
    model_arch = gguf.MODEL_ARCH.CODESHELL

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("CodeShell")
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_query_groups"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_rope_freq_base(10000.0)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        tensors = dict(self.get_tensors())
        has_lm_head = "lm_head.weight" in tensors.keys() or "output.weight" in tensors.keys()
        for name, data_torch in tensors.items():
            # we don't need these
            if name.endswith((".attn.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

            if not has_lm_head and name == "transformer.wte.weight":
                self.gguf_writer.add_tensor("output.weight", data)
                logger.info(name, f"=> output.weight, shape = {data.shape}, {old_dtype} --> {data.dtype}")


@Model.register("InternLM2ForCausalLM")
class InternLM2Model(Model):
    model_arch = gguf.MODEL_ARCH.INTERNLM2

    def set_vocab(self):
        # (TODO): Is there a better way?
        # Copy from _set_vocab_sentencepiece, The only difference is that we will treat the character
        # \x00 specially and convert it into an emoji character to prevent it from being mistakenly
        # recognized as an empty string in C++.
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            logger.error(f'Error: Missing {tokenizer_path}')
            sys.exit(1)

        sentencepiece_model = model.ModelProto()
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix

        tokenizer = SentencePieceProcessor(str(tokenizer_path))
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        for token_id in range(vocab_size):
            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)
            if text == b"\x00":
                # (TODO): fixme
                # Hack here and replace the \x00 characters.
                logger.debug(f"InternLM2 convert token '{text}' to '🐉'!")
                text = "🐉"

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        old_eos = special_vocab.special_token_ids["eos"]
        if "chat" in os.path.basename(self.dir_model.absolute()):
            # For the chat model, we replace the eos with '<|im_end|>'.
            # TODO: this is a hack, should be fixed
            #       https://github.com/ggerganov/llama.cpp/pull/6745#issuecomment-2067687048
            special_vocab.special_token_ids["eos"] = self._try_get_sft_eos(tokenizer)
            logger.warning(f"Replace eos:{old_eos} with a special token:{special_vocab.special_token_ids['eos']} \
in chat mode so that the conversation can end normally.")

        special_vocab.add_to_gguf(self.gguf_writer)

    def _try_get_sft_eos(self, tokenizer):
        unused_145_list = tokenizer.encode('[UNUSED_TOKEN_145]')
        im_end_list = tokenizer.encode('<|im_end|>')
        assert (len(unused_145_list) == 1) ^ (len(im_end_list) == 1)
        if len(unused_145_list) == 1:
            eos_token = unused_145_list[0]
        if len(im_end_list) == 1:
            eos_token = im_end_list[0]
        return eos_token

    def _hf_permute_qk(self, weights, n_head: int, n_head_kv: int):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("InternLM2")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])

    def post_write_tensors(self, tensor_map, name, data_torch):
        old_dtype = data_torch.dtype

        # convert any unsupported data types to float32
        if data_torch.dtype not in (torch.float16, torch.float32):
            data_torch = data_torch.to(torch.float32)

        data = data_torch.squeeze().numpy()

        # map tensor names
        new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")

        n_dims = len(data.shape)
        data_dtype = data.dtype

        # if f32 desired, convert any float16 to float32
        if self.ftype == 0 and data_dtype == np.float16:
            data = data.astype(np.float32)

        # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
        if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            data = data.astype(np.float32)

        # if f16 desired, convert any float32 2-dim weight tensors to float16
        if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
            data = data.astype(np.float16)

        logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
        self.gguf_writer.add_tensor(new_name, data)

    def write_tensors(self):
        from einops import rearrange

        num_heads = self.hparams.get("num_attention_heads")
        num_kv_heads = self.hparams.get("num_key_value_heads")
        hidden_size = self.hparams.get("hidden_size")
        q_per_kv = num_heads // num_kv_heads
        head_dim = hidden_size // num_heads
        num_groups = num_heads // q_per_kv

        block_count = self.hparams["num_hidden_layers"]
        model_kv = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        qkv_pattern = r"model\.layers\.(\d+)\.attention\.wqkv"
        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            if re.match(qkv_pattern, name):
                bid = re.findall(qkv_pattern, name)[0]
                qkv = data_torch
                qkv = rearrange(qkv.T, " o (g n i) ->o g n i", g=num_groups, n=q_per_kv + 2, i=head_dim)
                q, k, v = qkv[..., : q_per_kv, :], qkv[..., q_per_kv: q_per_kv + 1, :], qkv[..., q_per_kv + 1: q_per_kv + 2, :]
                # The model weights of q and k equire additional reshape.
                q = self._hf_permute_qk(rearrange(q, " o g n i ->  o (g n i)").T, num_heads, num_heads)
                k = self._hf_permute_qk(rearrange(k, " o g n i ->  o (g n i)").T, num_heads, num_kv_heads)
                v = rearrange(v, " o g n i ->  o (g n i)").T
                self.post_write_tensors(tensor_map, f"model.layers.{bid}.attention.wq.weight", q)
                self.post_write_tensors(tensor_map, f"model.layers.{bid}.attention.wk.weight", k)
                self.post_write_tensors(tensor_map, f"model.layers.{bid}.attention.wv.weight", v)
            else:
                self.post_write_tensors(tensor_map, name, data_torch)


@Model.register("BertModel", "CamembertModel")
class BertModel(Model):
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)

        # get pooling path
        pooling_path = None
        module_path = self.dir_model / "modules.json"
        if module_path.is_file():
            with open(module_path, encoding="utf-8") as f:
                modules = json.load(f)
            for mod in modules:
                if mod["type"] == "sentence_transformers.models.Pooling":
                    pooling_path = mod["path"]
                    break

        # get pooling type
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling["pooling_mode_mean_tokens"]:
                pooling_type = gguf.PoolingType.MEAN
            elif pooling["pooling_mode_cls_token"]:
                pooling_type = gguf.PoolingType.CLS
            else:
                raise NotImplementedError("Only MEAN and CLS pooling types supported")
            self.gguf_writer.add_pooling_type(pooling_type)

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.vocab_size = len(tokens)

        # we need this to validate the size of the token_type embeddings
        # though currently we are passing all zeros to the token_type embeddings
        self.gguf_writer.add_token_type_count(2)  # "Sequence A" or "Sequence B"

        # convert to phantom space vocab
        def phantom(tok):
            if tok.startswith("[") and tok.endswith("]"):
                return tok
            if tok.startswith("##"):
                return tok[2:]
            return "\u2581" + tok
        tokens = list(map(phantom, tokens))

        # add vocab to gguf
        self.gguf_writer.add_tokenizer_model("bert")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # handle special tokens
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def write_tensors(self):
        tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        tensors = dict(self.get_tensors())
        for name, data_torch in tensors.items():
            # we are only using BERT for embeddings so we don't need the pooling layer
            if name in ("embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"):
                continue  # we don't need these

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()
            n_dims = len(data.shape)
            new_dtype: type[np.floating[Any]]

            if (
                self.ftype == 1 and name.endswith(".weight") and n_dims == 2
                and name != "embeddings.token_type_embeddings.weight"  # not used with get_rows, must be F32
            ):
                # if f16 desired, convert any float32 2-dim weight tensors to float16
                new_dtype = np.float16
            else:
                # if f32 desired, convert any float16 to float32
                new_dtype = np.float32

            logger.info(f"{new_name}, n_dims = {n_dims}, {data_torch.dtype} --> {new_dtype}")

            if data.dtype != new_dtype:
                data = data.astype(new_dtype)

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("NomicBertModel")
class NomicBertModel(BertModel):
    model_arch = gguf.MODEL_ARCH.NOMIC_BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # the HF config claims n_ctx=8192, but it uses RoPE scaling
        self.hparams["n_ctx"] = 2048

        # SwigLU activation
        assert self.hparams["activation_function"] == "swiglu"
        # this doesn't do anything in the HF version
        assert self.hparams["causal"] is False
        # no bias tensors
        assert self.hparams["qkv_proj_bias"] is False
        assert self.hparams["mlp_fc1_bias"] is False
        assert self.hparams["mlp_fc2_bias"] is False
        # norm at end of layer
        assert self.hparams["prenorm"] is False
        # standard RoPE
        assert self.hparams["rotary_emb_fraction"] == 1.0
        assert self.hparams["rotary_emb_interleaved"] is False
        assert self.hparams["rotary_emb_scale_base"] is None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])


@Model.register("GemmaForCausalLM")
class GemmaModel(Model):
    model_arch = gguf.MODEL_ARCH.GEMMA

    def set_vocab(self):
        self._set_vocab_sentencepiece()

        # TODO: these special tokens should be exported only for the CodeGemma family
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'fsep', 'eot'])
        special_vocab._set_special_token("prefix", 67)
        special_vocab._set_special_token("suffix", 69)
        special_vocab._set_special_token("middle", 68)
        special_vocab._set_special_token("fsep",   70)
        special_vocab._set_special_token("eot",    107)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"] if "num_key_value_heads" in hparams else hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            # lm_head is not used in llama.cpp, while autoawq will include this tensor in model
            # To prevent errors, skip loading lm_head.weight.
            if name == "lm_head.weight":
                logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
            if name.endswith("norm.weight"):
                data_torch = data_torch + 1
            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("Starcoder2ForCausalLM")
class StarCoder2Model(Model):
    model_arch = gguf.MODEL_ARCH.STARCODER2


@Model.register("MambaForCausalLM", "MambaLMHeadModel")
class MambaModel(Model):
    model_arch = gguf.MODEL_ARCH.MAMBA

    def set_vocab(self):
        vocab_size = self.hparams["vocab_size"]
        # Round vocab size to next multiple of 8
        pad_vocab = self.hparams.get("pad_vocab_size_multiple", 8)
        # pad using ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        vocab_size = -(vocab_size // -pad_vocab) * pad_vocab
        self.hparams["vocab_size"] = vocab_size

        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            # Use the GPT-NeoX tokenizer when no tokenizer files are present
            tokenizer_path = Path(sys.path[0]) / "models" / "ggml-vocab-gpt-neox.gguf"
            logger.warning(f"Using tokenizer from '{os.path.relpath(tokenizer_path, os.getcwd())}'")
            neox_reader = gguf.GGUFReader(tokenizer_path, "r")

            field = neox_reader.get_field(gguf.Keys.Tokenizer.MODEL)
            self.gguf_writer.add_tokenizer_model(bytes(field.parts[-1]))

            field = neox_reader.get_field(gguf.Keys.Tokenizer.PRE)
            self.gguf_writer.add_tokenizer_pre(bytes(field.parts[-1]))

            field = neox_reader.get_field(gguf.Keys.Tokenizer.LIST)
            self.gguf_writer.add_token_list([bytes(field.parts[i]) for i in field.data][:vocab_size])

            field = neox_reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE)
            self.gguf_writer.add_token_types([field.parts[i].tolist()[0] for i in field.data][:vocab_size])

            field = neox_reader.get_field(gguf.Keys.Tokenizer.MERGES)
            self.gguf_writer.add_token_merges([bytes(field.parts[i]) for i in field.data])

            field = neox_reader.get_field(gguf.Keys.Tokenizer.BOS_ID)
            self.gguf_writer.add_bos_token_id(field.parts[-1].tolist()[0])

            field = neox_reader.get_field(gguf.Keys.Tokenizer.EOS_ID)
            self.gguf_writer.add_eos_token_id(field.parts[-1].tolist()[0])

            field = neox_reader.get_field(gguf.Keys.Tokenizer.UNK_ID)
            self.gguf_writer.add_unk_token_id(field.parts[-1].tolist()[0])

    def set_gguf_parameters(self):
        d_model = self.find_hparam(["hidden_size",       "d_model"])
        d_conv  = self.find_hparam(["conv_kernel",       "d_conv"],  optional=True) or 4
        d_inner = self.find_hparam(["intermediate_size", "d_inner"], optional=True) or 2 * d_model
        d_state = self.find_hparam(["state_size",        "d_state"], optional=True) or 16
        # ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        # ref: https://github.com/state-spaces/mamba/blob/ce59daea3a090d011d6476c6e5b97f6d58ddad8b/mamba_ssm/modules/mamba_simple.py#L58
        dt_rank      = self.find_hparam(["time_step_rank",     "dt_rank"],      optional=True) or -(d_model // -16)
        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-5

        # Fail early for models which don't have a block expansion factor of 2
        assert d_inner == 2 * d_model

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(2**20) # arbitrary value; for those who use the default
        self.gguf_writer.add_embedding_length(d_model)
        self.gguf_writer.add_feed_forward_length(0) # unused, but seemingly required when loading
        self.gguf_writer.add_head_count(0) # unused, but seemingly required when loading
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(dt_rank)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams["n_layer"]
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        tok_embd = None
        tok_embd_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.TOKEN_EMBD] + ".weight"
        output_name   = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.OUTPUT]     + ".weight"

        for name, data_torch in self.get_tensors():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            if name.endswith(".A_log"):
                logger.debug("A_log --> A ==> " + new_name)
                data_torch = -torch.exp(data_torch)

            # assuming token_embd.weight is seen before output.weight
            if tok_embd is not None and new_name == output_name:
                if torch.equal(tok_embd, data_torch):
                    logger.debug(f"{output_name} is equivalent to {tok_embd_name}, omitting")
                    continue
            if new_name == tok_embd_name:
                tok_embd = data_torch

            data = data_torch.squeeze().numpy()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert big float32 2-dim weight tensors to float16
            new_weight_name = new_name[:-len(".weight")] if new_name.endswith(".weight") else ""
            if self.ftype == 1 and data_dtype == np.float32 and new_weight_name.endswith((".ssm_in", ".ssm_out", "token_embd", "output")) and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


@Model.register("CohereForCausalLM")
class CommandR2Model(Model):
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        self.hparams["max_position_embeddings"] = self.hparams["model_max_length"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@Model.register("OlmoForCausalLM")
@Model.register("OLMoForCausalLM")
class OlmoModel(Model):
    model_arch = gguf.MODEL_ARCH.OLMO

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_layer_norm_eps(1e-5)
        if "clip_qkv" in self.hparams is not None:
            self.gguf_writer.add_clamp_kqv(self.hparams["clip_qkv"])

    # Same as super class, but permuting q_proj, k_proj
    # Copied from: LlamaModel
    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        n_head = self.hparams.get("num_attention_heads")
        n_kv_head = self.hparams.get("num_key_value_heads")
        for name, data_torch in self.get_tensors():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.numpy()

            if name.endswith("q_proj.weight"):
                data = permute(data, n_head, n_head)
            if name.endswith("k_proj.weight"):
                data = permute(data, n_head, n_kv_head)

            data = data.squeeze()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # 1d tensors need to be converted to float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)


def preprocess_for_t_mac(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits = 2,
    g    = 4,
) -> np.ndarray:
    from t_mac.weights import preprocess_weights

    M, K = w.shape
    cf = configparser.ConfigParser()
    cf.read(args.kcfg)
    secs = cf.sections()
    found = False
    for sec in secs:
        sec_splits = str(sec).split('_')
        if sec_splits[-4] == "m" + str(M * bits) and sec_splits[-3] == "k" + str(K):
            bm = int(cf.get(sec, 'bm'))
            kfactor = int(cf.get(sec, 'kfactor'))
            simd_n_in = int(cf.get(sec, 'simd_n_in'))
            simd_n_out = int(cf.get(sec, 'simd_n_out'))
            found = True
            break

    if not found:
        raise KeyError("GEMM of shape ({}, {}) is not found in {}. Please compile the kernels using T-MAC first.".format(M, K, args.kcfg))
    
    w, scales = preprocess_weights(w, scales, zeros, bits=bits, g=g, bm=bm, kfactor=kfactor, simd_n_in=simd_n_in, simd_n_out=simd_n_out)
    return np.concatenate([w.flatten(), scales.astype(np.float32).view(np.uint8).flatten()])
    

@Model.register("BitnetForCausalLM")
class BitnetModel(Model):
    model_arch = gguf.MODEL_ARCH.BITNET

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        
    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def weight_quant(self, weight):
        dtype = weight.dtype
        weight = weight.float()
        s =  1 / weight.abs().mean().clamp(min=1e-5)
        result = (weight * s).round().clamp(-1, 1) / s
        return result.type(dtype)

    def transform_to_i2(self, x: np.ndarray):
        scale = np.max(np.abs(x))
        res = np.round(x / scale + 2).astype(np.uint8)
        res = preprocess_for_t_mac(res, scale.reshape(1))
        return res

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # quant weight to i2 (in fp16)
        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", 
                          "down_proj.weight", "up_proj.weight", "gate_proj.weight",
                          "o_proj.weight")):
            data_torch = self.weight_quant(data_torch)

        return [(self.map_tensor_name(name), data_torch)]

    def write_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data in ((n, d.squeeze().numpy()) for n, d in self.modify_tensors(data_torch, name, bid)):
                data: np.ndarray = data  # type hint
                data_shape = data.shape
                n_dims = len(data.shape)
                data_dtype = data.dtype
                data_qtype: gguf.GGMLQuantizationType | None = None

                # when both are True, f32 should win
                # extra_f32 = self.extra_f32_tensors(name, new_name, bid, n_dims)
                # extra_f16 = self.extra_f16_tensors(name, new_name, bid, n_dims)
                extra_f32 = False
                extra_f16 = False

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                extra_f32 = any(cond for cond in (
                    extra_f32,
                    n_dims == 1,
                    new_name.endswith("_norm.weight"),
                ))

                # Some tensor types are always in float32
                extra_f32 = extra_f32 or any(self.match_model_tensor_name(new_name, key, bid) for key in (
                    gguf.MODEL_TENSOR.FFN_GATE_INP,
                    gguf.MODEL_TENSOR.POS_EMBD,
                    gguf.MODEL_TENSOR.TOKEN_TYPES,
                ))

                # if f16 desired, convert any float32 2-dim weight tensors to float16
                extra_f16 = any(cond for cond in (
                    extra_f16,
                    (name.endswith(".weight") and n_dims >= 2),
                ))

                suit_i2 = True
                if name.endswith('embed_tokens.weight') or name.endswith('norm.weight'):
                    suit_i2 = False

                if self.ftype != LlamaFType.F32 and extra_f16 and not extra_f32:
                    if self.ftype == LlamaFType.MOSTLY_IN and suit_i2:
                        data = self.transform_to_i2(data)
                        assert data.dtype == np.uint8
                        data_qtype = gguf.GGMLQuantizationType.I2

                    else:  # default to float16 for quantized tensors
                        if data_dtype != np.float16:
                            data = data.astype(np.float16)
                        data_qtype = gguf.GGMLQuantizationType.F16

                if data_qtype is None:  # by default, convert to float32
                    if data_dtype != np.float32:
                        data = data.astype(np.float32)
                    data_qtype = gguf.GGMLQuantizationType.F32

                shape = data_shape
                # shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape
                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

                self.gguf_writer.add_tensor(new_name, data, raw_shape=shape, raw_dtype=data_qtype)


###### CONVERSION LOGIC ######


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--awq-path", type=Path, default=None,
        help="Path to scale awq cache file")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "i2", "i3", "i4", "in"], default="f16",
        help="output format - use f32 for float32, f16 for float16",
    )
    parser.add_argument("--bigendian", action="store_true", help="model is executed on big endian machine")
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file",
    )
    parser.add_argument("--use-temp-file", action="store_true", help="use the tempfile library while processing (helpful when running out of memory, process killed)")
    parser.add_argument("--model-name", type=str, default=None, help="name of the model")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("--kcfg", type=str, default="", help="Path to T-MAC kcfg.ini")
    parser.add_argument("--quant-type", type=str, default="bitnet", choices=["bitnet", "bitdistiller", "gptqv2"])
    parser.add_argument("--group-size", type=int, default=128)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dir_model = args.model

    if args.awq_path:
        sys.path.insert(1, str(Path(__file__).parent / 'awq-py'))
        from awq.apply_awq import add_scale_weights  # type: ignore[import-not-found]
        tmp_model_path = args.model / "weighted_model"
        dir_model = tmp_model_path
        if tmp_model_path.is_dir():
            logger.info(f"{tmp_model_path} exists as a weighted model.")
        else:
            tmp_model_path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving new weighted model ...")
            add_scale_weights(str(args.model), str(args.awq_path), str(tmp_model_path))
            logger.info(f"Saved weighted model at {tmp_model_path}.")

    if not dir_model.is_dir():
        logger.error(f'Error: {args.model} is not a directory')
        sys.exit(1)

    ftype_map = {
        "f32": LlamaFType.F32,
        "f16": LlamaFType.MOSTLY_F16, 
        "in" : LlamaFType.MOSTLY_IN,
        "i2" : LlamaFType.MOSTLY_IN,
        "i3" : LlamaFType.MOSTLY_IN,
        "i4" : LlamaFType.MOSTLY_IN,
    }

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_model / f'ggml-model-{args.outtype}.gguf'

    logger.info(f"Loading model: {dir_model.name}")

    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        model_class = Model.from_model_architecture(hparams["architectures"][0])
        model_instance = model_class(dir_model, ftype_map[args.outtype], fname_out, args.bigendian, args.use_temp_file)

        logger.info("Set model parameters")
        model_instance.set_gguf_parameters()

        logger.info("Set model tokenizer")
        model_instance.set_vocab()

        if args.vocab_only:
            logger.info(f"Exporting model vocab to '{fname_out}'")
            model_instance.write_vocab()
        else:
            logger.info(f"Exporting model to '{fname_out}'")
            model_instance.write()

        logger.info(f"Model successfully exported to '{fname_out}'")


if __name__ == '__main__':
    args = parse_args()

    main()
