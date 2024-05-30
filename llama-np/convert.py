"""
Convert weights to Lumi weights format (.lmw)

This achieves two purposes.
1. No dependence on PyT during inference.
2. The Lumi weights format is Python agnostic (not using Pickle) and can be
   read by other languages.
"""

import argparse
import glob
import json
import math
import os
import struct
import time

import numpy as np
import safetensors.torch
import torch
from lumi_type import LumiDType

from read_lumi import read_lumi


def log(str):
    print(str)


def error(str):
    print(str)
    exit(-1)


def write_padded_string(out_file, s, block_size=4):
    """Pad the string `s` with spaces to make its length a multiple of `block_size`."""
    padding_length = block_size - len(s) % block_size
    padded = (s + (" " * padding_length)).encode()
    format_str = f"I{len(padded)}s"
    out_file.write(struct.pack(format_str, len(padded), padded))


def serialize_bf16_fp32(out_file, tensor, transposed=False):
    """
    In many cases, weights are used transposed. To save network build time,
    we can save the weights transposed.
    """
    assert tensor.dtype == torch.float32 or tensor.dtype == torch.bfloat16
    is_bf16 = tensor.dtype == torch.bfloat16
    d = tensor.detach().cpu()

    if transposed:
        d = d.transpose(0, 1)
        out_file.write(struct.pack("i", 1))
    else:
        out_file.write(struct.pack("i", 0))

    if is_bf16:
        out_file.write(struct.pack("i", LumiDType.bf16.value))
        d = torch.flatten(d).view(torch.int16).numpy()
        b = struct.pack(f"{len(d)}h", *d)
    else:
        out_file.write(struct.pack("i", LumiDType.fp32.value))
        d = torch.flatten(d).to(torch.float32).numpy()
        b = struct.pack(f"{len(d)}f", *d)

    out_file.write(b)


def revert_hf_permute(w, n_heads):
    """
    Undo the following permutation done by the Huggingface transformer library.
     https://github.com/huggingface/transformers/blob/12c39e5693f7223be162a1e84de026a6545029eb/src/transformers/models/llama/convert_llama_weights_to_hf.py#L133
     def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    """
    dim1, dim2 = w.shape
    return (
        w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )
    # return w.reshape(n_heads, 2, dim1 // n_heads // 2, dim2).swapaxes(1,2).reshape(dim1, dim2)


def export_lumi(
    model_name, params, tokenizer_model, state_dict, filepath, n_records=-1
):
    """
    Export the model to the Lumi weights format.

    Args:
        model_name: Name of the model
        params: Dictionary containing the model parameters
        tokenizer_model: Tokenizer model buffer
        state_dict: Dictionary containing the model weights
        filepath: Output file path
        n_records: Number of records to export. -1 for all records. Used for testing.

    Notes:
        We include the model_name in the Lumi weights format, so to enable the model executor
        make model specific decisions.

        The following alternative was considered, tried and rejected:
        - generalize the weights format to include all possible variations of models -
          including hyperparameters, minor differences in model architecture, etc.
        This could make the model executor agnostic to the specifics in each model.
        However, this would also make the format too complex and difficult to maintain,
        as well as paying the cost of re-exporting all models when the format changes or
        pumping up the LMW version.

        The Lumi weights format is as follows:
        1. "lumi" magic number (4 bytes)
        2. lumi format version (4 bytes)
        3. model name (string)
        4. params (dim, n_layers, n_heads, n_kv_heads, multiple_of, vocab_size, max_seq_len, rope_theta, norm_eps)
        5. tokenizer model (buffer)
        6. weight data (name, shape, data)

        The weight data is as follows:
        a. length_of_padded_name, padded_name_string
        b. num of shape dim, size of each dim
        c. transposed flag, dtype, weight values

    """

    start_time = time.time()
    log("writing ..")
    out_file = open(filepath, "wb")

    # 1. "lumi" magic
    out_file.write(struct.pack("I", 0x696D756C))

    # 2. lumi format version
    out_file.write(struct.pack("i", 1))

    # 3. model name
    write_padded_string(out_file, model_name)

    # 4. params
    log(f"write params {params}")
    p = struct.pack(
        "iiiiiiiff",
        params["dim"],
        params["n_layers"],
        params["n_heads"],
        params["n_kv_heads"],
        params["multiple_of"],
        params["vocab_size"],
        params["max_seq_len"],
        params["rope_theta"],
        params["norm_eps"],
    )
    out_file.write(p)

    # 5. tokenizer model
    # write out the tokenizer_model, which is a buffer, using pack
    log(f"Writing tokenizer model of {len(tokenizer_model)} bytes")
    out_file.write(struct.pack("I", len(tokenizer_model)))
    out_file.write(tokenizer_model)
    # write out the padding for the tokenizer model so that the next section starts at a multiple of 4
    padding_length = 4 - len(tokenizer_model) % 4
    out_file.write(b"\0" * padding_length)

    # 6. weight data
    n = 0
    for weight_name in list(state_dict):
        weight = state_dict[weight_name]
        log(f"Writing {weight_name} of {weight.size()}")

        # length_of_padded_name, padded_name_string
        write_padded_string(out_file, weight_name)

        # num of shape dim, size of each dim
        shape = weight.shape
        out_file.write(struct.pack("I", len(shape)))
        for i in shape:
            out_file.write(struct.pack("I", i))

        out_file.flush()
        # weight data
        # if weight_name ends with ".weight" and is not "tok_embeddings.weight" and the weight is 2D
        # then set transposed to True.
        if (
            weight_name.endswith(".weight")
            and weight_name != "tok_embeddings.weight"
            and len(weight.shape) == 2
        ):
            transposed = True
        else:
            transposed = False
        serialize_bf16_fp32(out_file, state_dict[weight_name], transposed=transposed)

        n += 1
        if n_records != -1 and n >= n_records:
            break

    # write to binary file
    out_file.close()

    elapsed_time = time.time() - start_time
    log(f"... {elapsed_time:.2f} seconds.")


def patch_params(params):
    # fill in the default param values if the model does not explicitly set them
    params.setdefault("n_kv_heads", params["n_heads"])
    params.setdefault("max_seq_len", 2048)
    params.setdefault("rope_theta", 10000.0)  # llama3
    params["norm_eps"] = 1e-05
    if params["vocab_size"] == -1:
        params["vocab_size"] = 32000


def read_tokenizer_model(tokenizer_model):
    log(f"reading {tokenizer_model}")
    with open(tokenizer_model, "rb") as f:
        tokenizer_buffer = f.read()
    return tokenizer_buffer


def read_meta_llama(model_path, tokenizer_model):
    log("reading params.json")
    params_path = os.path.join(model_path, "params.json")
    with open(params_path) as f:
        params = json.load(f)
        print(params)
    patch_params(params)

    tokenizer_model_buffer = read_tokenizer_model(tokenizer_model)

    log("reading consolatated.*.pth")
    start_time = time.time()
    model_paths = sorted(glob.glob(os.path.join(model_path, "consolidated.*.pth")))
    models = [torch.load(p, map_location="cpu") for p in model_paths]
    elapsed_time = time.time() - start_time
    log(f"... {elapsed_time:.2f} seconds.")

    state_dict = {}
    for name in list(models[0]):
        tensors = [model[name] for model in models]
        if len(tensors) == 1 or len(tensors[0].shape) == 1:
            state_dict[name] = tensors[0]
            continue
        is_axis_1 = (
            name.startswith("tok_embeddings.")
            or name.endswith(".attention.wo.weight")
            or name.endswith(".feed_forward.w2.weight")
        )
        axis = 1 if is_axis_1 else 0
        log(f"concatnating weight {name}")
        state_dict[name] = torch.cat(tensors, dim=axis)
        for model in models:
            del model[name]

    del models

    return params, tokenizer_model_buffer, state_dict


def read_tinystories_pt(model_path, tokenizer_model):
    log(f"reading {model_path}")
    start_time = time.time()
    content = torch.load(model_path, map_location="cpu")
    elapsed_time = time.time() - start_time
    log(f"... {elapsed_time:.2f} seconds.")

    tokenizer_model_buffer = read_tokenizer_model(tokenizer_model)

    state_dict = {}
    model = content["model"]
    for name in list(model):
        if "_orig_mod" in name:
            # special case for stories260K
            state_dict[name[10:]] = model[name]
        else:
            state_dict[name] = model[name]

    params = {}
    args = content["model_args"]
    params["dim"] = args["dim"]
    params["n_layers"] = args["n_layers"]
    params["n_heads"] = args["n_heads"]
    params["n_kv_heads"] = args["n_kv_heads"]
    params["multiple_of"] = args["multiple_of"]
    params["vocab_size"] = args["vocab_size"]
    params["max_seq_len"] = args["max_seq_len"]
    patch_params(params)

    return params, tokenizer_model_buffer, state_dict


def read_qwen15_tinyllama_data(model_path, tokenizer_model, revert_hf_per):
    start_time = time.time()
    tokenizer_model_buffer = read_tokenizer_model(tokenizer_model)

    model_paths = sorted(glob.glob(os.path.join(model_path, "model-*safetensors")))
    log(f"reading {model_paths}")

    state_dict = {}
    for params_path in model_paths:
        model = safetensors.torch.load_file(params_path)

        for name in sorted(list(model)):
            if name == "lm_head.weight":
                state_dict["output.weight"] = model[name]
            elif name == "model.embed_tokens.weight":
                state_dict["tok_embeddings.weight"] = model[name]
            elif name == "model.norm.weight":
                state_dict["norm.weight"] = model[name]
            elif "layers" in name:
                t = name.split(".")
                if t[3] == "input_layernorm":
                    state_dict["layers." + t[2] + ".attention_norm.weight"] = model[
                        name
                    ]
                if t[3] == "post_attention_layernorm":
                    state_dict["layers." + t[2] + ".ffn_norm.weight"] = model[name]
                elif t[4] == "q_proj":
                    if t[5] == "weight":
                        if revert_hf_per:
                            state_dict["layers." + t[2] + ".attention.wq.weight"] = (
                                revert_hf_permute(model[name], 32)
                            )
                        else:
                            state_dict["layers." + t[2] + ".attention.wq.weight"] = (
                                model[name]
                            )
                    else:
                        state_dict["layers." + t[2] + ".attention.wq.bias"] = model[
                            name
                        ]
                elif t[4] == "k_proj":
                    if t[5] == "weight":
                        if revert_hf_per:
                            state_dict["layers." + t[2] + ".attention.wk.weight"] = (
                                revert_hf_permute(model[name], 4)
                            )
                        else:
                            state_dict["layers." + t[2] + ".attention.wk.weight"] = (
                                model[name]
                            )
                    else:
                        state_dict["layers." + t[2] + ".attention.wk.bias"] = model[
                            name
                        ]
                elif t[4] == "v_proj":
                    if t[5] == "weight":
                        state_dict["layers." + t[2] + ".attention.wv.weight"] = model[
                            name
                        ]
                    else:
                        state_dict["layers." + t[2] + ".attention.wv.bias"] = model[
                            name
                        ]
                elif t[4] == "o_proj":
                    if t[5] == "weight":
                        state_dict["layers." + t[2] + ".attention.wo.weight"] = model[
                            name
                        ]
                    else:
                        state_dict["layers." + t[2] + ".attention.wo.bias"] = model[
                            name
                        ]
                elif t[4] == "gate_proj":
                    state_dict["layers." + t[2] + ".feed_forward.w1.weight"] = model[
                        name
                    ]
                elif t[4] == "up_proj":
                    state_dict["layers." + t[2] + ".feed_forward.w3.weight"] = model[
                        name
                    ]
                elif t[4] == "down_proj":
                    state_dict["layers." + t[2] + ".feed_forward.w2.weight"] = model[
                        name
                    ]
            else:
                print(f"unknown weight {name}")
                exit()

    elapsed_time = time.time() - start_time
    log(f"... {elapsed_time:.2f} seconds.")

    return tokenizer_model_buffer, state_dict


def read_tinyllama(model_path, tokenizer_model):
    tokenizer_model_buffer, state_dict = read_qwen15_tinyllama_data(
        model_path, tokenizer_model, True
    )

    params = {}
    params["dim"] = 2048
    params["n_layers"] = 22
    params["n_heads"] = 32
    params["n_kv_heads"] = 4
    params["multiple_of"] = 8  # just a guess
    params["vocab_size"] = state_dict["tok_embeddings.weight"].shape[0]
    params["max_seq_len"] = 1024  # just a guess
    params["norm_eps"] = 1e-05
    patch_params(params)

    return params, tokenizer_model_buffer, state_dict


def read_qwen10(model_path, tokenizer_model):
    start_time = time.time()

    tokenizer_model_buffer = read_tokenizer_model(tokenizer_model)

    model_paths = sorted(glob.glob(os.path.join(model_path, "model-*safetensors")))
    log(f"reading {model_paths}")

    state_dict = {}
    for params_path in model_paths:
        model = safetensors.torch.load_file(params_path)

        for name in sorted(list(model)):
            if name == "lm_head.weight":
                state_dict["output.weight"] = model[name]
            elif name == "transformer.wte.weight":
                state_dict["tok_embeddings.weight"] = model[name]
            elif name == "transformer.ln_f.weight":
                state_dict["norm.weight"] = model[name]
            elif "transformer.h" in name:
                t = name.split(".")
                if t[3] == "ln_1":
                    state_dict["layers." + t[2] + ".attention_norm.weight"] = model[
                        name
                    ]
                elif t[3] == "ln_2":
                    state_dict["layers." + t[2] + ".ffn_norm.weight"] = model[name]

                elif t[3] == "attn":
                    if t[4] == "c_attn" and t[5] == "weight":
                        q, k, v = model[name].chunk(3, dim=0)
                        state_dict["layers." + t[2] + ".attention.wq.weight"] = q
                        state_dict["layers." + t[2] + ".attention.wk.weight"] = k
                        state_dict["layers." + t[2] + ".attention.wv.weight"] = v
                    elif t[4] == "c_attn" and t[5] == "bias":
                        q, k, v = model[name].chunk(3, dim=0)
                        state_dict["layers." + t[2] + ".attention.wq.bias"] = q
                        state_dict["layers." + t[2] + ".attention.wk.bias"] = k
                        state_dict["layers." + t[2] + ".attention.wv.bias"] = v
                    elif t[4] == "c_proj":
                        state_dict["layers." + t[2] + ".attention.wo.weight"] = model[
                            name
                        ]
                    else:
                        print(f"unknown weight {name}")
                        exit()
                elif t[3] == "mlp":
                    if t[4] == "c_proj":
                        state_dict["layers." + t[2] + ".feed_forward.w2.weight"] = (
                            model[name]
                        )
                    elif t[4] == "w1":
                        state_dict["layers." + t[2] + ".feed_forward.w3.weight"] = (
                            model[name]
                        )
                    elif t[4] == "w2":
                        state_dict["layers." + t[2] + ".feed_forward.w1.weight"] = (
                            model[name]
                        )
                    else:
                        print(f"unknown weight {name}")
                        exit()
            else:
                print(f"unknown weight {name}")
                exit()

    elapsed_time = time.time() - start_time
    log(f"... {elapsed_time:.2f} seconds.")

    params = {}
    params["dim"] = state_dict["tok_embeddings.weight"].shape[1]
    params["n_layers"] = 32
    params["n_heads"] = 32
    params["n_kv_heads"] = 32
    params["multiple_of"] = 8  # just a guess
    params["vocab_size"] = state_dict["tok_embeddings.weight"].shape[0]
    params["max_seq_len"] = 8192  # just a guess
    params["norm_eps"] = 1e-06
    params["rope_theta"] = 1000000
    patch_params(params)

    return params, tokenizer_model_buffer, state_dict


def read_qwen15(model_path, tokenizer_model):
    tokenizer_model_buffer, state_dict = read_qwen15_tinyllama_data(
        model_path, tokenizer_model, False
    )

    params = {}
    params["dim"] = state_dict["tok_embeddings.weight"].shape[1]
    params["n_layers"] = 32
    params["n_heads"] = 32
    params["n_kv_heads"] = 32
    params["multiple_of"] = 8  # just a guess
    params["vocab_size"] = state_dict["tok_embeddings.weight"].shape[0]
    params["max_seq_len"] = 8192  # just a guess
    params["norm_eps"] = 1e-06
    params["rope_theta"] = 1000000
    patch_params(params)

    return params, tokenizer_model_buffer, state_dict


def compare(meta_params, meta_dict, lumi_params, lumi_dict, n_records):
    for p in list(meta_params):
        if p == "vocab_size" and meta_params[p] == -1:
            continue
        if p == "ffn_dim_multiplier" or p == "rope_theta":
            continue
        log(f"comparison - {p} {meta_params[p]}  vs  {lumi_params[p]}")
        if not math.isclose(meta_params[p], lumi_params[p], rel_tol=1e-04):
            error(f"comparison failed")

    n = 0
    for w in list(meta_dict):
        log(f"comparson - weight {w}")
        meta = meta_dict[w].detach().cpu().view(-1).to(torch.float32).numpy()
        if meta_dict[w].shape != lumi_dict[w].shape:
            error(f"shape mismatch: {meta_dict[w].shape} vs {lumi_dict[w].shape}")
        if not np.allclose(meta, lumi_dict[w].reshape(-1), rtol=1e-05, equal_nan=False):
            error(f"comparison failed")
        n += 1
        if n_records != -1 and n >= n_records:
            break
    return


if __name__ == "__main__":
    """
    python convert.py model tokenizer.model output.lmw
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="name of the model. This is used to make model specific decisions in the model executor.",
    )
    parser.add_argument(
        "input_model",
        type=str,
        help="Path of input model. Maybe a file or a directory, depending on the model.",
    )
    parser.add_argument(
        "tokenizer_model",
        type=str,
        help="Path of the tokenizer.model file. Should include the filename.",
    )
    parser.add_argument(
        "lumi_model",
        type=str,
        help="Path of the output lumi model. Should include the filename.",
    )

    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test by first exporting and reading back a few weights in the lumi format and then comparing the results.",
    )
    args = parser.parse_args()

    log(f"input: {args.input_model}")
    log(f"output: {args.lumi_model}")

    # Detect the model and process accordingly
    if "llama-2" in args.model_name.lower():
        print("Reading llama-2 model")
        meta_params, tokenizer_model, meta_dict = read_meta_llama(
            args.input_model, args.tokenizer_model
        )
    elif "llama-3" in args.model_name.lower():
        print("Reading llama-3 model")
        meta_params, tokenizer_model, meta_dict = read_meta_llama(
            args.input_model, args.tokenizer_model
        )
    elif "stories" in args.model_name.lower():
        print("Reading TinyStories model")
        meta_params, tokenizer_model, meta_dict = read_tinystories_pt(
            args.input_model, args.tokenizer_model
        )
    elif "tinyllama" in args.model_name.lower():
        print("Reading TinyLlama model")
        meta_params, tokenizer_model, meta_dict = read_tinyllama(
            args.input_model, args.tokenizer_model
        )
    elif "qwen1.0-7b-chat" in args.model_name.lower():
        print("Reading Qwen model 1.0 7B Chat")
        meta_params, tokenizer_model, meta_dict = read_qwen10(
            args.input_model, args.tokenizer_model
        )
    elif "qwen1.5-7b-chat" in args.model_name.lower():
        print("Reading Qwen model 1.5 7B Chat")
        meta_params, tokenizer_model, meta_dict = read_qwen15(
            args.input_model, args.tokenizer_model
        )
    else:
        print("Unknown model type")
        exit()

    if args.test:
        n_records = 15
        export_lumi(
            args.model_name,
            meta_params,
            tokenizer_model,
            meta_dict,
            args.lumi_model,
            n_records,
        )
        lumi_params, lumi_tokenizer_model, lumi_dict, _ = read_lumi(
            args.lumi_model, n_records
        )
        compare(meta_params, meta_dict, lumi_params, lumi_dict, n_records)
    else:
        export_lumi(
            args.model_name, meta_params, tokenizer_model, meta_dict, args.lumi_model
        )
