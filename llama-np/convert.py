"""
Convert Meta's official Llama3 weights to Lumi weights format (.lmw).

This achives two purposes.
1. No dependence on PyT during inference.
2. The Lumi weights format is python agnostic (not using Pickle) and can be
   read by other lanaguages.
"""

import argparse
import glob
import json

import math
import os
import struct
import time

import numpy as np
import torch
import safetensors.torch

from read_lumi import read_lumi

from torch import nn


def log(str):
    print(str)


def error(str):
    print(str)
    exit(-1)


def pad_string(s, block_size=4):
    """Pad the string `s` with spaces to make its length a multiple of `block_size`."""
    padding_length = block_size - len(s) % block_size
    return (s + (" " * padding_length)).encode()


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)

# 
# https://github.com/huggingface/transformers/blob/12c39e5693f7223be162a1e84de026a6545029eb/src/transformers/models/llama/convert_llama_weights_to_hf.py#L133
#     def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
#        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
# 
def revert_hf_permute(w, n_heads):
    dim1, dim2 = w.shape
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1,2).reshape(dim1, dim2)
    #return w.reshape(n_heads, 2, dim1 // n_heads // 2, dim2).swapaxes(1,2).reshape(dim1, dim2)

def export_lumi(params, state_dict, filepath, n_records=-1):

    start_time = time.time()
    log("writing ..")
    out_file = open(filepath, "wb")

    # 1. "lumi" magic
    out_file.write(struct.pack("I", 0x696D756C))

    # 2. version
    out_file.write(struct.pack("i", 1))

    # 3. params
    log(f"write params {params}")
    p = struct.pack(
        "iiiiiiiff",
        params["dim"],
        params["n_layers"],
        params["n_heads"],
        params["n_kv_heads"],
        params["multiple_of"],
        params['vocab_size'], #state_dict["tok_embeddings.weight"].shape[0],  # vocab_size
        params['max_seq_len'], #2048,  # max_seq_len
        params['rope_theta'],
        params["norm_eps"],
    )
    out_file.write(p)

    # 4. weight data
    n = 0
    for weight_name in list(state_dict):
        weight = state_dict[weight_name]
        log(f"Writing {weight_name} of {weight.size()}")

        # length_of_padded_name, padded_name_string
        name = pad_string(weight_name)
        format_str = f"I{len(name)}s"
        out_file.write(struct.pack(format_str, len(name), name))

        # num of shape dim, size of each dim
        shape = weight.shape
        out_file.write(struct.pack("I", len(shape)))
        for i in shape:
            out_file.write(struct.pack("I", i))

        out_file.flush()
        # weight data
        serialize_fp32(out_file, state_dict[weight_name])

        n += 1
        if n_records != -1 and n >= n_records:
            break

    # write to binary file
    out_file.close()

    elapsed_time = time.time() - start_time
    log(f"... {elapsed_time:.2f} seconds.")


def patch_params(params):
    # fill in the default param values if the model does not explicitly set them
    params.setdefault('n_kv_heads', params['n_heads']) 
    params.setdefault('max_seq_len', 2048) 
    params.setdefault('rope_theta', 10000.0) # llama3 
    params['norm_eps'] = 1e-05
    if params['vocab_size'] == -1:
        params['vocab_size'] = 32000 

def read_meta_llama(model_path):
    log("reading params.json")
    params_path = os.path.join(model_path, "params.json")
    with open(params_path) as f:
        params = json.load(f)
        print(params)
    patch_params(params)

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

    return (params, state_dict)

def read_tinystories_pt(model_path):
    log(f"reading {model_path}")
    start_time = time.time()
    content = torch.load(model_path, map_location="cpu") 
    elapsed_time = time.time() - start_time
    log(f"... {elapsed_time:.2f} seconds.")

    state_dict = {}
    model = content['model']
    for name in list(model):
        if '_orig_mod' in name:
            # special case for stories260K
            state_dict[name[10:]] = model[name]
        else:
            state_dict[name] = model[name]

    params = {}
    args = content['model_args']
    params['dim'] = args['dim']
    params['n_layers'] = args['n_layers']
    params['n_heads'] = args['n_heads']
    params['n_kv_heads'] = args['n_kv_heads']
    params['multiple_of'] = args['multiple_of']
    params['vocab_size'] = args['vocab_size']
    params['max_seq_len'] = args['max_seq_len']
    patch_params(params)

    return (params, state_dict)

def read_tinyllama(model_path):
    start_time = time.time()
    if "Chat" in model_path:
        params_path = os.path.join(model_path, "model.safetensors")
        log(f"reading {params_path}")
        model = safetensors.torch.load_file(params_path)
    else:
        #params_path = os.path.join(model_path, "pytorch_model.bin")
        #log(f"reading {params_path}")
        #model = torch.load(params_path, map_location="cpu")
        params_path = os.path.join(model_path, "model.safetensors")
        log(f"reading {params_path}")
        model = safetensors.torch.load_file(params_path)
    elapsed_time = time.time() - start_time
    log(f"... {elapsed_time:.2f} seconds.")

    state_dict = {}
    for name in sorted(list(model)):
        if name == 'lm_head.weight':
            state_dict["output.weight"] = model[name]
        elif name == 'model.embed_tokens.weight':
            state_dict["tok_embeddings.weight"] = model[name]
        elif name == 'model.norm.weight':
            state_dict["norm.weight"] = model[name]
        elif 'layers' in name:
            t = name.split('.')
            if t[3] == 'input_layernorm':
                state_dict["layers."+t[2]+".attention_norm.weight"] = model[name]
            if t[3] == 'post_attention_layernorm':
                state_dict["layers."+t[2]+".ffn_norm.weight"] = model[name]
            elif t[4] == 'q_proj':
                state_dict["layers."+t[2]+".attention.wq.weight"] = revert_hf_permute(model[name],32)
            elif t[4] == 'k_proj':
                state_dict["layers."+t[2]+".attention.wk.weight"] = revert_hf_permute(model[name],4)
            elif t[4] == 'v_proj':
                state_dict["layers."+t[2]+".attention.wv.weight"] = model[name]
            elif t[4] == 'o_proj':
                state_dict["layers."+t[2]+".attention.wo.weight"] = model[name]
            elif t[4] == 'gate_proj':
                state_dict["layers."+t[2]+".feed_forward.w3.weight"] = model[name]
            elif t[4] == 'up_proj':
                state_dict["layers."+t[2]+".feed_forward.w1.weight"] = model[name]
            elif t[4] == 'down_proj':
                state_dict["layers."+t[2]+".feed_forward.w2.weight"] = model[name]
        else:
            print(f"unknown weight {name}")
            exit()

    params = {}
    params['dim'] = 2048
    params['n_layers'] = 22 
    params['n_heads'] = 32 
    params['n_kv_heads'] = 4 
    params['multiple_of'] = 8 # just a guess 
    params['vocab_size'] = 32000 
    params['max_seq_len'] = 1024  # just a guess
    params['norm_eps'] = 1e-05
    patch_params(params)

    return (params, state_dict)

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
    '''
    Directory as the input path
    % python convert.py llama-2-7b llama-2-7b.lmw -t
    % python convert.py llama-2-7b llama-2-7b.lmw 

    File as the input path
    % python convert.py TinyStories/stories15M.pt stories15M.lmw 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="input path of input model")
    parser.add_argument("lumipath", type=str, help="output path of lumi model")
    parser.add_argument(
        "-t",
        action="store_true",
        help="test by first exporting and reading back a few weights in the lumi format and then comparing the results",
    )
    args = parser.parse_args()

    log(f"input: {args.input_path}")
    log(f"output: {args.lumipath}")

    # Detect the model and process accordingly
    if "llama-2" in args.input_path:
        (meta_params, meta_dict) = read_meta_llama(args.input_path)
    elif ("Llama-3" in args.input_path) or ("llama-3" in args.input_path):
        (meta_params, meta_dict) = read_meta_llama(args.input_path)
    elif "stories" in args.input_path:
        (meta_params, meta_dict) = read_tinystories_pt(args.input_path)
    elif "TinyLlama" in args.input_path:
        (meta_params, meta_dict) = read_tinyllama(args.input_path)
    else:
        print("Unknown model type")
        exit()

    if args.t:
        n_records = 15
        export_lumi(meta_params, meta_dict, args.lumipath, n_records)
        (lumi_params, lumi_dict) = read_lumi(args.lumipath, n_records)
        compare(meta_params, meta_dict, lumi_params, lumi_dict, n_records)
    else:
        export_lumi(meta_params, meta_dict, args.lumipath)
