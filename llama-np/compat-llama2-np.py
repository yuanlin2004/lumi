import argparse
import math
import random
import time

import numpy as np
from read_lumi import read_lumi
from tokenizer import Tokenizer_Llama3, Tokenizer_Llama2


class Linear:
    def __init__(self, weight, bias=None):
        self.weight, self.bias = weight.T, bias

    def __call__(self, x):
        y = np.matmul(x, self.weight)
        return y + self.bias if self.bias else y


class SiLU:
    def __call__(self, x):
        return x * (1 / (1 + np.exp(-x)))


class FeedForward:
    def __init__(self, w1, w2, w3):
        self.w1, self.w2, self.w3 = Linear(w1), Linear(w2), Linear(w3)
        self.silu = SiLU()

    def __call__(self, x):
        return self.w2(self.w3(x) * self.silu(self.w1(x)))


class RMSNorm:
    def __init__(self, weight, eps: float = 1e-5):
        self.eps, self.weight = eps, weight

    def __call__(self, x):
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class RoPE:
    def __init__(self, theta=10000):
        self.theta = theta

    def __call__(self, x, start_pos=0):
        dim, s = x.shape[-1], x.shape[-2]
        theta = self.theta ** (-2.0 * np.array([t // 2 for t in range(dim)]) / dim)
        m = np.arange(start_pos, s + start_pos).reshape(-1, 1)
        m_theta = m * theta  # outer product
        cos, sin = np.cos(m_theta), np.sin(m_theta)
        y = np.ndarray(x.shape)
        y[..., 1::2] = x[..., 0::2]  # in-place update
        y[..., 0::2] = -x[..., 1::2]  # in-place update
        return x * cos + y * sin


class Softmax:
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Attention:
    def __init__(self, wq_weight, wk_weight, wv_weight, wo_weight, pos_emb, n_heads, n_kv_heads):
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.softmax, self.pos_emb = Softmax(), pos_emb
        self.wq, self.wk = Linear(wq_weight), Linear(wk_weight)
        self.wv, self.wo = Linear(wv_weight), Linear(wo_weight)
        self.k_cache, self.v_cache = None, None

    def __call__(self, q, start_pos):
        xq, xk, xv = self.wq(q), self.wk(q), self.wv(q)

        (seq, dim) = xq.shape  # [seq, dim]
        head_size = dim // self.n_heads
        xxq = np.reshape(xq, (-1, self.n_heads, head_size)).transpose(
            1, 0, 2
        )  # (n_heads, s, head_size)
        xxk = np.reshape(xk, (-1, self.n_kv_heads, head_size)).transpose(
            1, 0, 2
        )  # (n_heads, s, head_size)
        xxv = np.reshape(xv, (-1, self.n_kv_heads, head_size)).transpose(
            1, 0, 2
        )  # (n_heads, s, head_size)

        xxq = self.pos_emb(xxq, start_pos)
        xxk = self.pos_emb(xxk, start_pos)

        xxk = np.moveaxis(
            xxk, -1, -2
        )  # same as xxk.transpose([0,2,1]) (n_heads, head_size, s)

        if self.k_cache is not None:
            xxk = np.concatenate((self.k_cache, xxk), axis=2)  # k is transposed
            xxv = np.concatenate((self.v_cache, xxv), axis=1)
        self.k_cache = xxk
        self.v_cache = xxv

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            xxk = np.repeat(xxk, rep, axis=0)
            xxv = np.repeat(xxv, rep, axis=0)

        scores = np.matmul(xxq, xxk)

        if seq > 1:  # masking
            minvalue = np.finfo(np.float32).min
            scores = scores + np.triu(np.full(scores.shape, minvalue), 1)

        scores_sm = self.softmax(scores / math.sqrt(head_size))
        value = np.matmul(scores_sm, xxv)
        return self.wo(np.reshape(value.transpose(1, 0, 2), (-1, dim)))


class TransformerBlock:
    def __init__(
        self,
        w_att_norm,
        n_heads,
        n_kv_heads,
        w_q,
        w_k,
        w_v,
        w_o,
        w_ffd_w1,
        w_ffd_w2,
        w_ffd_w3,
        w_ffd_norm,
        rope_theta,
    ):
        self.att_rmsnorm = RMSNorm(w_att_norm)
        self.attention = Attention(w_q, w_k, w_v, w_o, RoPE(rope_theta), n_heads, n_kv_heads)
        self.ffd_rmsnorm = RMSNorm(w_ffd_norm)
        self.feedforward = FeedForward(w_ffd_w1, w_ffd_w2, w_ffd_w3)

    def __call__(self, x, start_pos):
        x = x + self.attention(self.att_rmsnorm(x), start_pos)
        return x + self.feedforward(self.ffd_rmsnorm(x))


class Transformer:
    def __init__(self, params, weight_dict):
        self.embedding_tab = weight_dict["tok_embeddings.weight"]
        self.n_layers = params["n_layers"]
        self.transformer_blocks = []
        for i in range(self.n_layers):
            tf_block = TransformerBlock(
                weight_dict[f"layers.{i}.attention_norm.weight"],
                params["n_heads"],
                params["n_kv_heads"],
                weight_dict[f"layers.{i}.attention.wq.weight"],
                weight_dict[f"layers.{i}.attention.wk.weight"],
                weight_dict[f"layers.{i}.attention.wv.weight"],
                weight_dict[f"layers.{i}.attention.wo.weight"],
                weight_dict[f"layers.{i}.feed_forward.w1.weight"],
                weight_dict[f"layers.{i}.feed_forward.w2.weight"],
                weight_dict[f"layers.{i}.feed_forward.w3.weight"],
                weight_dict[f"layers.{i}.ffn_norm.weight"],
                params["rope_theta"]
            )
            self.transformer_blocks.append(tf_block)
        self.rmsnorm = RMSNorm(weight_dict["norm.weight"])
        self.lm_head = Linear(weight_dict["output.weight"])

    def __call__(self, input_tokens, start_pos):
        x = self.embedding_tab[input_tokens]
        for b in self.transformer_blocks:
            print(".", end="", flush=True)
            x = b(x, start_pos)
        return self.lm_head(self.rmsnorm(x))


class Llama:
    def __init__(self, model_path, tokenizer_path, max_seq_len, seed=34):
        random.seed(seed)
        if ('llama-3' in model_path.lower()) or ('llama3' in model_path.lower()):
            self.tokenizer = Tokenizer_Llama3(model_path=tokenizer_path)
        else:
            self.tokenizer = Tokenizer_Llama2(model_path=tokenizer_path)
        (params, weight_dict) = read_lumi(model_path)
        self.params = params
        self.model = Transformer(params, weight_dict)
        self.max_seq_len = max_seq_len

    def generate(self, input_tokens, start_pos):
        return self.sample(self.model(input_tokens, start_pos)[-1])

    def sample(self, logits):  # greedy
        return np.argmax(logits)  # scalar

    def text_completion(self, prompt_str):
        input_tokens = self.tokenizer.encode(prompt_str, bos=True, eos=False)

        len_prompt = len(input_tokens)
        output_tokens = input_tokens[:]
        in_tokens = input_tokens[:]

        i = 0
        all_start_time = time.perf_counter()
        while i < self.max_seq_len:
            generated_token = int(self.generate(in_tokens, i))

            if generated_token == self.tokenizer.eos_id:
                break
            i = i + (len_prompt if i == 0 else 1)
            output_tokens.append(generated_token)
            s = self.tokenizer.decode(output_tokens)
            print(f"\n{s}", flush=True)
            in_tokens = [generated_token]

        all_end_time = time.perf_counter()
        print(f"{(i-len_prompt)/(all_end_time - all_start_time):0.4f} tok/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, required=True, help="tokenizer model")
    parser.add_argument("-w", type=str, required=True, help="lumi model")
    parser.add_argument("-i", type=str, required=True, help="input prompt")
    parser.add_argument("--seqlength", type=int, default=128, help="max seq len")
    args = parser.parse_args()

    llama = Llama(args.w, args.t, args.seqlength)
    llama.text_completion(args.i)
