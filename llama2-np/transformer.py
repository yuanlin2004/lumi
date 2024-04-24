import math
from logging import getLogger

import numpy as np

from decotimer import *
from config import *

logger = getLogger(__name__)


class Linear:
    def __init__(self, weight, bias=None):
        # weight is always 2-D
        # the current implementation does not support
        self.weight = weight.T
        self.bias = bias
        return

    @decotimer
    def __call__(self, x):
        y = np.matmul(x, self.weight)
        if self.bias:
            y = y + self.bias
        return y


class SiLU:
    def __call__(self, x):
        return x * (1 / (1 + np.exp(-x)))


class FeedForward:
    def __init__(self, w1, w2, w3):
        self.w1 = Linear(w1)
        self.w2 = Linear(w2)
        self.w3 = Linear(w3)
        self.silu = SiLU()
        return

    @decotimer
    def __call__(self, x):
        return self.w2(self.w3(x) * self.silu(self.w1(x)))


class RMSNorm:
    def __init__(self, weight, eps: float = 1e-5):
        self.eps = eps
        self.weight = weight

    def __call__(self, x):
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class RoPE:
    def __init__(self, theta=10000):
        self.theta = theta

    @decotimer
    def __call__(self, x, start_pos=0):
        dim = x.shape[-1]
        s = x.shape[-2]
        theta = self.theta ** (-2.0 * np.array([t // 2 for t in range(dim)]) / dim)
        m = np.arange(start_pos, s + start_pos).reshape(-1, 1)
        m_theta = m * theta  # outer product
        cos = np.cos(m_theta)
        sin = np.sin(m_theta)
        y = np.ndarray(x.shape)
        y[..., 1::2] = x[..., 0::2]  # in-place update
        y[..., 0::2] = -x[..., 1::2]  # in-place update
        result = x * cos + y * sin
        return result


class Softmax:
    @decotimer
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Attention:
    def __init__(
        self,
        wq_weight,
        wk_weight,
        wv_weight,
        wo_weight,
        max_seq_len,
        exp_args,
        pos_emb,
        n_heads,
        n_kv_heads,
    ):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.softmax = Softmax()
        self.wq_matmul = Linear(wq_weight)
        self.wk_matmul = Linear(wk_weight)
        self.wv_matmul = Linear(wv_weight)
        self.wo_matmul = Linear(wo_weight)
        self.pos_emb = pos_emb
        self.no_kv_cache = exp_args.no_kv_cache
        self.use_in_place_kv_cache = exp_args.use_in_place_kv_cache
        self.k_cache = None
        self.v_cache = None
        if not exp_args.no_kv_cache:
            if exp_args.use_in_place_kv_cache:
                head_size = wq_weight.shape[0] // n_heads
                assert head_size == wk_weight.shape[0] // n_kv_heads
                assert head_size == wv_weight.shape[0] // n_kv_heads
                self.k_cache = np.zeros(
                    [n_kv_heads, head_size, max_seq_len], dtype=np.float32
                )
                self.v_cache = np.zeros(
                    [n_kv_heads, max_seq_len, head_size], dtype=np.float32
                )
        return

    @decotimer
    def __call__(self, q, start_pos, no_masking, kv=None):
        # self attention when kv is None
        if kv is None:
            kv = q
        xq = self.wq_matmul(q)
        xk = self.wk_matmul(kv)
        xv = self.wv_matmul(kv)

        (seq, dim) = xq.shape  # [seq, dim]
        head_size = dim // self.n_heads
        xxq = np.reshape(xq, (-1, self.n_heads, head_size)).transpose(
            1, 0, 2
        )  # (n_heads, s, head_size)
        xxk = np.reshape(xk, (-1, self.n_kv_heads, head_size)).transpose(
            1, 0, 2
        )  # (n_kv_heads, s, head_size)
        xxv = np.reshape(xv, (-1, self.n_kv_heads, head_size)).transpose(
            1, 0, 2
        )  # (n_kv_heads, s, head_size)

        xxq = self.pos_emb(xxq, start_pos)
        xxk = self.pos_emb(xxk, start_pos)

        xxk = np.moveaxis(
            xxk, -1, -2
        )  # same as xxk.transpose([0,2,1]) (n_kv_heads, head_size, s)

        if not self.no_kv_cache:
            if self.use_in_place_kv_cache:
                self.k_cache[:, :, start_pos : start_pos + seq] = xxk
                self.v_cache[:, start_pos : start_pos + seq, :] = xxv
                xxk = self.k_cache[:, :, : start_pos + seq]
                xxv = self.v_cache[:, : start_pos + seq, :]
            else:
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
        # print(f"score before masking {scores}")

        # apply masking
        logger.debug(f"no_masking {no_masking}")
        if (not no_masking) and seq > 1:
            # Given a single row, np.triu will expand the row into a square matrix
            # and then apply triu. Not what we need here. Therefore guard this with
            # seq > 1.
            minvalue = np.finfo(np.float32).min
            mask = np.triu(np.full(scores.shape, minvalue), 1)
            # print(f"masking {mask}")
            scores = scores + mask

        # print(f"score after masking {scores}")

        scores_sm = self.softmax(scores / math.sqrt(head_size))
        value = np.matmul(scores_sm, xxv)
        value = np.reshape(value.transpose(1, 0, 2), (-1, dim))
        result = self.wo_matmul(value)
        return result


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
        max_seq_len: int,
        rope_theta,
        exp_args,
    ):
        self.att_rmsnorm = RMSNorm(w_att_norm)
        self.attention = Attention(
            w_q, w_k, w_v, w_o, max_seq_len, exp_args, RoPE(rope_theta), n_heads, n_kv_heads
        )
        self.ffd_rmsnorm = RMSNorm(w_ffd_norm)
        self.feedforward = FeedForward(w_ffd_w1, w_ffd_w2, w_ffd_w3)
        return

    def __call__(self, x, start_pos, no_masking):
        # x = x + self.attention(self.att_rmsnorm(x))
        norm = self.att_rmsnorm(x)
        logger.debug(f"att rmsnorm [50] {norm[0][50]}")
        att = self.attention(norm, start_pos, no_masking)
        logger.debug(f"att [50] {att[0][50]}")
        x = x + att
        logger.debug(f"rescon1 [50] {x[0][50]}")

        # x = x + self.feedforward(self.ffd_rmsnorm(x))
        norm = self.ffd_rmsnorm(x)
        logger.debug(f"ffd rmsnorm [50] {norm[0][50]}")
        ffd = self.feedforward(norm)
        logger.debug(f"ffd [50] {ffd[0][50]}")
        x = x + ffd
        logger.debug(f"rescon2 [50] {x[0][50]}")
        return x


class Transformer:
    def __init__(self, params, weight_dict, max_seq_len, exp_args):
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
                max_seq_len,
                params["rope_theta"],
                exp_args,
            )
            self.transformer_blocks.append(tf_block)
        self.rmsnorm = RMSNorm(weight_dict["norm.weight"])
        self.lm_head = Linear(weight_dict["output.weight"])
        return

    def __call__(self, input_tokens, start_pos, no_masking):
        logger.debug(f"input tokens: {input_tokens}")
        x = self.embedding_tab[input_tokens]
        logger.debug(f"input embedding [50]: {x[0][50]}")
        i = 0
        for b in self.transformer_blocks:
            print(".", end="", flush=True)
            logger.debug(f"== layer {i} ==")
            x = b(x, start_pos, no_masking)
            i += 1
        return self.lm_head(self.rmsnorm(x))
