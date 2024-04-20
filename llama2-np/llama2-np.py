import argparse
import logging
import os
import random
import time

from decotimer import *
# from cProfile import Profile
from typing import List, Literal, Optional, Tuple, TypedDict

import numpy as np

from read_lumi import read_lumi
from tokenizer import Tokenizer
from transformer import *
from sysutil import *
from config import *


logger = logging.getLogger(__name__)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]


class Llama:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        exp_args,
        seed: int = 34,
    ) -> "Llama":
        random.seed(seed)
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        (params, weight_dict) = read_lumi(model_path) 
        self.params = params
        self.model = Transformer(params, weight_dict, exp_args)
        self.max_seq_len = max_seq_len

    def generate(
        self, input_tokens: List[int], start_pos, no_masking  # list of tokens
    ) -> int:  # a token
        """
        Give a list of tokens converted from the input prompt, generate an output token
        """
        logits = self.model(input_tokens, start_pos, no_masking)[-1]
        logger.debug(f"logits[0]: {logits[0]}")
        assert len(logits) == self.params["vocab_size"], f"{len(logits)} vs {self.params['vocab_size']}"
        result = self.sample(logits)
        return result

    def sample(self, logits):
        assert len(logits) == self.params["vocab_size"], f"{len(logits)} vs {self.params['vocab_size']}"
        # Just give the top one for now
        result = np.argmax(logits)  # scalar
        return result

    def text_completion(
        self,
        prompt_str: str,
        exp_args,
        no_masking=False,
    ):
        """
        Given an inpout string (prompt), generate and print out a text string for completion
        """
        input_tokens = self.tokenizer.encode(prompt_str, bos=True, eos=False)

        len_prompt = len(input_tokens)
        print(f"max seq lenght: {self.max_seq_len}   lenght of input prompt: {len_prompt}")
        if exp_args.one_a_time:
            # feed the token in the prompt one a time
            output_tokens = []
            in_tokens = [input_tokens[0]]
        else:
            # feed the prompt tokens all together
            output_tokens = input_tokens[:]
            in_tokens = input_tokens[:]

        i = 0
        n_generated = 0
        all_start_time = time.perf_counter()
        while i < self.max_seq_len :
            start_time = time.perf_counter()
            generated_token = int(self.generate(in_tokens, i, no_masking))
            end_time = time.perf_counter()
            print(f" {end_time-start_time:0.4f} seconds")
            n_generated +=1

            logger.debug(f"generated token: {generated_token}")
            if generated_token == self.tokenizer.eos_id:
                print("EOS generated")
                break
            if exp_args.one_a_time:
                if i < (len_prompt - 1):
                    generated_token = input_tokens[i + 1]
                i += 1
            else:
                if i == 0:
                    i += len_prompt
                else:
                    i += 1
            output_tokens.append(generated_token)
            s = self.tokenizer.decode(output_tokens)
            print("")
            print(f"{s}", flush=True)
            print("")

            if exp_args.report_mem:
                report_mem()

            if exp_args.no_kv_cache:
                in_tokens = output_tokens
            else:
                # kv cache is stateful, so only need to feed in the last token generated
                in_tokens = [generated_token]

        all_end_time = time.perf_counter()
        print(f"{n_generated/(all_end_time - all_start_time):0.4f} tok/s")
        
        if i >= self.max_seq_len:
            print(f"max {i} tokens reached")

        return


if __name__ == "__main__":

    max_n_tokens = 128

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", type=str, required=True, help="input path of the tokenizer model"
    )
    parser.add_argument(
        "-w", type=str, required=True, help="input path of the lumi model"
    )
    parser.add_argument("-i", type=str, required=True, help="input prompt")
    parser.add_argument(
        "-fill1",
        action="store_true",
        default=False,
        help="force one token at a time in the fill stage",
    )
    parser.add_argument(
        "--seqlength", type=int, default=max_n_tokens, help="max sequence length"
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        help="set the loog level: DEBUG, INFO, WARN, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--nomask", action="store_true", help="do not use causal mask - just for play"
    )
    parser.add_argument(
        "--nokvcache", action="store_true", default=False, help="do not use kv cache"
    )
    parser.add_argument(
        "--timer", action="store_true", help="enable timer for methods"
    )
    parser.add_argument(
        "--reportmem", action="store_true", default=False, help="report memory usage"
    )
    args = parser.parse_args()

    token_file = args.t
    weight_file = args.w

    max_n_tokens = args.seqlength

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    decotimer_set(args.timer)
    exp_args = ExperimentArgs(no_kv_cache=args.nokvcache, one_a_time=args.fill1, report_mem=args.reportmem)

    if exp_args.report_mem:
        report_mem()
    llama = Llama(weight_file, token_file, max_n_tokens, exp_args)
    print()
    if exp_args.report_mem:
        report_mem()
    llama.text_completion(
        args.i, exp_args, no_masking=args.nomask
    )
