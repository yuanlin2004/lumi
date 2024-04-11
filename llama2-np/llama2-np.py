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
        seed: int = 34,
    ) -> "Llama":
        random.seed(seed)
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        (params, weight_dict) = read_lumi(model_path, skip_weight=False)
        # (params, weight_dict) = read_lumi(model_path, skip_weight=True)
        self.params = params
        self.model = Transformer(params, weight_dict)

    def generate(
        self, input_tokens: List[int], start_pos, no_masking  # list of tokens
    ) -> int:  # a token
        """
        Give a list of tokens converted from the input prompt, generate an output token
        """
        logits = self.model(input_tokens, start_pos, no_masking)[-1]
        logger.debug(f"logits[0]: {logits[0]}")
        assert len(logits) == self.params["vocab_size"]
        result = self.sample(logits)
        return result

    def sample(self, logits):
        assert len(logits) == self.params["vocab_size"]
        # Just give the top one for now
        result = np.argmax(logits)  # scalar
        return result

    def text_completion(
        self,
        prompt_str: str,
        max_num_gen_tokens: int,
        one_a_time=False,
        no_masking=False,
    ):
        """
        Given an inpout string (prompt), generate and print out a text string for completion
        """
        input_tokens = self.tokenizer.encode(prompt_str, bos=True, eos=False)

        # fix me to be limited by context window size
        max_tokens = max_num_gen_tokens

        len_prompt = len(input_tokens)
        if one_a_time:
            # feed the token in the prompt one a time
            output_tokens = []
            in_tokens = [input_tokens[0]]
        else:
            # feed the prompt tokens all together
            output_tokens = input_tokens[:]
            in_tokens = input_tokens[:]

        i = 0
        while i < (max_tokens - len_prompt + 1):
            start_time = time.perf_counter()
            generated_token = int(self.generate(in_tokens, i, no_masking))
            end_time = time.perf_counter()
            print(f" {end_time-start_time:0.4f} seconds")

            logger.debug(f"generated token: {generated_token}")
            if generated_token == self.tokenizer.eos_id:
                break
            if one_a_time:
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

            # the generate() is stateful, so only need to feed in the last token generated
            in_tokens = [generated_token]

        return


if __name__ == "__main__":

    max_n_tokens = 64

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
        help="force one token at a time in the fill stage",
    )
    parser.add_argument(
        "--seqlength", type=int, dest="max_n_tokens", help="max sequence length"
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
        "--timer", action="store_true", help="enable timer for methods"
    )
    args = parser.parse_args()

    token_file = args.t
    weight_file = args.w

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    decotimer_set(args.timer)

    llama = Llama(weight_file, token_file, 2048)
    print()
    llama.text_completion(
        args.i, max_n_tokens, one_a_time=args.fill1, no_masking=args.nomask
    )
