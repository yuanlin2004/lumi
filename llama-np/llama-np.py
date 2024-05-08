import argparse
import random
import signal
import sys
import time
import pickle

from decotimer import *

# from cProfile import Profile
from typing import List, Literal, Optional, Tuple, TypedDict

import numpy as np

from read_lumi import read_lumi
from tokenizer import ChatFormat, Tokenizer_Llama2, Tokenizer_Llama3
from transformer import *
from sysutil import *
from config import *


logger = lumi_logging.getLogger(__name__)


class Sampler:
    # topp "nucleus sampling" sampling
    def __init__(self, temperature, topp, save_history=False):
        self.temp = temperature
        self.topp = topp
        assert topp > 0 and topp <= 1, f"{topp} should be >0 and <=1"
        self.history = []
        self.save_history = save_history

    def __call__(self, logits):
        if self.temp == 0:
            return np.argmax(logits)  # scalar

        logits = Softmax(logits / self.temp, use="numpy")
        indices = np.argsort(-logits)
        values = logits[indices]
        cumsum = np.cumsum(values)

        # Find the first position in the cumsum where
        # the value is >= topp.
        bigger_than_topp = cumsum >= self.topp
        if np.any(bigger_than_topp):
            n = np.argmax(bigger_than_topp)
        else:
            # argmax would return 0 if none is >= topp (may due to fp rounding)
            # so we use np.any() to check
            n = logits.shape[0] - 1

        picked = indices[random.randint(0, n)]
        # print(f"Sampler - Cut @ {n}  Highest @ {indices[0]} with {values[0]}  Picked {picked} with {logits[picked]}")
        if self.save_history:
            self.history.append((n, indices[0], values[0], picked, logits[picked]))   
        return picked

    def add_str(self, s):
        if self.save_history:
            self.history.append(s)

    def save_history(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.history, f)

class Llama:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        temperature,
        topp,
        exp_args,
        seed: int = 134,
    ) -> "Llama":
        random.seed(seed)
        if ("llama-3" in model_path.lower()) or ("llama3" in model_path.lower()):
            self.tokenizer = Tokenizer_Llama3(model_path=tokenizer_path)
            self.llama_version = 3
        else:
            self.tokenizer = Tokenizer_Llama2(model_path=tokenizer_path)
            self.llama_version = 2
        (params, weight_dict) = read_lumi(model_path)

        print("Building the network . ", end="", flush=True)
        start_time = time.perf_counter()
        self.params = params
        # Note: some weights in weight_dict will be 'del'ed in the following call,
        # to keep the memory footprint small, as transposed copies will be made.
        self.model = Transformer(params, weight_dict, max_seq_len, exp_args)

        # just to tighten up the loose end
        del weight_dict

        self.max_seq_len = max_seq_len
        end_time = time.perf_counter()
        print(f"{end_time-start_time:0.4f} seconds")

        self.sampler = Sampler(temperature, topp, save_history=(exp_args.sample_history is not None))

    def generate(
        self,
        input_tokens: List[int],
        start_pos,
        print_dot,
        no_masking,
        use_cupy,  # list of tokens
    ) -> int:  # a token
        """
        Give a list of tokens converted from the input prompt, generate an output token
        """
        logits = self.model(input_tokens, start_pos, print_dot, no_masking, use_cupy)[-1]
        logger.debug(lambda: f"logits[0]: {logits[0]}")
        assert (
            len(logits) == self.params["vocab_size"]
        ), f"{len(logits)} vs {self.params['vocab_size']}"
        result = self.sampler(logits)
        return result

    def text_completion(
        self,
        input_tokens,
        exp_args,
        emit_one_token,
        no_masking=False,
        print_new_only=False,
    ):
        # This code is a bit convoluted because it deals with combinations of the following different cases:
        # 1. use kv cache or not
        # 2. when kv cache is not, feed the prompt tokens one a time or all together
        # 3. decode and print the generated token one a time or together with the previous tokens.
        #    - llama 2: need to decode and print all tokens together because of the sentencepiece tokenizer
        #    - llama 3: can decode and print one token at a time because of the tiktoken tokenizer
        #    - emit_one_token is used to control this behavior
        # 4. print the generated token only or the whole prompt and the generated token
        #    - print_new_only is used to control this behavior
        #    - print_new_only is useful in the chat mode because we don't want the print the whole history.  

        len_prompt = len(input_tokens)
        print(
            f"[max seq length: {self.max_seq_len}   length of input prompt: {len_prompt}]"
        )
        if exp_args.one_a_time:
            # feed the token in the prompt one a time
            output_tokens = []
            in_tokens = [input_tokens[0]]
        else:
            # feed the prompt tokens all together
            output_tokens = input_tokens[:]
            in_tokens = input_tokens[:]

        # The sentencepiece tokenizer (used by llama 2) does not emit space when given one token at
        # a time. So it is better to emit (decode and print) all tokens together.
        # The tiktoken tokenizer (used by llama 3) does not have this issue.
        if emit_one_token is None:
            # emit_one_token = (self.llama_version == 3)
            emit_one_token = False

        if emit_one_token and not print_new_only:
            s = self.tokenizer.decode(output_tokens)
            print(f"{s}", flush=True, end="")

        i = 0
        n_generated = 0
        all_start_time = time.perf_counter()
        while i < self.max_seq_len:
            start_time = time.perf_counter()
            generated_token = int(
                self.generate(
                    in_tokens, i, (not emit_one_token), no_masking, exp_args.use_cupy
                )
            )
            end_time = time.perf_counter()
            if not emit_one_token:
                print(f"{generated_token} {end_time-start_time:0.4f} seconds")
            n_generated += 1

            logger.debug(lambda: f"generated token: {generated_token}")
            if generated_token in self.tokenizer.stop_tokens:
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
            if emit_one_token:
                s = self.tokenizer.decode([generated_token])
                self.sampler.add_str(s)
                print(f"{s}", flush=True, end="")
            else:
                s = self.tokenizer.decode(output_tokens)
                print("")
                print(f"{s}", flush=True)
                print("")

            if exp_args.report_mem:
                report_mem(exp_args)

            if exp_args.no_kv_cache:
                in_tokens = output_tokens
            else:
                # kv cache is stateful, so only need to feed in the last token generated
                in_tokens = [generated_token]

        all_end_time = time.perf_counter()
        if emit_one_token:
            print()
        print(f"[{n_generated/(all_end_time - all_start_time):0.4f} tok/s]")

        if i >= self.max_seq_len:
            print(f"max {i} tokens reached")

        return output_tokens

    def gen_text(
        self,
        prompt_str: str,
        exp_args,
        emit_one_token,
        no_masking=False,
    ):
        input_tokens = self.tokenizer.encode(prompt_str, bos=True, eos=False)
        self.text_completion(input_tokens, exp_args, emit_one_token, no_masking)

    def chat(
        self,
        prompt_str: str,
        exp_args,
        emit_one_token,
        no_masking=False,
    ):
        # The chat mode goes through the following stages:
        # 1. Preemptive dialog
        #    - input: the preemptive dialog and the initial user prompt
        #    - output: the first response from the model (a list of new tokens)
        #    - text_completion()
        #      - Fill stage: starts with an empty kv cache and a list of input tokens, generate one token
        #      - Generate stage: kv cache is filled with the tokens generated so far, take in the previously
        #           generated token, generate a new token. Repeat until EoT.
        # 2. Continuous dialog
        #    - input: user's input
        #    - output: the response from the model
        #    - text_completion()
        #      - Fill stage: kv cache is filled with history, input is a list of tokens.
        #      - Generate stage: same as the one in the preemptive dialog.
        #
        # Currently, there is no such distinction in the code. Both stages are handled in the same way with a
        # clean kv cache. All the tokens generated so far are kept in a list and fed to the model at each turn
        # as the context. Therefore the generating the first token in each turn is a bit slower than having the
        # continuous dialog mode.
        chat_format = ChatFormat(self.tokenizer)
        preemptive_diaglog = [
            {"role": "system", "content": "Always answer precisely."},
            {"role": "user", "content": "Let's get started."},
            {"role": "assistant", "content": "I am ready to help you. Let's start."},
        ]
        all_tokens = []
        while True:
            
            command = input("> ").lstrip()
            if command[0] == "#":
                # command mode
                if command.lower() in ["#restart", "#reset", "#new"]:
                    continue
                elif command.lower() in ["#mem", "#reportmem", "#memory"]:
                    report_mem(exp_args)
                    continue
                elif command.lower() in ["#bye", "#quit", "#stop"]:
                    break
                elif command.lower() in ["#help"]:
                    print(
                        "[#restart, #reset, #new], [#mem, #reportmem, #memory], [#bye, #quit, #stop], #help"
                    )
                    continue

            diaglog = [{"role": "user", "content": command}]
            diaglog = preemptive_diaglog + diaglog
            tokens = chat_format.encode_dialog_prompt(diaglog)
            preemptive_diaglog = []

            self.model.restart()
            all_tokens.extend(tokens)
            all_tokens = self.text_completion(
                all_tokens, exp_args, True, no_masking, print_new_only=True
            )


def command_loop():
    while True:
        command = input("> ")
        if command.lower() in ["bye", "quit", "stop", "exit"]:
            return "exit"
        elif command.lower() in ["continue", "next", "cont", "con"]:
            return "continue"
        elif command.lower() in ["show"]:
            show("hello")
            continue
        elif command.lower() in ["mem", "reportmem", "memory"]:
            report_mem(exp_args)
            continue


def signal_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)  # ignore additional signals
    print()
    print(f"Signal handler called with signal {signum}. Lumi console entered:")
    result = command_loop()
    if result == "exit":
        sys.exit(34)
    elif result == "continue":
        signal.signal(signum, signal_handler)
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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", type=str, help="input prompt string")
    group.add_argument("-f", type=str, help="input prompt file")

    parser.add_argument(
        "--temp", type=float, default=0.6, help="temperature for the sampler"
    )
    parser.add_argument("--topp", type=float, default=0.9, help="topp for the sampler")
    parser.add_argument(
        "--chat",
        action="store_true",
        default=False,
        help="chat mode. Default is text completion.",
    )
    parser.add_argument(
        "--fill1",
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
        help="set the log level: DEBUG, INFO, WARN, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--nomask", action="store_true", help="do not use causal mask - just for play"
    )
    parser.add_argument(
        "--nokvcache", action="store_true", default=False, help="do not use kv cache"
    )
    parser.add_argument(
        "--useinplacekvcache",
        action="store_true",
        default=False,
        help="use in-place kv cache",
    )
    parser.add_argument("--timer", action="store_true", help="enable timer for methods")
    parser.add_argument("--cupy", action="store_true", default=False, help="use cupy")
    parser.add_argument("--sampler-history", type=str, dest="sample_history", default=False, help="dump the sampler history to a file")
    parser.add_argument(
        "--reportmem", action="store_true", default=False, help="report memory usage"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--emit-one-token",
        action="store_true",
        dest="emit_one_token",
        default=None,
        help="emit one token, default for llama 3 models",
    )
    group.add_argument(
        "--emit-all-tokens",
        action="store_false",
        dest="emit_one_token",
        default=None,
        help="emit all tokens, default for llama 2 models",
    )

    args = parser.parse_args()

    token_file = args.t
    weight_file = args.w

    if args.i:
        input_str = args.i
    else:
        with open(args.f, "r") as f:
            input_str = f.read()

    max_n_tokens = args.seqlength

    lumi_logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    decotimer_set(args.timer)
    exp_args = ExperimentArgs(
        no_kv_cache=args.nokvcache,
        one_a_time=args.fill1,
        report_mem=args.reportmem,
        use_in_place_kv_cache=args.useinplacekvcache,
        use_cupy=args.cupy,
    )

    if exp_args.report_mem:
        report_mem(exp_args)

    signal.signal(signal.SIGINT, signal_handler)

    llama = Llama(weight_file, token_file, max_n_tokens, args.temp, args.topp, exp_args)
    print()
    if exp_args.report_mem:
        report_mem(exp_args)
    if args.chat:
        llama.chat(input_str, exp_args, args.emit_one_token, no_masking=args.nomask)
    else:
        llama.gen_text(input_str, exp_args, args.emit_one_token, no_masking=args.nomask)

    if args.sample_history is not None:
        llama.sampler.save_history(args.sampler_history)