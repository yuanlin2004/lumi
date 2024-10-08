# This file basically merges
#   https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
# and
#   https://github.com/meta-llama/llama/blob/main/llama/tokenizer.py
# with some modifications to make it work with the current codebase.

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import tiktoken
from sentencepiece import SentencePieceProcessor
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)

#
# llama3
#

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

class HF_Tokenizer:

    def __init__(self, model_path: str, model_name: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, resume_download=True)
        self.stop_tokens = {self.tokenizer.eos_token_id}
        # all the special tokens are added by the AutoTokenizer already
        self.model_name = model_name
        self.special_tokens = dict(zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)) 

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.tokenizer.encode(s, add_special_tokens=False)
        if eos:
            t = t + [self.tokenizer.eos_token_id]
        return t

    def decode(self, t: List[int], skip_bos=False) -> str:
        return self.tokenizer.decode(t)
    

def hf_unicode_to_bytes():
    # This function is modified from HuggingFace's bytes_to_unicode() in tokenization_gpt2.py. 
    # https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/models/gpt2/tokenization_gpt2.py#L38
    # 
    # HF's BPE tokenizer handles the control characters differently from Tiktoken. 
    # - Tiktoken: control characters are mapped to the same byte value as the corresponding printable characters.
    # - HF: control characters are mapped to the byte values + 2**8
    # I do not fully understand why HF does this. Its comments says "avoids mapping to whitespace/control characters the bpe code barfs on."
    #
    # To use the vocab.json file from HF's GPT2 tokenizer with Tiktoken, I need to convert the byte values in the vocab.json file to the 
    # ones that Tiktoken uses. See the use of this function in Tokenizer_Qwen1_5.__init__().
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs.copy()
    i = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + i)
            i += 1
    cs = [chr(i) for i in cs]
    return dict(zip(cs, bs))

class Tokenizer_Qwen1_5:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_name: str, model_str: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_str (str): The str of the loaded Tiktoken model file.
        """
        import json
        
        unicode2bytes = hf_unicode_to_bytes() 

        data = json.loads(model_str)
        mergeable_ranks = {bytes([unicode2bytes[c] for c in k]):v for k,v in data.items()}

        num_base_tokens = len(mergeable_ranks)
        self.model_name = model_name
        special_tokens = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=model_name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.debug(f"Reloaded tiktoken model")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        #self.bos_id: int = self.special_tokens["<|endoftext|>"]
        self.eos_id: int = self.special_tokens["<|endoftext|>"]
        self.stop_tokens = {
            self.special_tokens["<|endoftext|>"],
            self.special_tokens["<|im_end|>"],
        }
        logger.debug(
        #    f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
            f"#words: {self.n_words} - BOS ID: N/A - EOS ID: N/A"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo2_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        # QWen models do not have bos_id and eos_id
        #if bos:
        #    t.insert(0, self.bos_id)
        #if eos:
        #    t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int], skip_bos=False) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        #if skip_bos and t[0] == self.bos_id:
        #    t = t[1:]
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

class Tokenizer_Llama3:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_name: str, model_str: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_str (str): The str of the loaded Tiktoken model file.
        """

        # copied from https://github.com/openai/tiktoken/blob/1b9faf2779855124f05174adf1383e53689ed94b/tiktoken/load.py#L148C5-L151C6
        import base64

        mergeable_ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in model_str.splitlines() if line)
        }

        num_base_tokens = len(mergeable_ranks)
        self.model_name = model_name
        if model_name == "qwen1.5-7b-chat":
            special_tokens = [
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
            ]
        else:
            special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|reserved_special_token_2|>",
                "<|reserved_special_token_3|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|reserved_special_token_4|>",
                "<|eot_id|>",  # end of turn
            ] + [
                f"<|reserved_special_token_{i}|>"
                for i in range(5, self.num_reserved_special_tokens - 5)
            ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=model_name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.debug(f"Reloaded tiktoken model")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        if model_name == "qwen1.5-7b-chat":
            self.bos_id: int = self.special_tokens["<|endoftext|>"]
            self.eos_id: int = self.special_tokens["<|endoftext|>"]
            self.stop_tokens = {
                self.special_tokens["<|endoftext|>"],
                self.special_tokens["<|im_end|>"],
            }
        else:
            self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
            self.eos_id: int = self.special_tokens["<|end_of_text|>"]
            self.pad_id: int = -1
            self.stop_tokens = {
                self.special_tokens["<|end_of_text|>"],
                self.special_tokens["<|eot_id|>"],
            }
        logger.debug(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int], skip_bos=False) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        if skip_bos and t[0] == self.bos_id:
            t = t[1:]
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


def GetChatFormat(tokenizer: Tokenizer_Llama3):
    if tokenizer.model_name in ["qwen1.5-7b-chat", "qwen2-0.5b-instruct", "qwen2-1.5b-instruct", "qwen2-7b-instruct"]:
        return ChatFormat_QWen(tokenizer)
    else:
        return ChatFormat(tokenizer)


#<|im_start|>system
#You are a helpful assistant.<|im_end|>
#<|im_start|>user
#message<|im_end|>
#<|im_start|>assistant
class ChatFormat_QWen:
    def __init__(self, tokenizer: Tokenizer_Llama3):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|im_start|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.extend(self.tokenizer.encode("\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|im_end|>"])
        tokens.extend(self.tokenizer.encode("\n", bos=False, eos=False))
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


class ChatFormat:
    def __init__(self, tokenizer: Tokenizer_Llama3):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


#
# llama2
#


class Tokenizer_Llama2:
    def __init__(self, model_name, model_buffer):
        self.model_name = model_name
        # reload tokenizer
        # Create a tmp file, store model_buffer in it, and use the file to create a SentencePieceProcessor.
        # This is a workaround for SentencePieceProcessor.load() not accepting a buffer.
        # I cannot figure out how to use the buffer with SentencePieceProcessor.LoadFromSerializedProto()
        import tempfile

        with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
            tmpfile.write(model_buffer)
            tmpfile.flush()
            self.sp_model = SentencePieceProcessor(model_file=tmpfile.name)
            logger.debug(f"Reloaded SentencePiece model from {tmpfile.name}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.debug(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        self.stop_tokens = {self.eos_id}

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int], skip_bos=False) -> str:
        # skip_bos is used in llama3 tokenizer. Add it here to make the interface consistent.
        return self.sp_model.decode(t)
