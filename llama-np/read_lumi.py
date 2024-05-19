import struct
import time

import numpy as np
from lumi_type import LumiDType

from sysutil import lumi_logger, lumi_logging


logger = lumi_logging.getLogger(__name__)


def read_padded_string(file):
    length_bytes = file.read(4)
    length = struct.unpack("I", length_bytes)[0]
    string_bytes = file.read(length)
    return string_bytes.decode().rstrip()  # Decode and remove any padding


def read_lumi(model_path, n_records=-1, skip_weight=False):
    start_time = time.perf_counter()
    print("Reading model file ", end="")

    logger.debug(lambda: f"Reading {model_path}")

    file = open(model_path, "rb")

    # 1. "lumi" magic
    magic = struct.unpack("I", file.read(4))[0]
    if magic != 0x696D756C:
        logger.error(lambda: "magic mis-match")

    # 2. version
    version = struct.unpack("I", file.read(4))[0]
    logger.debug(lambda: f"Version {version}")

    # 3. model name
    model_name = read_padded_string(file)
    logger.debug(lambda: f"Model name {model_name}")

    # 4. params
    lumi_params = {}
    p = struct.unpack("iiiiiiiff", file.read(36))
    logger.debug(lambda: f"params {p}")
    lumi_params["dim"] = p[0]
    lumi_params["n_layers"] = p[1]
    lumi_params["n_heads"] = p[2]
    lumi_params["n_kv_heads"] = p[3]
    lumi_params["multiple_of"] = p[4]
    lumi_params["vocab_size"] = p[5]
    lumi_params["max_seq_len"] = p[6]
    lumi_params["rope_theta"] = p[7]
    lumi_params["norm_eps"] = p[8]

    # 5. tokenizer model
    tokenizer_model_size = struct.unpack("I", file.read(4))[0]
    logger.debug(lambda: f"tokenizer model size {tokenizer_model_size}")
    tokenizer_model = file.read(tokenizer_model_size)
    # Skip the padding
    padding_length = 4 - len(tokenizer_model) % 4
    file.seek(file.tell() + padding_length)

    # 5. weight data
    n = 0
    dict = {}
    while True:
        print(".", end="", flush=True)

        # check if end of file is reached
        original_pos = file.tell()
        chunk = file.read(4)
        if not chunk:
            # end of the file
            logger.debug(lambda: "end of file")
            break
        file.seek(original_pos)

        # length_of_padded_name, padded_name_string
        name = read_padded_string(file)
        logger.debug(lambda: f"Reading weight {name}")

        # num of shape dim, size of each dim
        dim = struct.unpack("I", file.read(4))[0]
        shape = [struct.unpack("I", file.read(4))[0] for _ in range(dim)]
        logger.debug(lambda: f"shape {shape}")

        # transposed or not
        transposed = struct.unpack("I", file.read(4))[0]
        logger.debug(lambda: f"transposed {transposed}")

        # is bf16
        lumitype = struct.unpack("I", file.read(4))[0]
        is_bf16 = lumitype == LumiDType.bf16.value
        logger.debug(lambda: f"is_bf16 {is_bf16}")
        unit = 2 if is_bf16 else 4

        # weight data
        weight_len = np.prod(shape)
        if skip_weight:
            # Skip reading the weights. This is for test and development.
            weight = np.random.rand(weight_len).astype(np.float32)
            file.seek(file.tell() + weight_len * unit)
            weight = weight.reshape(shape)
        else:
            if transposed:
                shape = shape[::-1]
            if not is_bf16:
                weight = np.frombuffer(file.read(weight_len * unit), dtype=np.float32)
                weight = weight.reshape(shape)
            else:
                # read as int16, convert to float32
                shape2 = shape + [2]
                w = np.ndarray(shape2, dtype=np.int16)
                buffer = np.frombuffer(file.read(weight_len * unit), dtype=np.int16)
                buffer = buffer.reshape(shape)
                w[..., 1] = buffer
                w[..., 0] = 0
                weight = w.view(dtype=np.float32).squeeze()
            if transposed:
                weight = weight.transpose()  # keep this as a view
        dict[name] = weight

        n += 1
        if n_records != -1 and n >= n_records:
            break

    file.close()

    print(".", end="", flush=True)
    end_time = time.perf_counter()
    print(f" {end_time-start_time:0.4f} seconds")

    return lumi_params, tokenizer_model, dict, model_name
