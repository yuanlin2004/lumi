import struct
import time
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


def read_lumi(model_path, n_records=-1, skip_weight=False):
    start_time = time.perf_counter()
    print("Reading model file ", end="")

    logger.debug(f"Reading {model_path}")

    file = open(model_path, "rb")

    # 1. "lumi" magic
    magic = struct.unpack("I", file.read(4))[0]
    if magic != 0x696D756C:
        logger.error("magic mis-match")

    # 2. version
    version = struct.unpack("I", file.read(4))[0]
    logger.debug(f"Version {version}")

    # 3. params
    lumi_params = {}
    p = struct.unpack("iiiiiif", file.read(28))
    logger.debug(f"params {p}")
    lumi_params["dim"] = p[0]
    lumi_params["n_layers"] = p[1]
    lumi_params["n_heads"] = p[2]
    lumi_params["multiple_of"] = p[3]
    lumi_params["vocab_size"] = p[4]
    lumi_params["max_seq_len"] = p[5]
    lumi_params["norm_eps"] = p[6]

    # 4. weight data
    n = 0
    dict = {}
    while True:
        print(".", end="", flush=True)

        # check if end of file is reached
        original_pos = file.tell()
        chunk = file.read(4)
        if not chunk:
            # end of the file
            logger.debug("end of file")
            break
        file.seek(original_pos)

        # length_of_padded_name, padded_name_string
        length_bytes = file.read(4)
        length = struct.unpack("I", length_bytes)[0]
        string_bytes = file.read(length)
        name = string_bytes.decode().rstrip()  # Decode and remove any padding
        logger.debug(f"Reading weight {name}")

        # num of shape dim, size of each dim
        dim = struct.unpack("I", file.read(4))[0]
        shape = [struct.unpack("I", file.read(4))[0] for _ in range(dim)]
        logger.debug(f"shape {shape}")

        # weight data
        weight_len = np.prod(shape)
        if skip_weight:
            # Skip reading the weights. This is for test and development.
            weight = np.random.rand(weight_len).astype(np.float32)
            file.seek(file.tell() + weight_len * 4)
        else:
            weight = np.frombuffer(file.read(weight_len * 4), dtype=np.float32)
        dict[name] = weight.reshape(shape)

        n += 1
        if n_records != -1 and n >= n_records:
            break

    file.close()

    print(".", end="", flush=True)
    end_time = time.perf_counter()
    print(f" {end_time-start_time:0.4f} seconds")

    return (lumi_params, dict)
