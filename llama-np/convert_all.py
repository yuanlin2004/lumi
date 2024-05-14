Table = """
| Model            | Size | Precision | Type    | Original Version                                             | Tokenizer Model                                                       | Lumi Version                 |
| ---------------- | ---- | --------- | ------- | ------------------------------------------------------------ | --------------------------------------------------------------------- | -----------------------------|
| TinyStories      | 260k |           | llama-2 | /mnt/data1t/DL-Models/TinyStories/stories260K/stories260K.pt | /mnt/data1t/DL-Models/TinyStories/stories260K/tok512.model            | stories260k.lmw              |
| TinyStories      | 15M  |           | llama-2 | /mnt/data1t/DL-Models/TinyStories/stories15M.pt              | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | stories15m.lmw               |
| TinyStories      | 42M  |           | llama-2 | /mnt/data1t/DL-Models/TinyStories/stories42M.pt              | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | stories42m.lmw               |
| TinyStories      | 110M |           | llama-2 | /mnt/data1t/DL-Models/TinyStories/stories110M.pt             | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | stories110m.lmw              |
| TinyLlama        | 1.1B | bf16      | llama-2 | /mnt/data1t/DL-Models/TinyLlama/1.1B-3T                      | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | tinyllama-1.1b-3t.lmw        |
| TinyLlama-Chat   | 1.1B | bf16      | llama-2 | /mnt/data1t/DL-Models/TinyLlama/1.1B-Chat-v1.0               | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | tinyllama-1.1b-chat-v1.0.lmw |
| Llama-2          | 7B   | bf16      | llama-2 | /mnt/data1t-2/DL-Models/Llama/Llama2/llama-2-7b              | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | llama-2-7b.lmw               |
| Llama-2-Chat     | 7B   | bf16      | llama-2 | /mnt/data1t-2/DL-Models/Llama/Llama2/llama-2-7b-chat         | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | llama-2-7b-chat.lmw          |
| Llama-3          | 8B   | bf16      | llama-3 | /mnt/data1t/DL-Models/llama3/Meta-Llama-3-8B                 | /mnt/data1t/DL-Models/llama3/Meta-Llama-3-8B/tokenizer.model          | llama-3-8b.lmw               |
| Llama-3-Instruct | 8B   | bf16      | llama-3 | /mnt/data1t/DL-Models/llama3/Meta-Llama-3-8B-Instruct        | /mnt/data1t/DL-Models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model | llama-3-8b-instruct.lmw      |
"""

LumiDir = "/home/yuan/DLModels/lumi"

import os

Lines = Table.split("\n")

for line in Lines[3:-1]:
    parts = line.split("|")
    model = parts[1].strip()
    size = parts[2].strip()
    precision = parts[3].strip()
    data_type = parts[4].strip()
    original_version = parts[5].strip()
    tokenizer_model = parts[6].strip()
    lumi_version = parts[7].strip()
    command = f"python3 convert.py {original_version} {tokenizer_model} {LumiDir}/{lumi_version}"
    print(command)
    os.system(command)