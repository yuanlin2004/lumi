import sys

# Adjust the input directories in `Weight File` and `Tokenizer Model` based on your settings
Table = """
| Model               | Size | Weight File                                                  | Tokenizer Model                                                       | Model Name               |
| ------------------- | ---- | ------------------------------------------------------------ | --------------------------------------------------------------------- | -------------------------|
| TinyStories         | 260k | /mnt/data1t/DL-Models/TinyStories/stories260K/stories260K.pt | /mnt/data1t/DL-Models/TinyStories/stories260K/tok512.model            | stories260k              |
| TinyStories         | 15M  | /mnt/data1t/DL-Models/TinyStories/stories15M.pt              | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | stories15m               |
| TinyStories         | 42M  | /mnt/data1t/DL-Models/TinyStories/stories42M.pt              | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | stories42m               |
| TinyStories         | 110M | /mnt/data1t/DL-Models/TinyStories/stories110M.pt             | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | stories110m              |
| TinyLlama           | 1.1B | /mnt/data1t/DL-Models/TinyLlama/1.1B-3T                      | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | tinyllama-1.1b-3t        |
| TinyLlama-Chat      | 1.1B | /mnt/data1t/DL-Models/TinyLlama/1.1B-Chat-v1.0               | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | tinyllama-1.1b-chat-v1.0 |
| Llama-2             | 7B   | /mnt/data1t-2/DL-Models/Llama/Llama2/llama-2-7b              | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | llama-2-7b               |
| Llama-2-Chat        | 7B   | /mnt/data1t-2/DL-Models/Llama/Llama2/llama-2-7b-chat         | /mnt/data1t-2/DL-Models/Llama/Llama2/tokenizer.model                  | llama-2-7b-chat          |
| Llama-3             | 8B   | /mnt/data1t/DL-Models/llama3/Meta-Llama-3-8B                 | /mnt/data1t/DL-Models/llama3/Meta-Llama-3-8B/tokenizer.model          | llama-3-8b               |
| Llama-3-Instruct    | 8B   | /mnt/data1t/DL-Models/llama3/Meta-Llama-3-8B-Instruct        | /mnt/data1t/DL-Models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model | llama-3-8b-instruct      |
| Qwen1.0-7B-Chat     | 7B   | /mnt/data1t/DL-Models/Qwen/Qwen-7B-Chat                      | /mnt/data1t/DL-Models/Qwen/Qwen-7B-Chat/qwen.tiktoken                 | qwen1.0-7b-chat          | 
| Qwen1.5-7B-Chat     | 7B   | /mnt/data1t/DL-Models/Qwen/Qwen1.5-7B-Chat                   | /mnt/data1t/DL-Models/Qwen/Qwen1.5-7B-Chat/vocab.json                 | qwen1.5-7b-chat          | 
| Qwen2-0.5B-Instruct | 0.5B | /mnt/data1t/DL-Models/Qwen/Qwen2-0.5B-Instruct               | /mnt/data1t/DL-Models/Qwen/Qwen2-0.5B-Instruct/vocab.json             | qwen2-0.5b-instruct      | 
| Qwen2-1.5B-Instruct | 1.5B | /mnt/data1t/DL-Models/Qwen/Qwen2-1.5B-Instruct               | /mnt/data1t/DL-Models/Qwen/Qwen2-1.5B-Instruct/vocab.json             | qwen2-1.5b-instruct      | 
| Qwen2-7B-Instruct   | 7B   | /mnt/data1t/DL-Models/Qwen/Qwen2-7B-Instruct                 | /mnt/data1t/DL-Models/Qwen/Qwen2-7B-Instruct/vocab.json               | qwen2-7b-instruct        | 
"""

# Adjust the output directory based on your settings
LumiDir = "/home/yuan/DLModels/lumi"

import os

run = False
if len(sys.argv) == 2 and sys.argv[1] == "run":
    run = True

Lines = Table.split("\n")

for line in Lines[3:-1]:
    parts = line.split("|")
    model = parts[1].strip()
    size = parts[2].strip()
    precision = parts[3].strip()
    data_type = parts[4].strip()
    original_version = parts[3].strip()
    tokenizer_model = parts[4].strip()
    model_name = parts[5].strip()
    command = f"python3 convert.py {model_name} {original_version} {tokenizer_model} {LumiDir}/{model_name}.lmw"
    print(command)
    if run:
        os.system(command)

if not run:
    print()
    print("Please run this script with 'run' as the argument to convert the models. For example")
    print(f"% python {sys.argv[0]} run")