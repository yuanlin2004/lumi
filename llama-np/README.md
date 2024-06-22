# `llama-np.py`

This is to play with models based on Meta's llama2 and llama3. The script `llama-np.py` implements an inference flow using `numpy` and `cupy` only, except for the tokenizers. This is a hobby project, just for fun. 

Due to the lack of support for lower precision and bf16 formats in `numpy` and `cupy`, fp32 is used. `llama-np.py` can execute the llama-3-8B model at approximately 1.2 tokens per second (tok/s) on CPU and 2.24 tok/s on GPU (AMD Ryzen 7700x CPU with over 32GB RAM and an RTX 4070 GPU). It is fast enough to play with. 

llama-np.py offers two operational modes:

1. Text completion (default) for prompt completions.
2. Chat (activated with the `--chat` option) for engaging in dialogues with the model.

It supports various experimental features:

- Enabling or disabling the key-value (kv) cache
- Methods for updating the kv cache: in-place or concatenation
- Prefill strategies for input tokens: all tokens in the input prompt at once or one token at a time
- Option to use cupy 
- An interactive mode in the sampler that let you pick the winning token manually

A simplified version `compact-llama-np.py` (less than 250 lines of code) is also provided to show the network architecture more clearly. 

#### Example of text completion
```sh
% python llama-np.py -w ./Meta-Llama-3-8B.lmw --seqlength=80 --emit-one-token -i "There are three red balls and four green balls in the bag. If I take out" 

There are three red balls and four green balls in the bag. If I take out a ball at random, what is the probability that it is green?

There are a total of seven balls.  Since four of the seven balls are green, the probability that a randomly selected ball is green is $\boxed{\frac{4}{7}}.$
Final Answer: The final answer is \frac{4}{7}. I hope it is correct.
[1.3231 tok/s]
...
```

#### Example of chat
``` sh
% python llama-np.py -w ./llama-3-8b-instruct.lmw --chat --seqlength 1024 
> Who was the US president in 2019?
The US president in 2019 was Donald Trump. He was the 45th President of the United States and served from January 20, 2017, to January 20, 2021.

> Who will be in 2025?
As of 2023, it is difficult to predict with certainty who will be the US President in 2025. However, I can provide some context.

The 2024 US presidential election is scheduled to take place on November 3, 2024. The incumbent ...
```

#### Example of Chinese support
```sh
% python llama-np.py -w ./qwen1.0-7b-chat.lmw -i ''\''曲径通幽处'\''的下一句是' --seqlength 1024 --temp 0
...
'曲径通幽处'的下一句是'禅房花木深'。'曲径通幽处，禅房花木深'出自唐代诗人常建的《题破山寺后禅院》。全诗如下： ...
```


## Supported Models

| Model            | Size | Model Name               |
| ---------------- | ---- | -------------------------|
| TinyStories      | 260k | stories260k              |
| TinyStories      | 15M  | stories15m               |
| TinyStories      | 42M  | stories42m               |
| TinyStories      | 110M | stories110m              |
| TinyLlama        | 1.1B | tinyllama-1.1b-3t        |
| TinyLlama-Chat   | 1.1B | tinyllama-1.1b-chat-v1.0 |
| Llama-2          | 7B   | llama-2-7b               |
| Llama-2-Chat     | 7B   | llama-2-7b-chat          |
| Llama-3          | 8B   | llama-3-8b               |
| Llama-3-Instruct | 8B   | llama-3-8b-instruct      |
| Qwen1.0-7B-Chat  | 7B   | qwen1.0-7b-chat          | 
| Qwen1.5-7B-Chat  | 7B   | qwen1.5-7b-chat          | 
| Qwen2-0.5B-Instruct | 0.5B   | qwen2-0.5b-instruct   | 
| Qwen2-1.5B-Instruct | 1.5B   | qwen2-1.5b-instruct   | 
| Qwen2-7B-Instruct | 7B   | qwen2-7b-instruct   | 


## Dependencies

The following packages are needed.

- python=3.11
- safetensors
- pytorch
- psutil
- sentencepiece
- conda-forge::tiktoken
- conda-forge::cupy
- nvtx
- psutil

Pytorch and safetensors are needed for model conversion only.

```sh
% conda create -f environment.yml
% conda activate lumi
```

## How to Run the Models
Running a model involves two primary steps: converting the weights into a compatible format and running the model itself.

### Weight Conversion
Weights need to be converted into the `.lmw` (lumi weight) format. The `convert.py` script can be used for such conversions. It contains many model-specific logics. The `convert_all.py` script (which calls `convert.py`) converts all supported models.

The tokenizer models will also be embedded in the `.lmw` files for ease of use.

```sh
% python convert.py <model_name> <model_path> <tokenizer_model_path> <lumi_weight_path>
```

### Model Execution
Once weights are converted, the `llama-np.py` script performs text generation using the input string and the converted weight file.

### Running Models
Given a weight file (`*.lmw`) and an initial text string, `llama-np.py` performs text generation based on the input string.

```
usage: llama-np.py [-h] -w W (-i I | -f F | --chat) [--temp TEMP] [--topp TOPP] [--seed SEED] [--fill1] [--seqlength SEQLENGTH] [--loglevel LOGLEVEL] [--nomask] [--nokvcache] [--useinplacekvcache] [--timer] [--cupy]
                   [--sampler-history SAMPLE_HISTORY] [--reportmem] [--you-pick] [--emit-one-token | --emit-all-tokens]

options:
  -h, --help            show this help message and exit
  -w W                  lumi weight file
  -i I                  prompt string
  -f F                  prompt file
  --chat                chat mode. Default is text completion.
  --temp TEMP           temperature (value in [0.0, 1.0]) for the topp sampler, default 0.6. 0 will use argmax.
  --topp TOPP           topp value (in [0.0, 1.0]) for the topp sampler, default 0.9
  --seed SEED           seed for the random number generator
  --fill1               force one token at a time in the fill stage
  --seqlength SEQLENGTH
                        max sequence length
  --loglevel LOGLEVEL   set the log level: DEBUG, INFO, WARN, ERROR, CRITICAL
  --nomask              do not use causal mask - just for play
  --nokvcache           do not use kv cache
  --useinplacekvcache   use in-place kv cache
  --timer               enable timer for methods
  --cupy                use cupy
  --sampler-history SAMPLE_HISTORY
                        dump the sampler history to a file
  --reportmem           report memory usage
  --you-pick            you pick the candidate in the sampler
  --emit-one-token      emit one token, default for llama 3 models
  --emit-all-tokens     emit all tokens, default for llama 2 models
```

## Model Specific Information

### Llama-3

#### Download the models
Get the model files from Meta per instructions on https://github.com/meta-llama/llama3/blob/main/README.md

#### Convert

```sh
% python convert.py llama-3-8b <path>/Meta-Llama-3-8B <path>/Meta-Llama-3-8B/tokenizer.model <path>/Meta-Llama-3-8B/llama-3-8b.lmw 
```

#### Inference Example
```sh
% python llama-np.py -w <path>/Meta-Llama-3-8B/llama-3-8b.lmw -i "It is easy"

It is easy to get caught up in the excitement of the holiday season. The decorations, the ...
```

### Llama-2

#### Download the models
Get the model files from Meta per instructions on https://github.com/meta-llama/llama/blob/main/README.md

#### Convert

```sh
% python convert.py llama-2-7b.lmw <path>/Llama2/llama-2-7b <path>/Llama2/tokenizer.model <path>/Llama2/llama-2-7b/llama-2-7b.lmw 
```

#### Inference Example
```sh
% python llama-np.py -w <path>/Llama2/llama-2-7b/llama-2-7b.lmw -i "It is easy"

It is easy to get lost in the world of the internet. It is easy ...
```

### QWen1.0

#### Download the models

Get the model from https://huggingface.co/Qwen/Qwen-7B-Chat

```sh
% GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen-7B-Chat
% cd Qwen-7B-Chat
% wget https://huggingface.co/Qwen/Qwen-7B-Chat/resolve/main/model-00001-of-00008.safetensors
% wget https://huggingface.co/Qwen/Qwen-7B-Chat/resolve/main/model-00002-of-00008.safetensors
% wget https://huggingface.co/Qwen/Qwen-7B-Chat/resolve/main/model-00003-of-00008.safetensors
% wget https://huggingface.co/Qwen/Qwen-7B-Chat/resolve/main/model-00004-of-00008.safetensors
% wget https://huggingface.co/Qwen/Qwen-7B-Chat/resolve/main/model-00005-of-00008.safetensors
% wget https://huggingface.co/Qwen/Qwen-7B-Chat/resolve/main/model-00006-of-00008.safetensors
% wget https://huggingface.co/Qwen/Qwen-7B-Chat/resolve/main/model-00007-of-00008.safetensors
% wget https://huggingface.co/Qwen/Qwen-7B-Chat/resolve/main/model-00008-of-00008.safetensors
```

#### Convert

```sh
% python3 convert.py qwen1.0-7b-chat <path>Qwen-7B-Chat <path>Qwen-7B-Chat/qwen.tiktoken <path>/qwen1.0-7b-chat.lmw
```

#### Inference Example
```sh
% python llama-np.py -w /home/yuan/DLModels/lumi/qwen1.0-7b-chat.lmw -i "我是孙悟空" --seqlength 16 --emit-one

我是孙悟空的妈妈，今天我想和大家分享一下我儿子孙悟空的教育心得
```
Sigh :(




### TinyLlama

#### Download the models

##### TinyLlama-1.1B-Chat-v1.0
Based on this page https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/model.safetensors
```
% wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
```

##### TinyLlama-1.1B-intermediate-step-1431k-3T
Based on this page https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/blob/main/pytorch_model.bin
```
% wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/resolve/main/pytorch_model.bin
```

They use the same tokenizer.model as the Llama2's. 

#### Convert

```sh
% python3 convert.py tinyllama-1.1b-3t <path>/TinyLlama/1.1B-3T <path>/Llama2/tokenizer.model tinyllama-1.1b-3t.lmw
% python3 convert.py tinyllama-1.1b-chat-v1.0 <path>TinyLlama/1.1B-Chat-v1.0 <path>/Llama2/tokenizer.model tinyllama-1.1b-chat-v1.0.lmw
```

#### Inference Example
```sh
% python llama-np.py -w <path>/tinyllama-1.1b-chat-v1.0.lmw -i "It is easy"
...
```



### TinyStories

#### Download the models from https://huggingface.co/karpathy/tinyllamas

```sh
% mkdir tinystories
% cd tinystoreis
% wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt
% wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.pt
% wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt
% wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt
% wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model
```

#### Convert
```
% python convert.py stories15m.lmw tinystories/stories15M.pt <path>/Llama2/tokenizer.model tinystories/stories15m.lmw
% python convert.py stories42m.lmw tinystories/stories42M.pt <path>/Llama2/tokenizer.model tinystories/stories42m.lmw
% python convert.py stories110m.lmw tinystories/stories110M.pt <path>/Llama2/tokenizer.model tinystories/stories110m.lmw
% python convert.py stories260k.lmw tinystories/stories260K.pt tinystories/tok512.model tinystories/stories260k.lmw
```

#### Inference Example


```
% python llama-np.py -w tinystories/stories15m.lmw -i "It is easy" 

It is easy for you to get up and play. But today you have to go to the doctor. He is very sick. He has a bad cough and a sore throat. He needs to take some medicine and rest.
Lily and Ben do not want to go to the doctor. They want to stay home and play. They say, "No, no, no! We are not sick! We are having fun!"
Mom says, "No, no, no! You have to go to the doctor. He will help you. He will make you feel better. He will give you some medicine

% python llama-np.py -w tinystories/stories260k.lmw -i "Once upon"

Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
```

## `compact-llama-np.py`: a simplified version  

The `compact-llama-np.py` script is a streamlined version of `llama-np.py`, reducing the code to under 250 lines (excluding tokenizer code and weight reading utilities). It maintains essential functions to illustrate the network architecture effectively. The Transformer part could be cut down to less than 50 lines of code. 

```
%  python compact-llama-np.py -w stories15m.lmw  -i 'There are three red balls and four green balls in the bag. If I take out' --seqlength 128
...
There are three red balls and four green balls in the bag. If I take out the red ball, I will be very happy. But I need to be careful. I don't want to get hurt."
The red balls were very excited. They wanted to play with the red ball. So, they started to roll and bounce. They were having so much fun.
But then, something unexpected happened. The red ball rolled out of the bag and into a big puddle. The red ball was sad. The red ball said, "I'm sorry, red ball. I didn't mean to
45.2931 tok/sec
```


