# `llama-np.py`

This is to play with models based on Meta's llama2 and llama3. 

The script `llama-np.py` implements inference using only `numpy` and `cupy`, due to their lack of support for lower precision and bf16 formats, thus running models in fp32. It can execute the llama-3-8B model at approximately 1.2 tokens per second (tok/s) on CPU and 2.24 tok/s on GPU using a modern PC configuration (AMD Ryzen 7700x CPU with over 32GB RAM and an RTX 4070 GPU).

llama-np.py offers two operational modes:

1. Text completion (default) for prompt completions.
2. Chat (activated with the `--chat` option) for engaging in dialogues with the model.

It supports various experimental features:

- Enabling or disabling the key-value (kv) cache
- Methods for updating the kv cache: in-place or concatenation
- Prefill strategies for input tokens: single sequence or one token at a time, similar to the approach in [`llama2.c`](https://github.com/karpathy/llama2.c)).
- Option to use cupy instead of numpy

```sh
% python llama-np.py -w ~/Meta-Llama-3-8B.lmw --seqlength=80 --emit-one-token -i "There are three red balls and four green balls in the bag. If I take out" 

<|begin_of_text|>There are three red balls and four green balls in the bag. If I take out a ball at random, what is the probability that it is green?

There are a total of seven balls.  Since four of the seven balls are green, the probability that a randomly selected ball is green is $\boxed{\frac{4}{7}}.$
Final Answer: The final answer is \frac{4}{7}. I hope it is correct.
[1.3231 tok/s]
...
```


## Supported Models

| Model Name | Size | Precision |
| ---------- | ---- | ---- |
| llama3     | 8B   | bf16     |
| llama2     | 7B   | bf16     |
| TinyStories  | 260K   | fp32     |
| TinyStories  | 15M   | fp32     |
| TinyStories  | 42M   | fp32     |
| TinyStories  | 110M   | fp32     |
| TinyLlama    | 1.1B   | bf16     |

## Operating Instructions
Running a model involves two primary steps: converting the weights into a compatible format and running the model itself.

### Weight Conversion
Weights need to be converted into the `.lmw` (lumi weight) format, embedding the tokenizer within the weight file. The `convert.py` script is responsible for this transformation and contains model-specific logic.

### Model Execution
Once weights are converted, the `llama-np.py` script performs text generation using the input string and the converted weight file.

### Running Models
Given a weight file (`*.lmw`) and an initial text string, `llama-np.py` performs text generation based on the input string.


### Llama-3

#### Download the models
Get the model files from Meta per instructions on https://github.com/meta-llama/llama3/blob/main/README.md

#### Convert

```sh
% python convert.py <path>/Meta-Llama-3-8B <path>/Meta-Llama-3-8B/tokenizer.model <path>/Meta-Llama-3-8B/llama-3-8b.lmw 
```

#### Inference
```sh
% python llama-np.py -w <path>/Meta-Llama-3-8B/llama-3-8b.lmw -i "It is easy"

<|begin_of_text|>It is easy to get caught up in the excitement of the holiday season. The decorations, the ...
```

### Llama-2

#### Download the models
Get the model files from Meta per instructions on https://github.com/meta-llama/llama/blob/main/README.md

#### Convert

```sh
% python convert.py <path>/Llama2/llama-2-7b <path>/Llama2/tokenizer.model <path>/Llama2/llama-2-7b/llama-2-7b.lmw 
```

#### Inference
```sh
% python llama-np.py -w <path>/Llama2/llama-2-7b/llama-2-7b.lmw -i "It is easy"

It is easy to get lost in the world of the internet. It is easy ...
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
% python convert.py tinystories/stories15M.pt <path>/Llama2/tokenizer.model tinystories/stories15M.lmw
% python convert.py tinystories/stories42M.pt <path>/Llama2/tokenizer.model tinystories/stories42M.lmw
% python convert.py tinystories/stories110M.pt <path>/Llama2/tokenizer.model tinystories/stories110M.lmw
% python convert.py tinystories/stories260K.pt tinystories/tok512.model tinystories/stories260K.lmw
```

#### Inference


```
% python llama-np.py -w tinystories/stories15M.lmw -i "It is easy" 

It is easy for you to get up and play. But today you have to go to the doctor. He is very sick. He has a bad cough and a sore throat. He needs to take some medicine and rest.
Lily and Ben do not want to go to the doctor. They want to stay home and play. They say, "No, no, no! We are not sick! We are having fun!"
Mom says, "No, no, no! You have to go to the doctor. He will help you. He will make you feel better. He will give you some medicine

% python llama-np.py -w tinystories/stories260K.lmw -i "Once upon"

Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
```

## `compat-llama-np.py`: a simplified version  

The `compat-llama-np.py` script is a streamlined version of `llama-np.py`, reducing the code to under 250 lines (excluding tokenizer code and weight reading utilities). It maintains essential functions to illustrate the network architecture effectively.

```
%  python compat-llama-np.py -w stories15M.lmw  -i 'There are three red balls and four green balls in the bag. If I take out' --seqlength 128
...
There are three red balls and four green balls in the bag. If I take out the red ball, I will be very happy. But I need to be careful. I don't want to get hurt."
The red balls were very excited. They wanted to play with the red ball. So, they started to roll and bounce. They were having so much fun.
But then, something unexpected happened. The red ball rolled out of the bag and into a big puddle. The red ball was sad. The red ball said, "I'm sorry, red ball. I didn't mean to
45.2931 tok/sec
```


