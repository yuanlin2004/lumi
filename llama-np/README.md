# llama-np.py

This is to play with models based on Meta's llama2 and llama3.

Currently it implements an inference flow from scratch using `numpy` only, except for the tokenizer.

- fp32
- a simple sampler that greedily picks the token with the highest score

`llama-np.py` implements different options for experimental purposes.  

- use kv-cache or not 
- update kv-cache using an in-place-update method or a concatenate method
- feed the input tokens, in the prefill stage, to the transform block as one sequence or one token at a time (similar to that in [`llama2.c`](https://github.com/karpathy/llama2.c)). 


```sh
% python llama-np.py -t ~/tokenizer.model -w ~/model.lmw -i "There are three red balls and four green balls in the bag. If I take out" 

There are three red balls and four green balls in the bag. If I take out one ball at random, what is the probability that it is a red ball?
The probability of getting a red ball is 3/7.
...
```

## Models Supported

| Model Name | Size | Precision |
| ---------- | ---- | ---- |
| llama3     | 8B   | fp32     |
| llama2     | 7B   | fp32     |
| TinyStories  | 260K   | fp32     |
| TinyStories  | 15M   | fp32     |
| TinyStories  | 42M   | fp32     |
| TinyStories  | 110M   | fp32     |


## Run the Models

There are two steps run an existing model - convert the weights, run the model. 

### Weight Conversion
Model weights need to be converted into the `lumi weight` format (.lmw) before being fed into `llama-np.py`. The `convert.py` script does the magic. It can read pytorch checkpoint (.pt) files.  Handling of Huggingface `.safetensors` files and `pytorch_model.bin` files is also implemented but inference results are wrong for some unknown reasons right now.

Due to the specifics of individual models, `convert.py` contains many hard-coded logics.

### Running Models
Given a tokenizer model (usually `tokenizer.model`), a weight file (`*.lmw`) and an initial text string, `llama-np.py` performs text generation based on the input string. 

`compat-llama-np.py` is a simplified version of `llama-np.py`, aiming to be as short as possible, but not too short. See below.

### Llama-3

#### Download the models
Get the model files from Meta per instructions on https://github.com/meta-llama/llama3/blob/main/README.md

#### Convert

```sh
% python convert.py <path>/Meta-Llama-3-8B <path>/Meta-Llama-3-8B/llama-3-8b.lmw 
```

#### Inference
```sh
% python llama-np.py -t <path>/Meta-Llama-3-8B/tokenizer.model -w <path>/Meta-Llama-3-8B/llama-3-8b.lmw -i "It is easy"

<|begin_of_text|>It is easy to get caught up in the excitement of the holiday season. The decorations, the ...
```

### Llama-2

#### Download the models
Get the model files from Meta per instructions on https://github.com/meta-llama/llama/blob/main/README.md

#### Convert

```sh
% python convert.py <path>/Llama2/llama-2-7b <path>/Llama2/llama-2-7b/llama-2-7b.lmw 
```

#### Inference
```sh
% python llama-np.py -t <path>/Llama2/tokenizer.model -w <path>/Llama2/llama-2-7b/llama-2-7b.lmw -i "It is easy"

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
% python convert.py tinystories/stories15M.pt tinystories/stories15M.lmw
% python convert.py tinystories/stories42M.pt tinystories/stories42M.lmw
% python convert.py tinystories/stories110M.pt tinystories/stories110M.lmw
% python convert.py tinystories/stories260K.pt tinystories/stories260K.lmw
```

#### Inference

All models uses the same `tokenizer.model` from `llama-2` above, except for `stories260K` which uses the `tok512.model`.

```
% python llama-np.py -t tokenizer.model -w tinystories/stories15M.lmw -i "It is easy" 

It is easy for you to get up and play. But today you have to go to the doctor. He is very sick. He has a bad cough and a sore throat. He needs to take some medicine and rest.
Lily and Ben do not want to go to the doctor. They want to stay home and play. They say, "No, no, no! We are not sick! We are having fun!"
Mom says, "No, no, no! You have to go to the doctor. He will help you. He will make you feel better. He will give you some medicine

% python llama-np.py --seqlength=257 -t tinystories/tok512.model -w tinystories/stories260K.lmw -i "Once upon"

Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
```

## `compat-llama-np.py`: Llama2/Llama3 under 250 LoC

`compat-llama-np.py` is a variant of `llama-np.py` under 250 LoC, excluding the tokenizer code and the weight reading code. It strips out many experimental features and debugging utilities. 

Since it does not have the boilerplate code, `compat-llama-np.py` is somewhat more readable than `llama-np.py` in terms of illustrating the llama network architecture. 

```
%  python compat-llama-np.py -t tokenizer.model -w stories15M.lmw  -i 'There are three red balls and four green balls in the bag. If I take out' --seqlength 128
...
There are three red balls and four green balls in the bag. If I take out the red ball, I will be very happy. But I need to be careful. I don't want to get hurt."
The red balls were very excited. They wanted to play with the red ball. So, they started to roll and bounce. They were having so much fun.
But then, something unexpected happened. The red ball rolled out of the bag and into a big puddle. The red ball was sad. The red ball said, "I'm sorry, red ball. I didn't mean to
45.2931 tok/sec
```


