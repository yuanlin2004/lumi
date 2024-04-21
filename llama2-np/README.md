# llama2-np.py

This is to play with Meta's llama2 model. 

Currently it implements an inference flow using `numpy` only, except for the tokenizer which uses [sentencepiece](https://github.com/google/sentencepiece).


- fp32
- tested with llama2-7B model and TinyStories models only
- no GQA yet (not used by the 7B model), only MHA.
- a simple sampler that greedily picks the token with the highest score
- ~35 sec/token (Ryzen 7 7700x, 32GB RAM)

`llama2-np.py` implements different options for experimental purposes.  

- use kv-cache or not 
- feed the input tokens, in the prefill stage, to the transform block as one sequence or one token at a time (similar to that in [`llama2.c`](https://github.com/karpathy/llama2.c)). 

The correctness is verfied by comparing the generated text and values of some intermediate tensors in `llama2-np.py` with those from `llama2.c`, both using the llama2-7B model. 


```sh
% python llama2-np.py -t ~/tokenizer.model -w ~/w.lmw -i "There are three red balls and four green balls in the bag. If I take out" 

There are three red balls and four green balls in the bag. If I take out one ball at random, what is the probability that it is a red ball?
The probability of getting a red ball is 3/7.
...
```

## Llama2 under 250 LoC: `compat-llama2-np.py`

`compat-llama2-np.py` is a variant of `llama2-np.py` under 250, exclusing the tokenizer code and the weight reading code. It strips out many experimental features and debugging utilites. 

Since it does not have the boilerplate code, `compat-llama2-np.py` is somewhat more readable than `llama2-np.py` in terms of illustrating the llama2 network architecture. 

```
%  python compat-llama2-np.py -t tokenizer.model -w stories15M.lmw  -i 'There are three red balls and four green balls in the bag. If I take out' --seqlength 128
...
There are three red balls and four green balls in the bag. If I take out the red ball, I will be very happy. But I need to be careful. I don't want to get hurt."
The red balls were very excited. They wanted to play with the red ball. So, they started to roll and bounce. They were having so much fun.
But then, something unexpected happened. The red ball rolled out of the bag and into a big puddle. The red ball was sad. The red ball said, "I'm sorry, red ball. I didn't mean to
45.2931 tok/sec
```


