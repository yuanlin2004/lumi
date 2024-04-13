# llama2-np.py

This is to play with Meta's llama2 model. 

Currently it implements an inference flow using `numpy` only, except for the tokenizer which uses [sentencepiece](https://github.com/google/sentencepiece).


- fp32
- tested with 7B model only
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
The probability of getting a red ball is 3/7.
...
```


