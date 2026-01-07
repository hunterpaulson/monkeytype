# constrained decoding

### goal

given a wordset, generate a sequence containing only words from the wordset

## requirements

### input

wordset: a set of words that, along with a space and maybe puncuation between them are allowed to be generated.
- note a word can be in _any_ language or _any_ case. this algo should be general enough to essentially handle any set of arbitrary strings.


NOTE: 
there is NO prompt. initial sample just starts with a beginning of sequence <|bos|> token or some previously generated token depending on context window size.
we are using a base model: distilgpt2 


## example

wordset: \[ "help", "hello", "world", "word" ]

initial input to llm is just vector with <|bos|> token
the only tokens that are allowed to be generated are the words or subword prefixes of the words in the wordset

so for example \[" h", " he", " hel", " help", " hello", " w", " wor"," world", " word"]
- since we are using gpt2 as our model spaces usually come before the word
- lets build using this assumption

say we sample "hel" making our sequence "<|bos|> hel"
- note this has a leading space. this is acceptible since I think it makes the whole process easier

then the only tokens that are allowed to be generated must be ones that are make the input sequence a valid sequence of words from the wordset.

so in this case "lo" or "p".
lets say we sample "lo" making our sequence "<|bos|> hello"

yay our first word. we just continue the process. since any space + word in our wordset would be valid next we should have a similar set of valid tokens again

\[" h", " he", " hel", " help", " hello", " w", " wor"," world", " word" ]

from which we sample " wor"

so we have "<|bos|> hello wor"

here our only valid tokens are " d" and " ld"
from which we sample "ld" making our sequence "<|bos|> hello world"

you get the picture.

## related problems

[character prefix conditioning](https://cursor.com/blog/cpc)

however their problem is slightly different because we are not taking input from a user that may fall on token boundaries.
we _always_ start with just a <|bos|> token. and only sample tokens at each step.
