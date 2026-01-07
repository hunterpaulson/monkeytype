# problem

I think monkey type uniformly samples words from the selected dataset. so `the` and the 200th most common word have the same probability of showing up in the sequence you type
- note this is not quite true since they prevent the any word from following itself

but in reality words follow zipfs law

in order to actually good at typing you need to practice common sequences of words. not just individual words. but how can we do this without quote mode

answer is really smol base llm

# idea

use the smallest and fastest base llm I can to inference super fast on monkeytype to generate the words that you type. this way sequence of words have high probablility. even if they make no sense. 

since nobody cares if the sequences make sense.

# challenges

tokens != words

do we still need to constrain to selected dataset (like constrain logprobs) or can we just let it generate 

how can we prevent it from going off the rails

multilingual

going off the rail, not enough entropy
wonder if this can be solved with a small like O(10) token context window. 
like context window is always like 5 words

# why this is a good project
very [[middle-out learning|middle-out]]
very [[build the ramp]] and interactable

would learn
- contributing to open source
- high performance local inference
- using base models
- learn about sampling
- how to build projects for users with latency requirements
- multi lingual products

