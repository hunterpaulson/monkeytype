
- needs to be fast, unusuable rn
    - benchmark
        - what is the min words that monkeytypes buffer will allow before showing text?
    - webgpu
    - context window
        - might be hard in js
    - faster algo?
        - dfa over tokenizer given wordset?


- discrete finite automaton
    - next token is going to contian more than just a single char


- kv cache?
- changing the context window makes us recompute prefill?
- or can we still use some of the kv cache?
- fixed size kv cache upper bound
    - WONT invalidate our kv cache or since we know the tokens I am thinking we can actually still use the kv cache we just have to drop the Ks and Vs for the first token. 
    - so at each decode step we have approx 0...k tokens where k is our context length. and then we drop the first so we have 1...k and we generate 1 more token. 
    - just some optimizations to think about while profiling since this makes it so we always know how much memory we will need for kv cache since we will always have an fixed small upper bound on context length. 


ux
- should use current active wordset (and constrain logits to it)
- can we "stream" words into the text input? or does it only allow batches of a certain size
    - that would make the ux so cool if you can see the words (or even better tokens) as they are generated
        - probably we can hack the word buffer that requests a next 100 words or so
- modes
    - support capitalization and punctuation



speed
- webgpu
    - apparently we are not using this
- everything parallel
- load the model, dfa, and more in parallel
- load the dfa while we download the model

bottlenecks
- model
    - download
        - how long does this take
    - load in memory
        - need to know exact size and which memory
        - how long does this take
- dfa
    - only needs to be computed once per wordset
- prefill
    - one token so this doesn't really exist
- decode (only decode)
- kv cache
    - we need to manage this ourselves since we are using a sliding fixed size context window
    - but this creates a sliding block of memory in memory
    - how much memory should we alloc. since we free beginning and append to end of memory block



draft pr
- deslop
- fix all lints
- make sure IGNORE lints are allowed or fix them
- test thoroughly
- make sure to follow all repo patterns and style conventions

blog
- show gif of tokens streaming in
- explain constrained decoding
- explain kv cache and context window
- benchmark with webgpu
    - metrics time to first token, inter-token latency, time to 100 words (monkeytype initial buffer size)
    - notice that we have optimize the inter-token latency so it is no longer our bottleneck
    - focus on model load and dfa build time as the main bottlenecks
    - parallelize
