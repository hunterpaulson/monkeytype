
# Problem

## Current Sampling Architecture

### 1. Default Mode: Uniform Random Sampling

The core sampling happens in frontend/src/ts/test/wordset.ts:

```ts
  randomWord(mode: FunboxWordsFrequency): string {
      if (mode === "zipf") {
        return this.words[zipfyRandomArrayIndex(this.words.length)] as string;
      } else {
        return randomElementFromArray(this.words);
      }
    }
```
By default (`mode === "normal"`), it uses `randomElementFromArray()` which does uniform random sampling — every word in the list has equal probability regardless of position.

### 2. Words ARE Ordered by Frequency

Interestingly, the language files like english.json have "orderedByFrequency": true and list words from most to least common:
wordset.tsLines 5-15
```json
  "words": [
        "the",      // #1 most common
        "be",       // #2
        "of",       // #3
        "and",      // #4
```

Your intuition is correct — despite words being ordered by frequency, MonkeyType samples them uniformly. So "the" and word #200 have equal probability of appearing!

### 3. Zipf Mode Exists (as a "Funbox")

There IS a Zipf sampling option, but it's hidden as a "funbox" (novelty mode):

```ts
export function zipfyRandomArrayIndex(dictLength: number): number {
  /**
   * get random index based on probability distribution of Zipf's law,
   * where PMF is (1/n)/H_N,
   * where H_N is the Harmonic number of (N), where N is dictLength
   * and the harmonic number is approximated using the formula:
   * H_n = ln(n + 0.5) + gamma
   */
  const gamma = 0.5772156649015329; // Euler–Mascheroni constant
  const H_N = Math.log(dictLength + 0.5) + gamma; // approximation of H_N
  const r = Math.random();
  /* inverse of CDF where CDF is H_n/H_N */
  const inverseCDF = Math.exp(r * H_N - gamma) - 0.5;
  return Math.floor(inverseCDF);
}
```

This correctly implements Zipf's law sampling using the inverse CDF method.

### 4. Anti-Repeat Logic
The word generator (words-generator.ts) prevents the same word from appearing twice in a row (or 3 times in the last 3 words):
words-generator.tsLines 852-874
```ts
      while (
        regenarationCount < 100 &&
        (previousWordRaw === firstAfterSplitLazy ||
          previousWord2Raw === firstAfterSplitLazy ||
```