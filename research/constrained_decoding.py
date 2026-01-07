"""
Constrained Decoding for Wordset-Limited Text Generation.

Uses a trie-based approach to constrain LLM generation to only produce
words from a predefined wordset. Useful for applications like typing
practice where you want realistic word sequences but limited vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    set_seed,
)


@dataclass
class TrieNode:
    """A node in the trie representing a character position."""

    children: dict[str, TrieNode] = field(default_factory=dict)
    is_word_end: bool = False


class Trie:
    """
    Prefix tree for efficient word/prefix lookup.
    
    Used to quickly determine if a string is a valid prefix or complete word
    from the wordset.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word into the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word_end = True

    def _traverse(self, prefix: str) -> Optional[TrieNode]:
        """Traverse to the node representing prefix, or None if not found."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def is_prefix(self, prefix: str) -> bool:
        """Check if prefix exists in trie (as prefix or complete word)."""
        if not prefix:
            return False
        return self._traverse(prefix) is not None

    def is_word(self, word: str) -> bool:
        """Check if word is a complete word in the trie."""
        if not word:
            return False
        node = self._traverse(word)
        return node is not None and node.is_word_end


class WordsetConstrainedLogitsProcessor(LogitsProcessor):
    """
    HuggingFace LogitsProcessor that constrains generation to wordset.
    
    At each generation step, masks out tokens that would produce invalid
    sequences (words not in the wordset).
    """

    def __init__(
        self,
        wordset: list[str],
        tokenizer: Optional[AutoTokenizer] = None,
        model_name: str = "distilgpt2",
    ) -> None:
        self.wordset = wordset
        self.model_name = model_name
        
        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        
        # Build trie with ONLY space-prefixed words
        # This simplifies logic: ALL words require leading space, including first word
        # Generated text will be: " word1 word2 word3..." (with leading space)
        self.trie = Trie()
        for word in wordset:
            self.trie.insert(" " + word)
        
        # Precompute token ID to decoded string mapping
        self.token_to_str: dict[int, str] = {}
        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            # Decode the token to get actual characters
            decoded = self.tokenizer.decode([token_id])
            self.token_to_str[token_id] = decoded
        
        # Track generation state
        self.current_prefix = ""
        self.generated_text = ""

    def reset(self) -> None:
        """Reset state for new generation."""
        self.current_prefix = ""
        self.generated_text = ""

    def get_valid_token_ids(self, current_prefix: str) -> set[int]:
        """
        Compute which token IDs are valid given the current prefix.
        
        A token is valid if appending it to current_prefix results in:
        1. A valid prefix of a word in the wordset, OR
        2. A complete word followed by a valid prefix of another word
        """
        valid_tokens = set()
        
        for token_id, token_str in self.token_to_str.items():
            if self._is_valid_continuation(current_prefix, token_str):
                valid_tokens.add(token_id)
        
        return valid_tokens

    def _is_valid_continuation(self, prefix: str, token_str: str) -> bool:
        """
        Check if appending token_str to prefix yields a valid state.
        
        Handles the complexity of tokens that might complete a word and
        start a new one (e.g., "ld " completing "world" and starting next).
        """
        candidate = prefix + token_str
        
        # Case 1: Simple prefix continuation
        if self.trie.is_prefix(candidate):
            return True
        
        # Case 2: The candidate completes one or more words
        # We need to find if there's a valid way to parse the candidate
        return self._can_parse_as_valid_sequence(candidate)

    def _can_parse_as_valid_sequence(self, text: str) -> bool:
        """
        Check if text can be parsed as a valid sequence of words + optional prefix.
        
        With space-prefixed-only design, all words are " word" format.
        Valid sequences: "", " hello", " hello world", " hel" (partial)
        """
        # Empty text is valid (ready for next " word")
        if not text:
            return True
        
        # If text itself is a valid prefix, it's valid
        if self.trie.is_prefix(text):
            return True
        
        n = len(text)
        
        # Try to find complete words followed by valid prefix
        for i in range(1, n + 1):
            potential_word = text[:i]
            remainder = text[i:]
            
            # Check if potential_word is a complete word (all words are " word" format)
            if self.trie.is_word(potential_word):
                # If no remainder, we completed a word - valid!
                if not remainder:
                    return True
                
                # If remainder is a valid prefix, valid!
                if self.trie.is_prefix(remainder):
                    return True
                
                # Try to recursively parse remainder
                if self._can_parse_as_valid_sequence(remainder):
                    return True
        
        return False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Apply constraints to logits at each generation step.
        
        Called by HuggingFace generate() at each token generation.
        """
        # Get the tokens generated so far (excluding BOS)
        # and update our state
        if input_ids.shape[1] > 1:
            # Decode all generated tokens to get current text state
            generated_ids = input_ids[0, 1:].tolist()  # Skip BOS
            self.generated_text = self.tokenizer.decode(generated_ids)
            
            # Find the current partial word (text after last complete word)
            self.current_prefix = self._get_current_prefix(self.generated_text)
        else:
            # Just BOS token, starting fresh
            self.current_prefix = ""
            self.generated_text = ""
        
        # Compute valid tokens for this step
        valid_token_ids = self.get_valid_token_ids(self.current_prefix)
        
        # If no valid tokens, something went wrong - allow EOS to terminate
        if not valid_token_ids:
            valid_token_ids = {self.tokenizer.eos_token_id}
        
        # Create mask and apply to scores
        mask = torch.full_like(scores, float("-inf"))
        for token_id in valid_token_ids:
            mask[0, token_id] = 0
        
        return scores + mask

    def _get_current_prefix(self, text: str) -> str:
        """
        Extract the current partial word being built.
        
        With space-prefixed-only design:
        - All words are " word" (with leading space)
        - Generated text is " word1 word2 word3..."
        - After completing " word", we're back to "" (ready for next " word")
        - While building " hel", prefix is " hel"
        
        Key insight: If matching a word leaves a remainder without a leading space,
        that's invalid - we should keep the whole thing as a prefix instead.
        E.g., " an" should NOT be split into " a" + "n", because "n" is invalid.
        """
        if not text:
            return ""  # Ready to start " word1"
        
        # Find where the last complete word ends
        n = len(text)
        last_valid_word_end = 0
        
        # Dynamic parsing to find word boundaries
        i = 0
        while i < n:
            # Try to match a word starting at position i
            # We want the LONGEST match that leaves a valid remainder
            best_match_end = -1
            
            for j in range(i + 1, n + 1):
                candidate = text[i:j]
                if self.trie.is_word(candidate):
                    remainder = text[j:]
                    # Only accept this match if remainder is valid:
                    # - Empty (we completed the text)
                    # - Starts with space (can be start of next word)
                    # - Is a valid prefix in trie
                    if not remainder or remainder.startswith(" ") or self.trie.is_prefix(remainder):
                        best_match_end = j
            
            if best_match_end > 0:
                last_valid_word_end = best_match_end
                i = best_match_end
            else:
                # No valid complete word found from here - rest is prefix
                break
        
        # Return whatever comes after the last complete word
        # If empty, we're ready for the next " word"
        return text[last_valid_word_end:]


def generate_constrained_text(
    wordset: list[str],
    num_words: int = 10,
    seed: Optional[int] = None,
    model_name: str = "distilgpt2",
    max_new_tokens: int = 100,
    context_window: Optional[int] = None,
    temperature: float = 0.8,
) -> str:
    """
    Generate text constrained to only use words from wordset.
    
    Args:
        wordset: List of allowed words
        num_words: Target number of words to generate (approximate)
        seed: Random seed for reproducibility
        model_name: HuggingFace model to use
        max_new_tokens: Maximum tokens to generate
        context_window: Number of recent tokens to use as context (None = unlimited).
                        Small values (e.g., 10) produce more varied, less coherent text.
                        Good for typing practice where you want word variety.
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated text string containing only words from wordset
    """
    if seed is not None:
        set_seed(seed)
        torch.manual_seed(seed)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # NOTE: use_safetensors=False avoids memory-mapping crash on some macOS/PyTorch combos
    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=False)
    model.eval()
    
    # Create our constrained logits processor
    constraint_processor = WordsetConstrainedLogitsProcessor(
        wordset=wordset,
        tokenizer=tokenizer,
        model_name=model_name,
    )
    
    # Custom generation loop to support context window truncation
    generated_ids = [tokenizer.bos_token_id]
    
    for _ in range(max_new_tokens):
        # Truncate to context window if specified
        if context_window is not None and len(generated_ids) > context_window:
            # Keep BOS + last (context_window - 1) tokens
            context_ids = [tokenizer.bos_token_id] + generated_ids[-(context_window - 1):]
        else:
            context_ids = generated_ids
        
        input_ids = torch.tensor([context_ids])
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last token's logits
        
        # Get current generated text (excluding BOS) for constraint checking
        generated_text = tokenizer.decode(generated_ids[1:]) if len(generated_ids) > 1 else ""
        current_prefix = constraint_processor._get_current_prefix(generated_text)
        
        # Get valid tokens
        valid_token_ids = constraint_processor.get_valid_token_ids(current_prefix)
        
        # If no valid tokens, stop (shouldn't happen with good wordset)
        if not valid_token_ids:
            print(f"\n[DEBUG] No valid tokens found!")
            print(f"  Generated text: '{current_text}'")
            print(f"  Current prefix: '{current_prefix}'")
            print(f"  Last few token IDs: {generated_ids[-5:]}")
            print(f"  Last few tokens decoded: {[tokenizer.decode([t]) for t in generated_ids[-5:]]}")
            break
        
        # Mask invalid tokens
        mask = torch.full_like(logits, float("-inf"))
        for tid in valid_token_ids:
            mask[tid] = 0
        masked_logits = logits + mask
        
        # Sample
        probs = torch.softmax(masked_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        generated_ids.append(next_token)
        
        # Check if we have enough words AND we're at a word boundary
        # (not in the middle of a partial word)
        current_text = tokenizer.decode(generated_ids[1:])
        current_prefix = constraint_processor._get_current_prefix(current_text)
        at_word_boundary = (current_prefix == "")
        
        if len(current_text.strip().split()) >= num_words and at_word_boundary:
            break
    
    # Decode and remove any trailing partial word (so we only return full words)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    trailing_prefix = constraint_processor._get_current_prefix(generated_text)
    if trailing_prefix:
        generated_text = generated_text[: -len(trailing_prefix)]
    
    # Trim to approximately num_words
    words = generated_text.strip().split()
    if len(words) > num_words:
        words = words[:num_words]
    
    return " ".join(words)


def main():
    """Demo the constrained decoding."""
    wordset = [
        "the",
        "be",
        "of",
        "and",
        "a",
        "to",
        "in",
        "he",
        "have",
        "it",
        "that",
        "for",
        "they",
        "I",
        "with",
        "as",
        "not",
        "on",
        "she",
        "at",
        "by",
        "this",
        "we",
        "you",
        "do",
        "but",
        "from",
        "or",
        "which",
        "one",
        "would",
        "all",
        "will",
        "there",
        "say",
        "who",
        "make",
        "when",
        "can",
        "more",
        "if",
        "no",
        "man",
        "out",
        "other",
        "so",
        "what",
        "time",
        "up",
        "go",
        "about",
        "than",
        "into",
        "could",
        "state",
        "only",
        "new",
        "year",
        "some",
        "take",
        "come",
        "these",
        "know",
        "see",
        "use",
        "get",
        "like",
        "then",
        "first",
        "any",
        "work",
        "now",
        "may",
        "such",
        "give",
        "over",
        "think",
        "most",
        "even",
        "find",
        "day",
        "also",
        "after",
        "way",
        "many",
        "must",
        "look",
        "before",
        "great",
        "back",
        "through",
        "long",
        "where",
        "much",
        "should",
        "well",
        "people",
        "down",
        "own",
        "just",
        "because",
        "good",
        "each",
        "those",
        "feel",
        "seem",
        "how",
        "high",
        "too",
        "place",
        "little",
        "world",
        "very",
        "still",
        "nation",
        "hand",
        "old",
        "life",
        "tell",
        "write",
        "become",
        "here",
        "show",
        "house",
        "both",
        "between",
        "need",
        "mean",
        "call",
        "develop",
        "under",
        "last",
        "right",
        "move",
        "thing",
        "general",
        "school",
        "never",
        "same",
        "another",
        "begin",
        "while",
        "number",
        "part",
        "turn",
        "real",
        "leave",
        "might",
        "want",
        "point",
        "form",
        "off",
        "child",
        "few",
        "small",
        "since",
        "against",
        "ask",
        "late",
        "home",
        "interest",
        "large",
        "person",
        "end",
        "open",
        "public",
        "follow",
        "during",
        "present",
        "without",
        "again",
        "hold",
        "govern",
        "around",
        "possible",
        "head",
        "consider",
        "word",
        "program",
        "problem",
        "however",
        "lead",
        "system",
        "set",
        "order",
        "eye",
        "plan",
        "run",
        "keep",
        "face",
        "fact",
        "group",
        "play",
        "stand",
        "increase",
        "early",
        "course",
        "change",
        "help",
        "line"
    ]
    
    print("=" * 60)
    print("Constrained Decoding Demo")
    print("=" * 60)
    print(f"\nWordset ({len(wordset)} words): {wordset[:10]}...")
    
    seed = None
    num_words = 100

    # Demo 1: Unlimited context (more coherent sentences)
    print("\n--- Unlimited context (coherent) ---")
    result1 = generate_constrained_text(
        wordset=wordset,
        num_words=num_words,
        seed=seed,
        context_window=None,  # Unlimited
    )
    print(f"Generated: {result1}")
    
    # Demo 2: Small context window (more varied, less coherent)
    print("\n--- Context window = 10 tokens (varied) ---")
    result2 = generate_constrained_text(
        wordset=wordset,
        num_words=num_words,
        seed=seed,
        context_window=10,
    )
    print(f"Generated: {result2}")
    
    # Demo 3: Very small context (very random)
    print("\n--- Context window = 5 tokens (very random) ---")
    result3 = generate_constrained_text(
        wordset=wordset,
        num_words=num_words,
        seed=seed,
        context_window=5,
    )
    print(f"Generated: {result3}")
    
    # Verify all words are in wordset
    for name, result in [("Unlimited", result1), ("ctx=10", result2), ("ctx=5", result3)]:
        words = result.strip().split()
        valid = all(w.lower() in [ws.lower() for ws in wordset] for w in words)
        print(f"\n{name}: All words in wordset: {valid}")
        if not valid:
            invalid_words = [w for w in words if w.lower() not in [ws.lower() for ws in wordset]]
            print(f"  Invalid words: {invalid_words}")


if __name__ == "__main__":
    main()
