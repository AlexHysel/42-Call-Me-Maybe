from pydantic import BaseModel
from src.utils import special_to_standart, standart_to_special, escape


class Encoder(BaseModel):
    token_of: dict[str, int]
    vocab: list[str]

    def __init__(self, tokens: dict[str, int]):
        vocab = [None] * len(tokens)
        for word, token in tokens.items():
            vocab[token] = word

        super().__init__(token_of = tokens, vocab = vocab)

    def encode(self, text: str) -> list[int]:
        """Translates human text to a list of tokens for LLM"""

        text = standart_to_special(text)
        ids = []
        while text:
            match_id = None
            match_len = -2
            for i in range(0, len(text) + 1):
                substr = text[:i]
                if substr in self.token_of:
                    match_id = self.token_of[substr]
                    match_len = len(substr)
            if match_id is not None:
                ids.append(match_id)
                text = text[match_len:]
            else:
                text = text[1:]

        return ids
    
    def encode_words(self, text: str) -> set[int]:
        """Returns all possible tokens from the string"""

        ids = set()
        words = text.split()
        for word in words:
            word = word.strip("'\".,!?")
            if not word:
                continue
            for token_id in self.encode(word):
                ids.add(token_id)
            for token_id in self.encode(' ' + word):
                ids.add(token_id)
        return ids

    def encode_all(self, text: str) -> set[int]:
        """Returns all possible tokens from the string"""

        char_set = set(text)
        ids = set()
        for token_id in range(len(self.vocab)):
            if self.vocab[token_id] and all(c in char_set for c in self.vocab[token_id]):
                ids.add(token_id)
        return ids

    def decode(self, tokens: list[int] | int) -> str:
        """Translates LLM tokens to human-readable text"""
        
        if isinstance(tokens, int):
            return self.vocab[token]
        result = ""
        for token in tokens:
            result += self.vocab[token]
        return special_to_standart(result)