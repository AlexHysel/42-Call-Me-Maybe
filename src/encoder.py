from pydantic import BaseModel
from typing import Any
from src.utils import special_to_standart, standart_to_special
import re


class Encoder(BaseModel):
    trie: dict[str, Any]
    vocab: list[str]

    def __init__(self, tokens: dict[str, int]):
        vocab: list[str | None] = [None] * len(tokens)
        trie: dict[str, Any] = {}
        print('Encoder: Building trie and vocab...')
        for word, token in tokens.items():
            vocab[token] = word
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node['token'] = token
        super().__init__(trie=trie, vocab=vocab)
        print('Encoder created.')

    def encode(self, text: str) -> list[int]:
        """Translates human text to a list of tokens for LLM."""

        text = standart_to_special(text)
        ids: list[int] = []
        i = 0
        while i < len(text):
            node = self.trie
            match_id = None
            match_len = -1
            j = i
            while j < len(text) and text[j] in node:
                node = node[text[j]]
                j += 1
                if 'token' in node:
                    match_id = node['token']
                    match_len = j - i
            if match_id is not None:
                ids.append(match_id)
                i += match_len
            else:
                i += 1
        return ids

    def encode_words(self, text: str) -> set[int]:
        """Returns all possible tokens from the string"""

        ids = set()
        words = text.split()
        for word in words:
            word = word.strip('.,!?')
            word = word.strip('"\'')
            if not word:
                continue
            for token_id in self.encode(word):
                ids.add(token_id)
            ids.add(self.encode(' ' + word)[0])
        return ids

    def encode_words_separated(self, text: str) -> list[list[int]]:
        """Returns tokenized prompt fragments."""
        ids: list[list[int]] = []

        colon_match = re.search(r':\s*(.+)$', text)
        if colon_match:
            full_value = colon_match.group(1).strip()
            ids.append(self.encode(full_value))

        pattern = r'''
            "(?:\\.|[^"])*"   |
            '(?:\\.|[^'])*'   |
            \S+
        '''
        unescaped = text.replace('\\"', '"')
        parts = re.findall(pattern, unescaped, re.VERBOSE)
        for part in parts:
            part = part.strip('".,!?:;\\')
            part = part.strip("'")
            if not part:
                continue
            ids.append(self.encode(part))

        return ids

    def encode_all(self, text: str) -> set[int]:
        """Returns all possible tokens from the string"""

        c_set = set(text)
        ids = set()
        for token_id in range(len(self.vocab)):
            if self.vocab[token_id]:
                if all(c in c_set for c in self.vocab[token_id]):
                    ids.add(token_id)
        return ids

    def decode(self, tokens: list[int] | int) -> str:
        """Translates LLM tokens to human-readable text"""

        if isinstance(tokens, int):
            return self.vocab[tokens]
        result = ""
        for token in tokens:
            result += self.vocab[token]
        return special_to_standart(result)
