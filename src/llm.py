import numpy as np
from pydantic import BaseModel
from src.llm_sdk import Small_LLM_Model

from src.encoder import Encoder


class LLM(BaseModel):
    _llm: Small_LLM_Model = Small_LLM_Model()
    _encoder: Encoder
    _t_instruction: list[int] | None = None

    def __init__(self, encoder: Encoder):
        self._encoder = encoder     

    def next_token(self, tokens: list[int], mask: set[int] = None) -> int:
        """Returns the next token for the provided tokens."""
        logits = self._get_logits(tokens, mask)
        
        #for token_id in set(generated_arg_tokens):
        #    logits[token_id] -= 5.0
        #best_logit = max(logits)# - i
        #probs = np.exp(logits - np.max(logits))
        #probs /= probs.sum()
        #if np.max(probs) < 0.4:
        #    break
        
        best_token = int(np.argmax(logits))

        print(f'{self.decode([best_token])}') #- {best_token:.1f}')
        return best_token

    def set_instruction(self, new: list[int] | str) -> None:
        """Set the instruction with information for LLM."""

        if new is str:
            new = self._encoder.encode(new)
        self._t_instruction = new

    def _get_logits(self, tokens: list[int], mask = None) -> list[int]:
        """
        Returns the list of logits for provided tokens. 
        Applies the mask optionally.
        """
        l = self.llm.get_logits_from_input_ids(self._t_instruction + tokens)
        if mask is not None:
            l = self._apply_mask(mask, l)
        return l

    def _apply_mask(self, allowed_ids: list[int], logits) -> list[int]:
        """
        Returns logits with mask applied by setting all forbidden 
        token scores to -infinity.
        """
        masked = np.full_like(logits, -float('inf'))
        for id in allowed_ids:
            masked[id] = logits[id]
        return masked
