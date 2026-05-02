import numpy as np
from pydantic import BaseModel, PrivateAttr
from src.llm_sdk import Small_LLM_Model

from src.encoder import Encoder


class LLM(BaseModel):
    _llm: Small_LLM_Model = PrivateAttr()
    _encoder: Encoder = PrivateAttr()
    _t_instruction: list[int] | None = None

    def __init__(self, llm: Small_LLM_Model, encoder: Encoder):
        super().__init__()
        self._llm = llm
        self._encoder = encoder

    def next_token(self, tokens: list[int], mask: set[int] = None) -> int:
        """Returns the next token for the provided tokens."""
        logits = self.get_logits(tokens, mask)
        
        best_token = int(np.argmax(logits))

        return best_token

    def set_instruction(self, new: list[int] | str) -> None:
        """Sets the instruction with information for LLM."""

        if new is str:
            new = self._encoder.encode(new)
        self._t_instruction = new
    
    def update_instruction(self, new: list[int] | str) -> None:
        """Updates the instruction with new information for LLM."""

        if new is str:
            new = self._encoder.encode(new)
        self._t_instruction += new

    def get_logits(self, tokens: list[int], mask = None) -> list[int]:
        """
        Returns the list of logits for provided tokens. 
        Applies the mask optionally.
        """
        l = self._llm.get_logits_from_input_ids(self._t_instruction + tokens)
        if mask is not None:
            l = self._apply_mask(mask, l)
        print(f'{self._encoder.decode(self._t_instruction + tokens)}')
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

    @property
    def encoder(self) -> Encoder:
        return self._encoder
