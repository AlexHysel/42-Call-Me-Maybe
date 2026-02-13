from src.llm_sdk import Small_LLM_Model
import json
import numpy as np

class CallMyMaybe:
    def __init__(self) -> None:
        self.llm = Small_LLM_Model()
        path = self.llm.get_path_to_vocabulary_json()
        with open(path, 'r', encoding='utf-8') as f:
            self.word_ids = json.load(f)

        self.vocab = [None] * len(self.word_ids)
        for k, v in self.word_ids.items():
            self.vocab[v] = k
        
    def apply_mask(self, allowed_ids: list[int], logits):
        masked = np.full_like(logits, -float('inf'))
        for id in allowed_ids:
            masked[id] = logits[id]
        return masked

    def translate(self, text: str) -> str:
        return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')

    def encode(self, text: str) -> list[int]:
        text = self.translate(text)
        ids = []
        while text:
            match_id = None
            match_len = 0
            for i in range(1, len(text) + 1):
                substr = text[:i]
                if substr in self.word_ids:
                    match_id = self.word_ids[substr]
                    match_len = len(substr)
            
            if match_id is not None:
                ids.append(match_id)
                text = text[match_len:]
            else:
                text = text[1:]
        return ids

    def print_constant(self, text: str, prompt_ids: list[int]):
        text_ids = self.encode(text)
        for t_id in text_ids:
            logits = self.llm.get_logits_from_input_ids(prompt_ids)
            masked_logits = self.apply_mask([t_id], logits)
            next_token = int(np.argmax(masked_logits))
            prompt_ids.append(next_token)
        return prompt_ids

    def process_operation(self, prompt: str):
        prompt_ids = self.encode(prompt)
        
        prompt_ids = self.print_constant('{\n\t"fn_name": "fn_', prompt_ids)
        
        print(self.llm._tokenizer.decode(prompt_ids))
        return prompt_ids


if __name__ == "__main__":
    cmm = CallMyMaybe()
    cmm.process_operation('Who is Einstein?\n')