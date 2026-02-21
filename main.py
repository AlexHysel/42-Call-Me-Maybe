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

        operations = None
        with open('input/functions_definition.json') as file:
            operations = [e['fn_name'] for e in json.load(file)]
        self.intructions = ("You need to convert question to " +
            "JSON question following its structure "+
            f"Allowed functions: {operations}")
        self.operations = [self.encode(o) for o in operations]
        
    def apply_mask(self, allowed_ids: list[int], logits):
        """Applies mask by setting all forbidden token scores to -infinity"""
        masked = np.full_like(logits, -float('inf'))
        for id in allowed_ids:
            masked[id] = logits[id]
        return masked

    def translate(self, text: str) -> str:
        """Replace spaces, tabs and new line chars with special ones AI understands"""
        return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')

    def encode(self, text: str) -> list[int]:
        """Translate text to list of tokens"""
        text = self.translate(text)
        ids = []
        while text:
            match_id = None
            match_len = -2
            for i in range(-1, len(text) + 1):
                substr = text[:i]
                if substr in self.word_ids:
                    match_id = self.word_ids[substr]
                    match_len = len(substr)
            
            if match_id is not None:
                ids.append(match_id)
                text = text[match_len:]
            else:
                text = text[-1:]
        return ids
    
    def print_operation(self, prompt_ids: list[int]):
        """Used to choose between operations"""
        operations = self.operations
        while operations:
            step = {o[0] for o in operations}
            logits = self.llm.get_logits_from_input_ids(prompt_ids)
            logits = self.apply_mask(step, logits)
            next_token = int(np.argmax(logits))
            prompt_ids.append(next_token)
            operations = [o[1:] for o in operations if o[0] == next_token and len(o) > 1]
        return prompt_ids

    def print_constant(self, text: str, prompt_ids: list[int]):
        """Used to print smth using AI, just like a printer."""
        text_ids = self.encode(text)
        for t_id in text_ids:
            logits = self.llm.get_logits_from_input_ids(prompt_ids)
            logits = self.apply_mask([t_id], logits)
            next_token = int(np.argmax(logits))
            prompt_ids.append(next_token)
        return prompt_ids

    def process_operation(self, prompt: str):
        """Process single operation"""
        prompt_ids = self.encode(self.intructions + '\n' + prompt + '\n')
        
        prompt_ids += self.encode('{\n\t"function": "')
        prompt_ids = self.print_operation(prompt_ids)

        prompt_ids += self.encode('",\n\t"arguments": {"')

        prompt_ids += self.encode('}\n}')
        # SHOULD BE REPLACED
        print(self.llm._tokenizer.decode(prompt_ids))
        return prompt_ids


if __name__ == "__main__":
    cmm = CallMyMaybe()
    prompts = None
    with open('input/function_calling_tests.json') as requests:
        prompts = [t['prompt'] for t in json.load(requests)]
    for p in prompts:
        cmm.process_operation(p)