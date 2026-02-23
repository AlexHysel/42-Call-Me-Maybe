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

        with open('input/functions_definition.json') as file:
            self.funcs = {e['fn_name']: e for e in json.load(file)}
        for func in self.funcs.values():
            func.pop('fn_name', None)
            func.pop('return_type', None)

        self.instruction = self.encode("You need to convert question to " +
            "JSON question following its structure "+
            f"Allowed functions: {self.funcs.keys()}")
        
    def apply_mask(self, allowed_ids: list[int], logits):
        """Applies mask by setting all forbidden token scores to -infinity"""
        masked = np.full_like(logits, -float('inf'))
        for id in allowed_ids:
            masked[id] = logits[id]
        return masked

    def encode(self, text: str) -> list[int]:
        """Translate text to list of tokens"""
        text = text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')
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
    
    def decode(self, tokens: List[int]):
        result = ""
        for token in tokens:
            result += self.vocab[token]
        return result.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
    
    def get_logits(self, prompt_ids: list[int], mask = None):
        l = self.llm.get_logits_from_input_ids(self.instruction + prompt_ids)
        if mask is not None:
            l = self.apply_mask(mask, l)
        return l

    def get_func(self, prompt_ids: list[int]):
        """Used to choose between operations and return tokens of chosen"""
        funcs = [self.encode(k) for k in self.funcs.keys()]
        result = []
        while funcs:
            step = {f[0] for f in funcs}
            logits = self.get_logits(prompt_ids, step)
            next_token = int(np.argmax(logits))
            result.append(next_token)
            funcs = [f[1:] for f in funcs if f[0] == next_token and len(f) > 1]
        return result

    def add_args(self, func: dict[str: list[str], str: str], prompt_ids: list[int]):
        """Used to add line with arguments"""
        prompt_ids += self.encode('\n\t"arguments": {')
        args = func['args_names']
        for i, arg in enumerate(args):
            if i != 0:
                prompt_ids += self.encode(', ')
            prompt_ids += self.encode(f'"{arg}": ')
            if func['args_types'][arg] == 'str':
                prompt_ids += self.encode('"')
            while True:
                logits = self.get_logits(prompt_ids)
                next = self.decode([int(np.argmax(logits))])
                flag = False
                for i in range(len(next) -1, -1, -1):
                    if next[i] in [',', '}', '\n']:
                        next = next[0:i]
                        flag = True
                prompt_ids += self.encode(next)
                if flag:
                    break
        prompt_ids += self.encode('}\n')
        return prompt_ids

    def process_func(self, prompt: str):
        """Process single operation"""
        prompt_ids = self.encode('{\n\t"prompt": "' + prompt + '",\n')

        prompt_ids += self.encode('\t"function": "')
        func = self.get_func(prompt_ids)
        prompt_ids += func
        prompt_ids += self.encode('",')

        prompt_ids = self.add_args(self.funcs[self.decode(func)], prompt_ids)

        prompt_ids += self.encode('}')

        print(self.decode(prompt_ids))
        return prompt_ids


if __name__ == "__main__":
    cmm = CallMyMaybe()
    prompts = None
    with open('input/function_calling_tests.json') as requests:
        prompts = [t['prompt'] for t in json.load(requests)]
    for p in prompts:
        cmm.process_func(p)