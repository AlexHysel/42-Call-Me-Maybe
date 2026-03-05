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
        
        # INSTRUCTION
        self.instruction = self.encode("Convert question to " +
            "JSON question following its structure. " +
            f"Allowed functions:\n\n")
        for name, data in self.funcs.items():
            types = ''
            for t in data['args_types'].values():
                types += t + ', '
            self.instruction += self.encode(f"- {name} \n   " +
                                            f"argument types: {types}\n\n")

    def apply_mask(self, allowed_ids: list[int], logits) -> list[int]:
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

    def all_tokens(self, text: str) -> list[int]:
        """Returns all tokens whose decoded form appears in text"""
        ids = set()

        for token_id in range(len(self.vocab)):
            decoded = self.decode([token_id])

            if decoded in text:
                ids.add(token_id)

        return list(ids)

    def decode(self, tokens: List[int]) -> str:
        """Translates tokens to text"""
        result = ""
        for token in tokens:
            result += self.vocab[token]
        return result.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')

    def get_logits(self, prompt_ids: list[int], mask = None) -> list[int]:
        """Returns logits for current ids. Adds mask optionally"""
        l = self.llm.get_logits_from_input_ids(self.instruction + prompt_ids)
        if mask is not None:
            l = self.apply_mask(mask, l)
        return l

    def get_func(self, prompt_ids: list[int]) -> list[int]:
        """Returns tokens of function that should be used"""
        funcs = [self.encode(k) for k in self.funcs.keys()]
        result = []
        
        while funcs:
            step = {f[0] for f in funcs}
            logits = self.get_logits(prompt_ids + result, list(step))
            next_token = int(np.argmax(logits))
            result.append(next_token)
            funcs = [f[1:] for f in funcs if f[0] == next_token and len(f) > 1]
        return result

    def add_args(self, func, prompt_ids, text) -> list[int]:
        prompt_ids += self.encode('\n\t\t"arguments": {')
        text = text.replace(' ', 'Ġ')#.replace('\n', 'Ċ').replace('\t', 'ĉ')
        allowed = {
            a for a in self.all_tokens(text + '"},0.')
            if a == self.encode('"')[0] or '"' not in self.decode([a])
        }

        for i, arg in enumerate(func['args_names']):
            if i > 0:
                prompt_ids += self.encode(', ')

            prompt_ids += self.encode(f'"{arg}": ')

            is_str = func['args_types'][arg] == 'str'
            if is_str:
                prompt_ids += self.encode('"')

            while True:
                logits = self.get_logits(prompt_ids, allowed)
                best = self.decode([int(np.argmax(logits))])
                
                if ',' in best or '}' in best:
                    break

                prompt_ids += self.encode(best)
                print(self.decode(prompt_ids))

                if is_str and '"' in best and '\\"' not in best:
                    break
                
        prompt_ids += self.encode('}\n')
        return prompt_ids

    def process_func(self, prompt: str) -> str:
        """Process single operation"""
        prompt = prompt.replace('\n', '\\n').replace('\t', '\\t').replace('\"', '\\"')
        prompt_ids = self.encode('\t{\n\t\t"prompt": "' + prompt + '",\n')

        prompt_ids += self.encode('\t\t"function": "')
        func = self.get_func(prompt_ids)
        prompt_ids += func
        prompt_ids += self.encode('",')

        prompt_ids = self.add_args(self.funcs[self.decode(func)], prompt_ids, prompt)

        prompt_ids += self.encode('\t}')

        return self.decode(prompt_ids)


if __name__ == "__main__":
    cmm = CallMyMaybe()
    print(cmm.word_ids)
    prompts = None
    with open('input/function_calling_tests.json') as requests:
        prompts = [t['prompt'] for t in json.load(requests)]
    output = open('output/function_calling_results.json', 'w')
    output.write('[\n')
    for i, p in enumerate(prompts):
        print(f'{i}. Processing \'{p}\'...')
        if i < len(prompts) - 1:
            output.write(cmm.process_func(p) + ',\n')
        else:
            output.write(cmm.process_func(p) + '\n')
    output.write(']')
    output.close()
  