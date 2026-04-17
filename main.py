from src.llm_sdk import Small_LLM_Model
import json
import numpy as np


class CallMeMaybe:
    def __init__(self) -> None:
        self.llm = Small_LLM_Model()

        with open(self.llm.get_path_to_vocabulary_json(), 'r', encoding='utf-8') as f:
            self.word_ids = json.load(f)

        self.vocab = [None] * len(self.word_ids)
        for word, token in self.word_ids.items():
            self.vocab[token] = word

        with open('input/functions_definition.json') as file:
            self.func_definitions = {e['fn_name']: e for e in json.load(file)}
    
        self.instruction = self.encode("Convert question to " +
            "JSON question following its structure. " +
            f"Allowed functions:\n\n")
        
        for name, data in self.func_definitions.items():
            types = ''
            for t in data['args_types'].values():
                types += t + ', '
            self.instruction += self.encode(f"- {name} \n\t" +
                                            f"argument types: {types}\n\n")
            
    def special_to_standart(text: str) -> str:
        """
        Replaces special AI characters for space, tab and new line with
        standart human ones
        """
        return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
    
    def standart_to_special(text: str) -> str:
        """
        Replaces standart spaces, tabs and new line chars with special ones
        AI can understand
        """
        return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')

    def apply_mask(self, allowed_ids: list[int], logits) -> list[int]:
        """
        Returns logits with mask applied by setting all forbidden 
        token scores to -infinity
        """
        masked = np.full_like(logits, -float('inf'))
        for id in allowed_ids:
            masked[id] = logits[id]
        return masked

    def encode(self, text: str) -> list[int]:
        """Translate text to list of tokens for LLM"""

        text = self.standart_to_special(text)
        ids = []
        while text:
            match_id = None
            match_len = -2
            for i in range(0, len(text) + 1):
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

    def all_tokens(self, text: str, allowed_chars: str = "") -> list[int]:
        """Returns the list of all possible tokens from the string"""

        ids = set()
        char_set = set(text) | set(allowed_chars)
        
        for token_id in range(len(self.vocab)):
            decoded = self.decode([token_id])
            
            if all(c in char_set for c in decoded):
                ids.add(token_id)
        return list(ids)

    def decode(self, tokens: List[int]) -> str:
        """Translates LLM tokens to human-readable text"""
        
        result = ""
        for token in tokens:
            result += self.vocab[token]
        return self.special_to_standart(result)

    def get_logits(self, prompt_ids: list[int], mask = None) -> list[int]:
        """
        Returns the list of logits for provided ids. 
        Applies the mask optionally
        """

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
        
        prompt_content_ids = set(self.all_tokens(text))
        
        json_struct_ids = set(self.all_tokens(' {},0.123456789"'))
        
        regex_chars = ' \\d[]+*-^$|?_()a-zABCDEFGHIJKLMNOPQRSTUVWXYZ'
        regex_ids = set(self.all_tokens(text + regex_chars))

        for i, arg in enumerate(func['args_names']):
            if i > 0: prompt_ids += self.encode(', ')
            prompt_ids += self.encode(f'"{arg}": ')
            
            arg_type = func['args_types'][arg]
            is_str = arg_type == 'str'
            
            # --- ЛОГИКА ОГРАНИЧЕНИЯ СЛОВАРЯ ---
            if arg == 'regex':
                # Для регулярок даем полную свободу
                allowed_ids = list(regex_ids)
            elif is_str:
                # Для обычных строк (имени Шрека и т.д.) разрешаем ТОЛЬКО 
                # то, что было в вопросе + кавычки/пробелы
                allowed_ids = list(prompt_content_ids | json_struct_ids)
            else:
                # Для чисел и прочего
                allowed_ids = list(json_struct_ids)

            if is_str: prompt_ids += self.encode('"')
            
            generated_arg_tokens = []
            for _ in range(60):
                logits = self.get_logits(prompt_ids, allowed_ids)
                
                # Штраф за повторы всё равно оставим, чтобы не зациклилась
                for token_id in set(generated_arg_tokens):
                    logits[token_id] -= 10.0 
                
                best_id = int(np.argmax(logits))
                best_text = self.decode([best_id])
                
                if is_str:
                    if '"' in best_text and '\\"' not in best_text:
                        parts = best_text.split('"')
                        if parts[0]:
                            prompt_ids.extend(self.encode(parts[0]))
                        break
                else:
                    if ',' in best_text or '}' in best_text:
                        break
                
                prompt_ids.append(best_id)
                generated_arg_tokens.append(best_id)
            
            if is_str: prompt_ids += self.encode('"')

        prompt_ids += self.encode('}\n')
        return prompt_ids

    def process_func(self, prompt: str) -> str:
        """Process single operation"""
        prompt = prompt.replace('\n', '\\n').replace('\t', '\\t')
        prompt_ids = self.encode('\t{\n\t\t"prompt": "' + prompt + '",\n')
        prompt_ids += self.encode('\t\t"function": "')
        func = self.get_func(prompt_ids)
        prompt_ids += func
        prompt_ids += self.encode('",')
        prompt_ids = self.add_args(self.funcs[self.decode(func)], prompt_ids, prompt)
        prompt_ids += self.encode('\t}')
        return self.decode(prompt_ids)


if __name__ == "__main__":
    cmm = CallMeMaybe()
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
  