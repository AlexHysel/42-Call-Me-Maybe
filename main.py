from src.llm_sdk import Small_LLM_Model
import json
import numpy as np
import re


class CallMeMaybe:
    def __init__(self) -> None:
        self.llm = Small_LLM_Model()        # LLM
        self.tokens:        dict[str, int]  # Dictionary returns the token of the word
        self.vocab:         list[str]       # List of the words
        self.functions = dict()             # Tokenized definitions
        self.t_instruction: list[int]       # Tokenized basic instruction
        self.t_full_instruction = []

        vocab_path = self.llm.get_path_to_vocabulary_json()
        definitions_path = 'input/functions_definition.json'

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.tokens = json.load(f)

        self.t_instruction = self.encode("Convert question to " +
            "JSON question following its structure. " +
            f"Allowed functions:\n\n")
        self.t_full_instruction = self.t_instruction + self.encode('\nAllowed functions:')

        with open(definitions_path) as file:
            for func in json.load(file):
                data = dict()
                data['t_name'] = self.encode(func['fn_name'])
                data['args_names'] = func['args_names']
                data['args_types'] = {k: self.encode(v) for k, v in func['args_types'].items()}
                self.functions[func['fn_name']] = data
                self.t_full_instruction += self.encode('\n' + func['fn_name'])

        self.vocab = [None] * len(self.tokens)
        for word, token in self.tokens.items():
            self.vocab[token] = word
        
        self.number_ids = self.encode_all('}, .0123456789"')
        self.regex_ids = self.encode_all('"\\d+[]*-^$|?_()a-zA-Z0-9[aeiou][AEIOU][aeiouAEIOU]')

    @staticmethod
    def special_to_standart(text: str) -> str:
        """
        Replaces special AI characters for space, tab and new line with
        standart human ones
        """
        return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
    
    @staticmethod
    def standart_to_special(text: str) -> str:
        """
        Replaces standart spaces, tabs and new line chars with special ones
        AI can understand
        """
        return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')

    @staticmethod
    def escape(text: str) -> str:
        return text.replace('\\', '\\\\').replace('"', '\\"')

    # === ENCODING and DECODING ===

    def encode(self, text: str) -> list[int]:
        """Translates human text to a list of tokens for LLM"""

        text = CallMeMaybe.standart_to_special(text)
        ids = []
        while text:
            match_id = None
            match_len = -2
            for i in range(0, len(text) + 1):
                substr = text[:i]
                if substr in self.tokens:
                    match_id = self.tokens[substr]
                    match_len = len(substr)
            if match_id is not None:
                ids.append(match_id)
                text = text[match_len:]
            else:
                text = text[1:]

        return ids
    
    def encode_words(self, text: str) -> set[int]:
        words = re.findall(r'\b\w+\b', text)
        ids = set()
        for word in words:
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

    def decode(self, tokens: List[int]) -> str:
        """Translates LLM tokens to human-readable text"""
        
        result = ""
        for token in tokens:
            result += self.vocab[token]
        return CallMeMaybe.special_to_standart(result)

    # === LLM and Constrained Decoding ===

    def get_logits(self, prompt_ids: list[int], instruction: list[int], mask = None) -> list[int]:
        """
        Returns the list of logits for provided ids. 
        Applies the mask optionally
        """

        l = self.llm.get_logits_from_input_ids(instruction + prompt_ids)
        if mask is not None:
            l = self.apply_mask(mask, l)
        return l

    def apply_mask(self, allowed_ids: list[int], logits) -> list[int]:
        """
        Returns logits with mask applied by setting all forbidden 
        token scores to -infinity
        """
        masked = np.full_like(logits, -float('inf'))
        for id in allowed_ids:
            masked[id] = logits[id]
        return masked

    def define_function(self, prompt_ids: list[int]) -> dict:
        """Returns the function that should be used"""

        # candidates: list of (remaining_token_ids, full_fn_name)
        candidates = [
            (data['t_name'], fn_name)
            for fn_name, data in self.functions.items()
        ]
        result = []

        while candidates:
            # valid next tokens = first token of each remaining candidate
            allowed = {tokens[0] for tokens, _ in candidates}
            logits = self.get_logits(
                prompt_ids + result,
                self.t_full_instruction,
                mask=allowed
            )
            next_token = int(np.argmax(logits))
            result.append(next_token)
            candidates = [
                (tokens[1:], name)
                for tokens, name in candidates
                if tokens[0] == next_token and len(tokens) > 1
            ]

        return self.functions[self.decode(result)]

    def add_args(self, function, prompt_ids: list[int], text: str) -> list[int]:
        prompt_ids += self.encode('\n\t\t"arguments": {')

        definition = function['t_name'] + self.encode(': ')
        for arg_name in function['args_names']:
            definition += self.encode(f'\n{arg_name}: ')
            definition += function['args_types'][arg_name]
        
        for i, arg_name in enumerate(function['args_names']):
            arg_type = self.decode(function['args_types'][arg_name])

            if i > 0: prompt_ids += self.encode(', ')
            prompt_ids += self.encode(f'"{arg_name}": ')
            
            if arg_type == 'str':
                allowed_ids = self.encode_words(text)
                for char in '"Ġ,*':
                    allowed_ids.update(set(self.encode(char)))
                if arg_name == 'regex':
                    allowed_ids.update(self.regex_ids)
                prompt_ids += self.encode('"')
            else:
                allowed_ids = self.number_ids
                
            generated_arg_tokens = []
            for _ in range(60):
                logits = self.get_logits(prompt_ids, self.t_instruction + definition, allowed_ids)
                
                for token_id in set(generated_arg_tokens):
                    logits[token_id] -= 15.0
                
                best_id = int(np.argmax(logits))
                best_text = self.decode([best_id])
                
                if arg_type == 'str':
                    if '"' in best_text and '\\"' not in best_text:
                        break
                else:
                    if ',' in best_text or '}' in best_text:
                        break
                
                prompt_ids.append(best_id)
                generated_arg_tokens.append(best_id)
    
            if arg_type == 'str':
                prompt_ids += self.encode('"')

        prompt_ids += self.encode('}\n')
        return prompt_ids

    def process_func(self, prompt: str) -> str:
        """Process single function"""

        prompt = self.escape(prompt)
        text = '\t{\n\t\t"prompt": "' + prompt + '",\n\t\t"function": "'
        t_result = self.encode(text)
        function = self.define_function(t_result)
        t_result += function['t_name']
        t_result += self.encode('",')
        t_result = self.add_args(function, t_result, prompt)
        t_result += (self.encode('\t}'))
        return self.decode(t_result)


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
    print('Finished.')
  