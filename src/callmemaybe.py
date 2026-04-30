from typing import Any
from pydantic import BaseModel
import numpy as np

from src.encoder import Encoder
from src.llm import LLM
from src.utils import special_to_standart, standart_to_special, escape


class CallMeMaybe(BaseModel):
    llm: LLM
    encoder: Encoder
    functions: dict[str, dict[str, Any]]
    t_definitions: list[int]
    t_regex: set[int]
    t_numbers: set[int]
    t_instruction: list[int]
    t_regex_knowledge: list[int]

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.encoder = llm.encoder

        with open('instructions/instruction.txt', 'r') as f:
            self.t_instruction = self.encode(f.read())
        with open('instructions/regex_knowledge.txt', 'r') as f:
            self.t_regex_knowledge = self.encode(f.read())

        self.functions = dict()
        self.t_definitions = self.encode('\nAllowed functions:')
        with open('input/functions_definition.json', 'r') as file:
            for func in self.t_definitions.values():
                data = dict()
                data['name'] = self.encode(func['name'])
                data['description'] = self.encode(func['description'])
                data['args_names'] = list(func['parameters'].keys())
                data['args_types'] = {
                    k: self.encode(v['type'])
                    for k, v in func['parameters'].items()
                }
                self.functions[func['name']] = data
                self.t_definitions += self.encode('\n' + func['name'])

        self.t_numbers = self.encoder.encode_all('}, .0123456789"')
        self.t_regex = {token for r in [
            '\\d+', '[aeiouAEIOU]', '[aeiou]', '[AEIOU]', '[a-zA-Z]+',
            '[a-z]+', '[A-Z]+', '\\s+', '\\w+', '[^a-zA-Z0-9]', '\\S+'
        ] for token in self.encode(r)}

    def define_function(self, prompt_ids: list[int]) -> dict:
        """Returns the function that should be used"""

        candidates = [
            (data['name'], name)
            for name, data in self.functions.items()
        ]
        result = []

        while candidates:
            allowed = {tokens[0] for tokens, _ in candidates}
            logits = self.get_logits(
                prompt_ids + result,
                self.t_instruction + self.t_definitions,
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

    def get_arg(self, arg_type: str, arg_name: str, definition: list[int], prompt_ids, allowed_ids: set[int]) -> list[int]:
        generated_arg_tokens = set()
        arg = []
        if arg_name == 'regex':
            definition += self.t_regex_knowledge

        for i in range(60):
            
            logits = self.get_logits(prompt_ids + arg, self.t_instruction + definition, allowed_ids)
            
            for token_id in set(generated_arg_tokens):
                logits[token_id] -= 5.0

            best_token = max(logits) - i

            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            if np.max(probs) < 0.4:
                break
            
            best_id = int(np.argmax(logits))
            best_text = self.decode([best_id])

            print(f'{best_text} - {best_token:.1f}')
            
            if arg_type == 'string':
                if '"' in best_text and '\\"' not in best_text:
                    break
            else:
                if ',' in best_text or '}' in best_text:
                    break
            
            arg.append(best_id)
            generated_arg_tokens.add(best_id)

        if arg_type == 'string':
            arg += [self.tokens['"']]
        return arg

    def encode_definition(self, function) -> list[int]:
        definition = function['name'] + self.encode(': ')
        definition += function['description']
        definition += self.encode('\nParameters:\n')
        for arg_name in function['args_names']:
            definition += self.encode(f'\n{arg_name}: ')
            definition += function['args_types'][arg_name]
        return definition
        
    def add_args(self, function, prompt_ids: list[int], text: str) -> list[int]:
        prompt_ids += self.encode('\n\t\t"arguments": {')
        definition = self.encode_definition(function)

        for i, arg_name in enumerate(function['args_names']):
            arg_type = self.decode(function['args_types'][arg_name])

            if i > 0: prompt_ids += self.encode(', ')
            prompt_ids += self.encode(f'"{arg_name}": ')
            
            if arg_type == 'string':
                allowed_ids = self.encode_words(text)
                for char in '"Ġ,*':
                    allowed_ids.update(set(self.encode(char)))
                if arg_name == 'regex':
                    allowed_ids.update(self.t_regex)
                prompt_ids += self.encode('"')
            else:
                allowed_ids = self.t_numbers
            prompt_ids += self.get_arg(arg_type, arg_name, definition, prompt_ids, allowed_ids)

        prompt_ids += self.encode('}\n')
        return prompt_ids

    def process_func(self, prompt: str) -> str:
        """Process single function"""

        prompt = escape(prompt)
        text = '\t{\n\t\t"prompt": "' + prompt + '",\n\t\t"function": "'
        t_result = self.encode(text)
        function = self.define_function(t_result)
        t_result += function['name']
        t_result += self.encode('",')
        t_result = self.add_args(function, t_result, prompt)
        t_result += (self.encode('\t}'))
        return self.decode(t_result)