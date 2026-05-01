import json
from typing import Any
from pydantic import BaseModel
import numpy as np

from src.encoder import Encoder
from src.function import Function
from src.llm import LLM
from src.utils import escape


class CallMeMaybe(BaseModel):
    llm: LLM
    encoder: Encoder
    functions: dict[str, Function]
    t_definitions: list[int]
    t_regex: set[int]
    t_numbers: set[int]
    t_boolean: set[int]
    t_instruction: list[int]
    t_regex_knowledge: list[int]

    def __init__(self, llm: LLM) -> None:
        encoder = llm.encoder

        with open('instructions/instruction.txt', 'r') as f:
            t_instruction = encoder.encode(f.read())
        with open('instructions/regex_knowledge.txt', 'r') as f:
            t_regex_knowledge = encoder.encode(f.read())

        functions = dict()
        t_definitions = encoder.encode('\nAllowed functions:')
        with open('input/functions_definition.json', 'r') as file:
            for func in json.load(file):
                functions[func['name']] = Function(func, encoder)
                t_definitions += encoder.encode('\n' + func['name'])
        t_definitions += encoder.encode('\n')

        t_numbers = encoder.encode_all('}, .0123456789"')
        t_boolean = encoder.encode('true') + encoder.encode('false') 
        t_regex = {token for r in [
            '\\d+', '[aeiouAEIOU]', '[aeiou]', '[AEIOU]', '[a-zA-Z]+',
            '[a-z]+', '[A-Z]+', '\\s+', '\\w+', '[^a-zA-Z0-9]', '\\S+', '}', ',', '\"'
        ] for token in encoder.encode(r)}

        super().__init__(
            llm=llm,
            encoder=encoder,
            functions=functions,
            t_definitions=t_definitions,
            t_regex=t_regex,
            t_numbers=t_numbers,
            t_boolean=t_boolean,
            t_instruction=t_instruction,
            t_regex_knowledge=t_regex_knowledge,
        )
        self.llm.set_instruction(t_instruction)

    def define_function(self, tokens: list[int]) -> Function:
        """Returns the function that should be used."""
        self.llm.set_instruction(self.t_instruction + self.t_definitions)

        candidates = [
            (func.t_name, name)
            for name, func in self.functions.items()
        ]
        result = []

        while candidates:
            if len(candidates) == 1:
                t_name, name = candidates[0]
                result += list(t_name)
                return self.functions[name]

            next_token = self.llm.next_token(
                self.t_definitions + tokens + result,
                {t_name[0] for t_name, _ in candidates}
            )
            result.append(next_token)
            candidates = [
                (t_name[1:], name)
                for t_name, name in candidates
                if t_name[0] == next_token and len(t_name) > 1
            ]
        self.llm.set_instruction(self.t_instruction)
        return self.functions[self.encoder.decode(result)]

    def encode_definition(self, function: Function) -> list[int]:
        """Encodes function definition for LLM context."""

        definition = function.t_name + self.encoder.encode(': ')
        definition += function.t_description
        definition += self.encoder.encode('\nParameters:\n')
        for arg_name in function.param_names:
            definition += self.encoder.encode(f'\n{arg_name}: ')
            definition += function.t_params[arg_name]
        return definition

    def get_arg(self, arg_type: str, arg_name: str, prompt_ids, allowed_ids: set[int]) -> list[int]:
        generated_arg_tokens = set()
        arg = []
        if arg_name == 'regex':
            self.llm.update_instruction(self.t_regex_knowledge)

        for i in range(60):
            
            logits = self.llm.get_logits(prompt_ids + arg, allowed_ids)
            
            for token_id in set(generated_arg_tokens):
                logits[token_id] -= 5.0

            best_token = max(logits) - i

            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            if np.max(probs) < 0.4:
                break
            
            best_id = int(np.argmax(logits))
            best_text = self.encoder.decode([best_id])

            if arg_type == 'string':
                if '"' in best_text and '\\"' not in best_text:
                    break
            else:
                if ',' in best_text or '}' in best_text:
                    break
            
            arg.append(best_id)
            generated_arg_tokens.add(best_id)

        if arg_type == 'string':
            arg += self.encoder.encode('"')
        return arg 

    def add_args(self, function: Function, tokens: list[int], text: str) -> list[int]:
        """Generates all arguments for a function call."""

        tokens = tokens + self.encoder.encode('\n\t\t"arguments": {')
        self.llm.update_instruction(self.encode_definition(function))

        for i, arg_name in enumerate(function.param_names):
            arg_type = function.params[arg_name]

            if i > 0:
                tokens += self.encoder.encode(', ')
            tokens += self.encoder.encode(f'"{arg_name}": ')

            if arg_type == 'string':
                mask = self.encoder.encode_words(text)
                for char in '"Ġ,*':
                    mask.update(set(self.encoder.encode(char)))
                if arg_name == 'regex':
                    mask.update(self.t_regex)
                tokens += self.encoder.encode('"')
            elif arg_type == 'number':
                mask = self.t_numbers
            elif arg_type == 'boolean':
                mask = self.t_boolean
            
            tokens += self.get_arg(arg_type, arg_name, tokens, mask)

        self.llm.set_instruction(self.t_instruction)
        tokens += self.encoder.encode('}\n')
        return tokens

    def process_func(self, prompt: str) -> str:
        """Processes a single prompt into a function call."""

        prompt = escape(prompt)
        text = '\t{\n\t\t"prompt": "' + prompt + '",\n\t\t"function": "'
        tokens = self.encoder.encode(text)
        function = self.define_function(tokens)
        tokens += function.t_name
        tokens += self.encoder.encode('",')
        tokens = self.add_args(function, tokens, prompt)
        tokens += self.encoder.encode('\t}')
        return self.encoder.decode(tokens)