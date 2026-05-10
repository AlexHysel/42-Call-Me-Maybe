import json
import re

import numpy as np
from pydantic import BaseModel

from src.encoder import Encoder
from src.function import Function
from src.llm import LLM
from src.utils import escape


REGEX_MAPPING = [
    (['vowel', 'vowels'], r'[aeiouAEIOU]'),
    (
        ['consonant', 'consonants'],
        r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]',
    ),
    (['digit', 'digits', 'number', 'numbers'], r'\\d+'),
    (['uppercase', 'upper', 'capital'], r'[A-Z]+'),
    (['lowercase', 'lower'], r'[a-z]+'),
    (['letter', 'letters', 'alphabetic'], r'[a-zA-Z]+'),
    (['space', 'spaces', 'whitespace'], r'\\s+'),
    (['punctuation', 'special'], r'[^\w\s]'),
    (['alphanumeric'], r'\\w+'),
    (['newline', 'newlines'], r'\\n+'),
    (['tab', 'tabs'], r'\\t+'),
]


class CallMeMaybe(BaseModel):
    llm: LLM
    encoder: Encoder
    functions: dict[str, Function]
    t_defintions: list[int]
    t_numbers: set[int]
    t_boolean: set[int]
    t_instruction_prefix: list[int]
    t_instruction_suffix: list[int]

    def __init__(self, llm: LLM, func_definitons: str) -> None:
        encoder = llm.encoder

        functions = {}
        with open(func_definitons, 'r') as f:
            for func in json.load(f):
                functions[func['name']] = Function(func, encoder)

        t_defintions = [token for func in functions.values() for token in func.t_definition]

        t_instruction_prefix = encoder.encode(
            '<|im_start|>system\n'
            'You are provided with function signatures '
            'within <tools></tools> XML tags:\n'
            '<tools>\n')
        t_instruction_suffix = encoder.encode(
            '</tools>\n'
            'For each function call, return a json '
            'object within <tool_call></tool_call> tags:\n'
            '<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            '</tool_call>\n'
            '<|im_end|>\n')

        super().__init__(
            llm=llm,
            encoder=encoder,
            functions=functions,
            t_numbers=encoder.encode_all('}, .0123456789"'),
            t_boolean=set(encoder.encode('true') + encoder.encode('false')),
            t_defintions=t_defintions,
            t_instruction_prefix=t_instruction_prefix,
            t_instruction_suffix=t_instruction_suffix
        )
        self.set_tools()

    def set_tools(self, func: Function | None = None) -> None:
        """Updates the LLM context with function definitions."""

        if func is not None:
            definitions = func.t_definition
        else:
            definitions = self.t_defintions
        new = self.t_instruction_prefix + definitions
        new += self.t_instruction_suffix
        self.llm.set_instruction(new)

    def define_function(self, tokens: list[int]) -> Function:
        """Selects the best matching function using constrained decoding."""

        candidates = [(f.t_name, n) for n, f in self.functions.items()]
        result: list[int] = []

        while candidates:
            if len(candidates) == 1:
                _, name = candidates[0]
                return self.functions[name]

            next_token = self.llm.next_token(
                tokens + result,
                {t_name[0] for t_name, _ in candidates}
            )
            result.append(next_token)
            candidates = [
                (t_name[1:], name)
                for t_name, name in candidates
                if t_name[0] == next_token and len(t_name) > 1
            ]

        return self.functions[self.encoder.decode(result)]

    def get_arg(self,
                arg_type: str,
                prompt_ids: list[int],
                mask: set[int]) -> list[int]:
        """Generates a single argument value using constrained decoding."""

        generated: set[int] = set()
        arg: list[int] = []

        for i in range(60):
            logits = self.llm.get_logits(prompt_ids + arg, mask)
            logits_list = [
                l - 5.0 if idx in generated else l
                for idx, l in enumerate(logits)
            ]

            if max(logits_list) - i < 0.4:
                break

            best_id = int(np.argmax(logits_list))
            best_text = self.encoder.decode([best_id])

            if arg_type == 'string':
                if '"' in best_text and '\\"' not in best_text:
                    break
            elif ',' in best_text or '}' in best_text:
                break

            arg.append(best_id)
            generated.add(best_id)

        if arg_type == 'string':
            arg += self.encoder.encode('"')
        return arg

    def regex_pattern(self, text: str) -> list[int]:
        """Resolves the regex pattern from prompt keywords."""

        words = {w.strip('\'\".,!?').lower() for w in text.split()}
        for keywords, pattern in REGEX_MAPPING:
            if words & set(keywords):
                return self.encoder.encode(pattern)

        match = re.search(r"['\"](\w+)['\"]", text)
        if match:
            return self.encoder.encode(match.group(1))

        return self.encoder.encode(r'\w+')

    def build_string_mask(self, arg_name: str, text: str) -> set[int]:
        """Builds the allowed token mask for string arguments."""

        mask = self.encoder.encode_words(text)
        for char in '"Ġ,*':
            mask.update(self.encoder.encode(char))
        return mask

    def encode_definition(self, function: Function) -> list[int]:
        """Encodes function definition for LLM context."""

        definition = (
            function.t_name
            + self.encoder.encode(': ')
            + function.t_description
            + self.encoder.encode('\nParameters:\n')
        )
        for arg_name in function.param_names:
            definition += self.encoder.encode(f'\n{arg_name}: ')
            definition += function.t_params[arg_name]
        return definition

    def add_args(self,
                 function: Function,
                 tokens: list[int],
                 text: str) -> list[int]:
        """Generates all arguments for a function call."""

        for i, arg_name in enumerate(function.param_names):
            arg_type = function.params[arg_name]

            if i > 0:
                tokens += self.encoder.encode(', ')
            tokens += self.encoder.encode(f'"{arg_name}": ')

            if arg_name == 'regex':
                tokens += self.encoder.encode('"')
                tokens += self.regex_pattern(text)
                tokens += self.encoder.encode('"')
                continue

            if arg_type == 'string':
                mask = self.build_string_mask(arg_name, text)
                tokens += self.encoder.encode('"')
            elif arg_type == 'number':
                mask = self.t_numbers
            elif arg_type == 'boolean':
                mask = self.t_boolean
            else:
                mask = self.t_numbers

            tokens += self.get_arg(arg_type, tokens, mask)

        tokens += self.encoder.encode('}\n')
        return tokens

    # === Main entry point ===

    def process_func(self, prompt: str) -> str:
        prompt = escape(prompt)
        text = (
            '<|im_start|>user\n' +
            prompt +
            '\n<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<tool_call>\n'
            '{"name": "'
        )
        tokens = self.encoder.encode(text)
        self.set_tools()
        function = self.define_function(tokens)
        tokens += function.t_name
        tokens += self.encoder.encode('", "arguments": {')
        self.set_tools(function)
        tokens = self.add_args(function, tokens, text)
        tokens += self.encoder.encode('}')

        raw = self.encoder.decode(tokens)
        tool_json = raw[raw.find('{"name":'):]
        data = json.loads(tool_json)

        return (
            '\t{\n'
            f'\t\t"prompt": "{prompt}",\n'
            f'\t\t"name": "{data["name"]}",\n'
            f'\t\t"parameters": {json.dumps(data["arguments"])}\n'
            '\t}'
        )
