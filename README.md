*This project has been created as part of the 42 curriculum by \<your_login\>.*

# Call Me Maybe

## Description

A function calling system that translates natural language prompts into structured JSON function calls using a small 0.6B parameter language model (Qwen3-0.6B). The key challenge is achieving near-perfect reliability with a model that would otherwise produce valid JSON only ~30% of the time. This is solved through **constrained decoding** — guiding the model's output token-by-token to guarantee valid structure.

Given a prompt like `"What is the sum of 2 and 3?"`, the system produces:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}
```

## Instructions

### Installation

```bash
# Clone the repository
git clone <your_repo_url>
cd call-me-maybe

# Copy llm_sdk into the project root
cp -r /path/to/llm_sdk ./llm_sdk

# Install dependencies
make install
# or manually:
uv sync
```

### Running

```bash
# Default paths (data/input/ → data/output/)
make run

# Custom paths
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Other commands

```bash
make debug    # Run with pdb debugger
make lint     # Run flake8 and mypy
make clean    # Remove caches
```

## Algorithm Explanation

### Constrained Decoding

The system generates output token-by-token, restricting which tokens are allowed at each step:

1. **Function selection**: Only tokens that are valid continuations of known function names are allowed. The model picks the most likely token from this restricted set, progressively narrowing candidates until one function remains.

2. **Argument generation**: For each argument, a mask of allowed tokens is built based on argument type:
   - `number`: only digit/punctuation tokens (`0-9`, `.`, `,`, `}`)
   - `boolean`: only `true`/`false` tokens
   - `string`: tokens extracted from the original prompt via word tokenization, ensuring the model can only output words that appeared in the input

3. **Regex arguments**: Handled via keyword matching — the prompt is scanned for known patterns (`vowels`, `digits`, `spaces`, etc.) and mapped to the corresponding regex pattern (`[aeiouAEIOU]`, `\d+`, `\s+`, etc.)

4. **Stopping conditions**: String generation stops when a closing `"` is encountered. Numeric generation stops on `,` or `}`. A confidence threshold prevents infinite generation.

### Chat Template

The model is prompted using Qwen's native tool-calling format:

```
<|im_start|>system
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "fn_add_numbers", ...}
</tools>
For each function call, return a json object within <tool_call></tool_call> tags:
...
<|im_end|>
<|im_start|>user
What is the sum of 2 and 3?
<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "fn_add_numbers", "arguments": {"a": 2, "b": 3}}
```

This matches Qwen3's training format, significantly improving argument extraction quality.

## Design Decisions

- **Pydantic everywhere**: All classes (`Encoder`, `LLM`, `Function`, `CallMeMaybe`) use pydantic `BaseModel` for validation and type safety.
- **Separate Encoder class**: Tokenization is decoupled from LLM inference, making both independently testable.
- **Short instruction during arg generation**: When generating arguments, only the current function's schema is kept in the instruction context (not all functions). This reduces token count and speeds up logit computation — which scales O(n²) with context length.
- **Keyword-based regex resolution**: Rather than asking the LLM to generate regex patterns (unreliable for 0.6B models), patterns are resolved deterministically from prompt keywords.
- **Native tool-call format**: Using Qwen's documented tool-calling template instead of a custom JSON format leverages the model's existing training.

## Performance Analysis

- **Accuracy**: ~87–91% on the provided test set (21–22/24 prompts fully correct)
- **Main failure cases**: Unusual names not appearing as single tokens (e.g., "shrek" tokenized as sub-parts), and `replacement` values described by word rather than symbol (e.g., "asterisks" instead of "*")
- **Speed**: ~2–4 seconds per prompt on CPU with float32, ~1–2s on GPU with float16. Full 24-prompt run completes well under 5 minutes.
- **JSON validity**: 100% — constrained decoding guarantees parseable output on every run.

## Challenges Faced

- **Token boundary mismatch**: Words like "shrek" may be split across multiple tokens by BPE tokenization. The model selecting individual sub-tokens instead of the full word was the primary source of name extraction errors. Solved partially by including both word and space-prefixed-word encodings in the mask.
- **LLM generating regex**: Early attempts to have the LLM generate regex patterns directly failed — the 0.6B model would mix keyword descriptions with pattern syntax. Replaced with deterministic keyword mapping.
- **Pydantic PrivateAttr**: Pydantic requires `PrivateAttr()` for private fields and `super().__init__()` + post-assignment for initialization. Standard Python `__init__` patterns don't work.
- **Context length and speed**: Including all function definitions in every logit call was slow. Solved by switching to a short instruction (single function schema) during argument generation.

## Testing Strategy

- Ran the provided `function_calling_tests.json` and manually verified each output against expected values.
- Tested edge cases: floating point numbers (`10.1`, `20.83`), single-character replacements (`*`), multi-word source strings, repeated words in source.
- Verified JSON validity by parsing all outputs with `json.loads()`.
- Tested CLI arguments with custom paths to ensure defaults and overrides work correctly.
- Tested error handling: missing input files, malformed JSON in input, empty function definitions.

## Example Usage

```bash
# Run with default test files
uv run python -m src

# Run with custom files
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

Example output (`data/output/function_calling_results.json`):

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0}
    },
    {
        "prompt": "Greet john",
        "name": "fn_greet",
        "parameters": {"name": "john"}
    }
]
```

## Resources

- [Qwen3 documentation](https://qwen.readthedocs.io/en/latest/)
- [Qwen tool calling format](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Constrained decoding overview](https://huggingface.co/docs/transformers/main/en/generation_strategies#constrained-decoding)

### AI Usage

Claude (Anthropic) was used throughout this project for:
- Debugging pydantic initialization patterns (`PrivateAttr`, `super().__init__()`)
- Designing the constrained decoding mask logic for different argument types
- Identifying the chat template format for Qwen3 tool calling
- Suggesting the keyword-mapping approach for regex pattern resolution