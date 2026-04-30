from src.llm_sdk import get_path_to_vocabulary_json
from src.encoder import Encoder
from src.llm import LLM
from src.callmemaybe import CallMeMaybe
import json

def create_encoder(vocab_path) -> Encoder:
    with open(vocab_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    return Encoder(tokens)


if __name__ == "__main__":
    encoder = create_encoder(get_path_to_vocabulary_json())
    llm = LLM(encoder)
    cmm = CallMeMaybe(llm)

    prompts = None
    with open('input/function_calling_tests.json') as requests:
        prompts = [t['prompt'] for t in json.load(requests)]
    output = open('output/function_calling_results.json', 'w')
    output.write('[\n')
    for i, p in enumerate(prompts):
        print(f'\n{i}. Processing \'{p}\'...')
        if i < len(prompts) - 1:
            output.write(cmm.process_func(p) + ',\n')
        else:
            output.write(cmm.process_func(p) + '\n')
    output.write(']')
    output.close()
    print('Finished.')
  