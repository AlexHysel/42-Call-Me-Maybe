from llm_sdk import Small_LLM_Model
from src.encoder import Encoder
from src.llm import LLM
from src.callmemaybe import CallMeMaybe
import json


def create_encoder(vocab_path) -> Encoder:
    with open(vocab_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    return Encoder(tokens)


if __name__ == "__main__":
    llm_model = Small_LLM_Model()
    encoder = create_encoder(llm_model.get_path_to_vocabulary_json())
    llm = LLM(llm_model, encoder)
    cmm = CallMeMaybe(llm)

    prompts = None
    with open('data/input/function_calling_tests.json') as requests:
        prompts = [t['prompt'] for t in json.load(requests)]
    output = open('data/output/function_calling_results.json', 'w')
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
