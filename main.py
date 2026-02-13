from src.llm_sdk import Small_LLM_Model
import torch
import json
import numpy


def apply_mask(allowed_ids: list[int], logits):
    """Returns logits where forbidden ids are set to -infinity."""
    masked = [-float('infinity')] * len(logits)
    for id in allowed_ids:
        masked[id] = logits[id]
    
    return masked

def translate(text: str) -> str:
    """Translates text replacing spaces, tabs and other characters with special charachters"""
    return text.replace(' ', 'Ġ').replace('\n', 'Ċ')

def encode(text: str, word_ids: list[str]) -> list[int]:
    text = translate(text)
    ids = []
    while text:
        print(text)
        id = None
        cut = None
        for i in range(0, len(text) + 1):
            substr = text[:i]
            if substr in word_ids:
                cut = len(substr)
                id = word_ids[substr]
        text = text[cut:]
        ids.append(id)
    return ids


llm = Small_LLM_Model()

vocab_path = llm.get_path_to_vocabulary_json()
word_ids = json.loads(open(vocab_path, 'r', encoding='utf_8').read())
vocab = [None] * len(word_ids)
for k, v in word_ids.items():
    vocab[v] = k


ids = encode('Who is Einstein?', word_ids)
print(ids)
print(llm._tokenizer.decode(ids))

#for a in range(1, 3):
    # Contains score for all words in dictionary
#    logits = llm.get_logits_from_input_ids(ids)

    #torch.tensor([2.45, 4.1, 1.4, -1.0])
#    best_id = get_masked_id(['physics', 'brick', 'war'], word_ids, vocab, logits)

#    print(vocab[best_id])
#    ids.append(best_id)