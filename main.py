from src.llm_sdk import Small_LLM_Model
import torch


llm = Small_LLM_Model()
tokenizer = llm._tokenizer
vocab = {v: k for k, v in llm._tokenizer.get_vocab().items()}
eof_id = tokenizer.eos_token_id

ids = tokenizer.encode('Who is Einstein?')
words = tokenizer.decode(ids)

print(llm.get_path_to_vocabulary_json())

while ids[-1] is not eof_id:
    # Contains score for all words in dictionary
    logits = llm.get_logits_from_input_ids(ids)

    #torch.tensor([2.45, 4.1, 1.4, -1.0])
    best_id = torch.argmax(torch.tensor(logits)).item()

    print(vocab[best_id])
    ids.append(best_id)
