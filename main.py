from src.llm_sdk import Small_LLM_Model


llm = Small_LLM_Model()
tokenizer = llm._tokenizer

print(tokenizer.encode('word'))
