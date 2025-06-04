from mindnlp.transformers import pipeline
generator = pipeline(model="openai-community/gpt2")

prompt = "I can't believe you did such a "
outputs = generator(prompt, do_sample=False, max_new_tokens=20)
print(outputs[0]['generated_text'])
