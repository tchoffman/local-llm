from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

try:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True
    )

    prompt = "What is the capital of France?"
    print(f"Processing prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
