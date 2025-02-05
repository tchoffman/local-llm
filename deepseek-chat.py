from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def initialize_model():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    try:
        model, tokenizer = initialize_model()
        print("\nModel loaded! Type 'exit' to quit.")
        
        while True:
            prompt = input("\nYou: ").strip()
            if prompt.lower() == 'exit':
                print("Goodbye!")
                break
            
            if prompt:
                print("\nAI: ", end='', flush=True)
                response = generate_response(prompt, model, tokenizer)
                print(response)
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()