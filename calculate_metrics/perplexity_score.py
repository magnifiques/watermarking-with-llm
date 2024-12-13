from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch

# Unified function to calculate perplexity for different GPT-2 models
def calculate_perplexity(model_name, text_without_watermark, text_with_watermark):
    
    if "gpt2-medium" in model_name.lower():
        print(f"Loading GPT-2 model: {model_name}")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        print(f"Loading model of type other than GPT-2: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    # Function to calculate perplexity
    def get_perplexity(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()
        
    print(f"Calculating perplexities for {model_name}...")
    perplexity_without_watermark = get_perplexity(text_without_watermark)
    perplexity_with_watermark = get_perplexity(text_with_watermark)
        
        
    return perplexity_without_watermark, perplexity_with_watermark

