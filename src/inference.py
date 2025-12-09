# inference.py
from unsloth import FastLanguageModel
import torch

class TamilTranslator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        FastLanguageModel.for_inference(self.model)

    def translate(self, text):
        prompt = f"Translate from Tamil to English.\n\nTamil: {text}\nEnglish:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=128, 
            temperature=0.1
        )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse output to extract just the english part
        parsed_result = result.split("English:")[-1].strip().split("\n")[0]
        return parsed_result if "English:" in result else result

def translate(model, tokenizer, text):
    translator = TamilTranslator(model, tokenizer)
    return translator.translate(text)