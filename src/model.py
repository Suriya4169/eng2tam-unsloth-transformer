# model.py
from unsloth import FastLanguageModel

class TamilTranslationModel:
    @staticmethod
    def from_pretrained(config):
        print(f"Loading model: {config.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        return model, tokenizer

def load_model(config):
    return TamilTranslationModel.from_pretrained(config)