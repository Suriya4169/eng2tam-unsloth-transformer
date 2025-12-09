# train.py
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
import os

class TamilTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Starting training...")
        trainer.train()
        
        # Save results
        eval_results = trainer.evaluate()
        print("\nFinal Training Metrics:")
        print(f"  Eval Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

        print(f"Saving model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        return trainer

def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    trainer_wrapper = TamilTrainer(model, tokenizer, train_dataset, eval_dataset, config)
    return trainer_wrapper.train()