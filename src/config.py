"""
Configuration module for Tamil-English translation model.
Contains all hyperparameters and training settings.
"""

class FastConfig:
    """Configuration class for model training and inference."""
    
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    max_seq_length = 512
    target_samples = 2000        
    train_split = 0.9            
    
    train_batch_size = 2         
    gradient_accumulation_steps = 8  
    learning_rate = 1.5e-4       
    max_steps = 600              
    eval_steps = 100             
    save_steps = 200             
    logging_steps = 25           
    warmup_steps = 100           
    
    # LoRA configuration
    lora_r = 64                  
    lora_alpha = 32              
    lora_dropout = 0.1           
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    test_sample_size = 100      
    
    output_dir = "./tamil-translation-fast"
    model_save_dir = "./tamil-translation-model"
    results_file = "./evaluation_results.txt"
    
    def __repr__(self):
        return (f"FastConfig(samples={self.target_samples}, "
                f"steps={self.max_steps}, batch_size={self.train_batch_size})")