# data_loader.py
import re
from datasets import load_dataset, Dataset

class TamilDataLoader:
    def __init__(self, config):
        self.config = config

    def collect_dataset(self):
        all_data = []
        print("Loading opus-100 (Real parallel data)...")
        try:
            dataset = load_dataset("Helsinki-NLP/opus-100", "en-ta", split="train", streaming=True)
            for item in dataset:
                if len(all_data) >= self.config.target_samples // 2:
                    break
                if 'translation' in item:
                    ta = item['translation'].get('ta', '').strip()
                    en = item['translation'].get('en', '').strip()
                    # Filter for length and Tamil characters
                    if (5 < len(ta) < 300 and 5 < len(en) < 300 and
                        re.search(r'[\u0B80-\u0BFF]', ta)):
                        all_data.append({'tamil': ta, 'english': en})
            print(f"Collected {len(all_data):,} real translation pairs from opus-100")
        except Exception as e:
            print(f" Opus-100 failed: {e}")

        print("Generating synthetic data (Common phrases)...")
        templates = [
            ("வணக்கம்!", "Hello!"),
            ("நன்றி!", "Thank you!"),
            ("எப்படி இருக்கிறீர்கள்?", "How are you?"),
            ("நான் நன்றாக இருக்கிறேன்", "I am fine"),
            ("உங்கள் பெயர் என்ன?", "What is your name?"),
            ("எனக்கு உதவி வேண்டும்", "I need help"),
            ("இன்று வானிலை எப்படி இருக்கிறது?", "How is the weather today?"),
        ]

        while len(all_data) < self.config.target_samples:
            ta, en = templates[len(all_data) % len(templates)]
            all_data.append({'tamil': ta, 'english': en})

        print(f"Total: {len(all_data):,} samples (Real + Synthetic)")
        return all_data

    def prepare_data(self):
        raw_data = self.collect_dataset()
        template = "Translate from Tamil to English.\n\nTamil: {tamil}\nEnglish: {english}"
        formatted_texts = [template.format(**item) for item in raw_data]

        dataset = Dataset.from_dict({"text": formatted_texts})
        
        # Keep original data for evaluation purposes
        original_dataset = Dataset.from_dict({
            'tamil': [item['tamil'] for item in raw_data],
            'english': [item['english'] for item in raw_data]
        })

        print(f"Formatted {len(dataset):,} samples")

        # Split for Training
        split = dataset.train_test_split(test_size=1 - self.config.train_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

        # Split for Testing (Raw format)
        original_split = original_dataset.train_test_split(test_size=0.1, seed=42)
        test_dataset_original = original_split["test"]

        print(f"📊 Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,} | Test: {len(test_dataset_original):,}")
        
        return train_dataset, eval_dataset, test_dataset_original

def load_tamil_data(config):
    loader = TamilDataLoader(config)
    return loader.prepare_data()