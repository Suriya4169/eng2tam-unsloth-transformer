# evaluate.py
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics import BLEU
import numpy as np
import random
from tqdm import tqdm
from .inference import TamilTranslator

class TamilEvaluator:
    def __init__(self, model, tokenizer, test_dataset, config):
        self.translator = TamilTranslator(model, tokenizer)
        self.test_dataset = test_dataset
        self.config = config
        self.sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.bleu_metric = BLEU()

    def compute_similarity(self, text1, text2):
        emb1 = self.sentence_model.encode(text1, convert_to_tensor=True).cpu().numpy()
        emb2 = self.sentence_model.encode(text2, convert_to_tensor=True).cpu().numpy()
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return float(cos_sim)

    def evaluate(self):
        print(f"\n🔍 Evaluating {self.config.test_sample_size} test samples...")
        
        # Select random samples if dataset is larger than sample size
        if len(self.test_dataset) > self.config.test_sample_size:
            indices = random.sample(range(len(self.test_dataset)), self.config.test_sample_size)
            dataset_subset = [self.test_dataset[i] for i in indices]
        else:
            dataset_subset = self.test_dataset

        similarities = []
        high_quality = 0
        semantic_matches = 0
        exact_matches = 0

        predictions = []
        references = []

        for example in tqdm(dataset_subset, desc="Evaluating"):
            tamil_text = example["tamil"]
            reference = example["english"]

            predicted = self.translator.translate(tamil_text)

            sim = self.compute_similarity(predicted, reference)
            similarities.append(sim)

            if sim > 0.7: high_quality += 1
            if sim > 0.5: semantic_matches += 1
            if predicted.lower().strip() == reference.lower().strip():
                exact_matches += 1

            predictions.append(predicted)
            references.append([reference])

        bleu_score = self.bleu_metric.corpus_score(predictions, references)
        
        # Calculate Statistics
        stats = {
            "bleu": bleu_score.score,
            "avg_sim": np.mean(similarities),
            "median_sim": np.median(similarities),
            "high_quality_pct": (high_quality / len(similarities)) * 100,
            "semantic_pct": (semantic_matches / len(similarities)) * 100,
            "exact_match_pct": (exact_matches / len(similarities)) * 100
        }

        self._print_results(stats, bleu_score)
        return stats

    def _print_results(self, stats, bleu_score):
        print(f"\n{'='*70}")
        print("COMPREHENSIVE EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"BLEU Score: {stats['bleu']:.2f}")
        print(f"BERT Sim Average: {stats['avg_sim']:.4f}")
        print(f"Excellent (>0.7): {stats['high_quality_pct']:.1f}%")
        print(f"Good (>0.5): {stats['semantic_pct']:.1f}%")
        print(f"Exact Matches: {stats['exact_match_pct']:.1f}%")
        print(f"{'='*70}")

def evaluate_model(model, tokenizer, test_dataset, config):
    evaluator = TamilEvaluator(model, tokenizer, test_dataset, config)
    return evaluator.evaluate()