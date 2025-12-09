# Tamil-English Translation Model (Unsloth)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Required-important)

This project trains a fast and efficient Tamil-to-English translation model using Unsloth and Meta-Llama-3.1-8B-bnb-4bit. It leverages real parallel data and synthetic phrase pairs for robust translation quality.

## Features
- Fast training with LoRA and 4-bit quantization
- BLEU and BERT-based evaluation metrics
- Synthetic and real data mixing for robust generalization
- Model and tokenizer saving for easy inference

## Quickstart
See [QUICKSTART.md](QUICKSTART.md) for fast setup instructions.

## Repository Structure
See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for a full breakdown.

## Requirements
- Python 3.8+
- CUDA-enabled GPU recommended
- See `requirements.txt` for dependencies

## License
MIT License. See [LICENSE](LICENSE).
