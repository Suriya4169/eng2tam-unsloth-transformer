"""
Tamil-English Translation Model Package
Fine-tuned LLaMA 3.1 8B for Tamil to English translation
"""

__version__ = "1.0.0"
__author__ = "Team B5"

from .config import FastConfig
from .model import TamilTranslationModel, load_model
from .data_loader import TamilDataLoader, load_tamil_data
from .train import TamilTrainer, train_model
from .inference import TamilTranslator, translate
from .evaluate import TamilEvaluator, evaluate_model

__all__ = [
    'FastConfig',
    'TamilTranslationModel',
    'TamilDataLoader',
    'TamilTrainer',
    'TamilTranslator',
    'TamilEvaluator',
    'load_model',
    'load_tamil_data',
    'train_model',
    'translate',
    'evaluate_model',
]