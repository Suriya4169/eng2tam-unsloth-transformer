from setuptools import setup, find_packages

setup(
    name='tamil-translation-unsloth',
    version='1.0.0',
    description='Fast Tamil-English translation model using Unsloth and Llama-3',
    author='Your Name',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'unsloth',
        'transformers',
        'peft',
        'accelerate',
        'bitsandbytes',
        'datasets',
        'sentence-transformers',
        'sacrebleu',
        'torch',
        'tqdm',
        'trl',
        'numpy',
        're',
        'random',
    ],
    python_requires='>=3.8',
)