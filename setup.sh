#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch and related packages
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Install other required packages
pip install transformers \
    datasets \
    huggingface_hub \
    tiktoken \
    numpy \
    matplotlib \
    tqdm

echo "Setup completed! You can activate the virtual environment using:"
echo "source venv/bin/activate"