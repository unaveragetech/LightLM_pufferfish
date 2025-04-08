import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from model import Transformer, ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
tokenizer_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

# Create model configuration - make sure these match your training configuration
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    num_dims=512,
    num_heads=16,
    num_kv_heads=4,
    num_layers=32,
    ffn_hidden_dims=512 * 4,
    context_len=1536,
    use_cache=True,
    use_flash=True,
    use_moe=False,
    moe_num_experts=2,
    moe_active_experts=2,
)

# Initialize model
model = Transformer(config)

# Load your trained checkpoint
checkpoint_path = os.path.join("model_testing", "model.checkpoint.latest.pt")  # This is the default path from train.py
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print(f"Successfully loaded checkpoint from: {checkpoint_path}")
else:
    print(f"Checkpoint not found at: {checkpoint_path}")
    print("Available files in model_testing directory:")
    if os.path.exists("model_testing"):
        print("\n".join(os.listdir("model_testing")))
    else:
        print("model_testing directory not found")
    sys.exit(1)

model = model.to(device)
model.eval()

# Test the model with a prompt
prompt = "I am a language model,"
print(f"\nPrompt: {prompt}")
input_ids = tokenizer([prompt], return_tensors="pt")['input_ids'].to(device)
with torch.no_grad():
    idx = model.generate(input_ids, temperature=0.48, top_k=40, max_tokens=30)
print("\nGenerated text:")
print(tokenizer.batch_decode(idx)[0])
