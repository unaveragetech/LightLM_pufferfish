import os
import argparse
import torch
from transformers import AutoTokenizer
from model import Transformer
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.table import Table

# Import TrainerConfig to make it available for unpickling
from trainer import TrainerConfig

# Register TrainerConfig for safe loading
try:
    import torch.serialization
    torch.serialization.add_safe_globals([TrainerConfig])
except Exception as e:
    print(f"Warning: Could not register TrainerConfig for safe loading: {e}")

TOKENIZER_ID = "HuggingFaceTB/SmolLM-360M"

def load_model(checkpoint_dir):
    """Load the trained model and configuration"""
    console = Console()
    console.print("\n[bold cyan]Loading model...[/bold cyan]")

    console.print("\n[bold cyan]Loading model checkpoint...[/bold cyan]")
    try:
        # Load checkpoint with proper error handling
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model checkpoint found at {model_path}")

        # Show checkpoint info
        checkpoint_size = os.path.getsize(model_path) / (1024 * 1024)
        console.print(f"[yellow]Checkpoint size: {checkpoint_size:.2f}MB")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        map_location = device

        # Try different loading methods
        try:
            console.print(f"[green]Loading model on {device.upper()}")
            # First try with weights_only=False (most compatible)
            checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
            console.print("[green]Successfully loaded model with weights_only=False")
        except Exception as e1:
            console.print(f"[yellow]Error loading with weights_only=False: {str(e1)}")
            try:
                # Try with default settings
                checkpoint = torch.load(model_path, map_location=map_location)
                console.print("[green]Successfully loaded model with default settings")
            except Exception as e2:
                console.print(f"[red]Failed to load model: {str(e2)}")
                raise

        # Validate checkpoint contents
        console.print("\n[cyan]Validating checkpoint contents...")
        expected_keys = ['model_state_dict', 'config']
        missing_keys = [k for k in expected_keys if k not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        config = checkpoint['config']
        console.print(f"[green]✓ Found model configuration")
        console.print(f"[yellow]Model trained for {getattr(config, 'num_epochs', 'unknown')} epochs")

        # Initialize model with parameters from saved config
        console.print("\n[cyan]Initializing model architecture...")

        # Create a ModelConfig object
        from model import ModelConfig

        # Detect model architecture from the state dictionary
        state_dict = checkpoint['model_state_dict']

        # Detect number of layers by counting blocks
        num_layers = 0
        while f"blocks.{num_layers}.attention.wq.weight" in state_dict:
            num_layers += 1

        # If no layers detected, fall back to default
        if num_layers == 0:
            num_layers = 1
            console.print("[yellow]Warning: Could not detect number of layers, using default: 1")

        # Detect model dimensions from embedding size
        if "tokens_embedding.weight" in state_dict:
            embedding_shape = state_dict["tokens_embedding.weight"].shape
            vocab_size = embedding_shape[0]
            num_dims = embedding_shape[1]
        else:
            num_dims = 64  # Default fallback
            vocab_size = config.vocab_size
            console.print("[yellow]Warning: Could not detect model dimensions, using default: 64")

        # Detect number of heads from attention weights
        if "blocks.0.attention.wq.weight" in state_dict:
            # For typical models, num_heads is often num_dims / 64 or num_dims / 128
            # Try to infer from common ratios
            if num_dims % 64 == 0:
                head_dim = 64
            elif num_dims % 128 == 0:
                head_dim = 128
            else:
                # Fallback to a reasonable guess
                head_dim = max(32, num_dims // 16)

            num_heads = num_dims // head_dim

            # Detect KV heads
            if "blocks.0.attention.wk.weight" in state_dict:
                kv_dim = state_dict["blocks.0.attention.wk.weight"].shape[0]
                # Ensure we don't divide by zero
                if head_dim > 0:
                    num_kv_heads = max(1, kv_dim // head_dim)
                else:
                    num_kv_heads = max(1, num_heads // 4)
            else:
                num_kv_heads = max(1, num_heads // 4)
        else:
            num_heads = 4  # Default fallback
            num_kv_heads = 2  # Default fallback
            console.print("[yellow]Warning: Could not detect number of heads, using defaults")

        # Detect FFN hidden dimensions
        if "blocks.0.ffn.w1.weight" in state_dict:
            ffn_hidden_dims = state_dict["blocks.0.ffn.w1.weight"].shape[0]
        else:
            ffn_hidden_dims = num_dims * 4  # Default ratio
            console.print("[yellow]Warning: Could not detect FFN dimensions, using default ratio")

        # Create model config with detected parameters
        model_config = ModelConfig(
            vocab_size=vocab_size,
            num_dims=num_dims,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            ffn_hidden_dims=ffn_hidden_dims,
            context_len=config.max_seq_len,
            use_cache=True,
            use_flash=False,
            use_moe=False,
            moe_num_experts=4,
            moe_active_experts=4
        )

        console.print("[bold green]Detected model architecture:[/bold green]")
        console.print(f"→ vocab_size: {model_config.vocab_size}")
        console.print(f"→ num_dims: {model_config.num_dims}")
        console.print(f"→ num_heads: {model_config.num_heads}")
        console.print(f"→ num_kv_heads: {model_config.num_kv_heads}")
        console.print(f"→ num_layers: {model_config.num_layers}")
        console.print(f"→ ffn_hidden_dims: {model_config.ffn_hidden_dims}")
        console.print(f"→ context_len: {model_config.context_len}")

        model = Transformer(model_config)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        console.print(f"[green]✓[/green] Model loaded successfully from {model_path}")

        # Print model configuration
        console.print("\n[bold cyan]Model Configuration:[/bold cyan]")
        console.print(f"→ Vocab Size: {model_config.vocab_size}")
        console.print(f"→ Context Length: {model_config.context_len}")
        console.print(f"→ Model Dimensions: {model_config.num_dims}")
        console.print(f"→ Number of Layers: {model_config.num_layers}")
        console.print(f"→ Number of Heads: {model_config.num_heads}")
        console.print(f"→ FFN Hidden Dimensions: {model_config.ffn_hidden_dims}")

        return model, config

    except Exception as e:
        console.print(f"[red]Error loading model: {str(e)}")
        console.print("[yellow]Tip: Make sure you have completed training first")
        raise

def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
    """Generate text from the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Ensure input_ids is of type Long
    input_ids = input_ids.long()

    # Reset KV cache for each new generation
    for block in model.blocks:
        block.attention.cache_k = None
        block.attention.cache_v = None

    # Generate
    with torch.no_grad():
        # Note: repetition_penalty is not used in the model's generate method
        # but we accept it as a parameter for future compatibility
        # Always use cache=False for more reliable generation
        output_ids = model.generate(
            input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=int(top_p * 100),  # Convert top_p to approximate top_k
            use_cache=False
        )

    # Decode and return
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def list_available_checkpoints(base_dir="model_testing"):
    """List all available model checkpoints with their details"""
    console = Console()
    checkpoints = []

    # Scan for epoch directories
    for dirname in os.listdir(base_dir):
        if dirname.startswith("epoch_"):
            checkpoint_path = os.path.join(base_dir, dirname, "model.pt")
            if os.path.exists(checkpoint_path):
                size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                epoch_num = int(dirname.split("_")[1])
                checkpoints.append({
                    "path": checkpoint_path,
                    "epoch": epoch_num,
                    "size": size_mb,
                    "dir": dirname
                })

    # Sort by epoch number
    checkpoints.sort(key=lambda x: x["epoch"])

    # Display checkpoints in a table
    if checkpoints:
        table = Table(title="Available Model Checkpoints")
        table.add_column("Epoch", justify="right")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Path")

        for cp in checkpoints:
            table.add_row(
                str(cp["epoch"]),
                f"{cp['size']:.2f}",
                cp["path"]
            )

        console.print(table)
        return checkpoints
    else:
        console.print("[yellow]No checkpoints found")
        return []

def get_generation_settings():
    """Interactive prompt for generation settings"""
    console = Console()

    console.print("\n[cyan]Generation Settings[/cyan]")

    settings = {
        "max_tokens": IntPrompt.ask(
            "Maximum tokens to generate",
            default=100,
            show_default=True
        ),
        "temperature": FloatPrompt.ask(
            "Temperature (0.1-2.0, lower = more focused)",
            default=0.8,
            show_default=True
        ),
        "top_p": FloatPrompt.ask(
            "Top-p sampling (0.0-1.0)",
            default=0.9,
            show_default=True
        ),
        "repetition_penalty": FloatPrompt.ask(
            "Repetition penalty (1.0-2.0)",
            default=1.1,
            show_default=True
        )
    }

    return settings

def main():
    console = Console()
    console.print("\n[bold cyan]Enhanced Text Generation Interface[/bold cyan]")

    # List available checkpoints
    checkpoints = list_available_checkpoints()
    if not checkpoints:
        return

    # Let user select checkpoint
    epoch_nums = [cp["epoch"] for cp in checkpoints]
    selected_epoch = IntPrompt.ask(
        "Select epoch number",
        default=max(epoch_nums),
        show_choices=False
    )

    # Find selected checkpoint
    selected_cp = next((cp for cp in checkpoints if cp["epoch"] == selected_epoch), None)
    if not selected_cp:
        console.print(f"[red]Error: Epoch {selected_epoch} not found")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    try:
        model, config = load_model(os.path.dirname(selected_cp["path"]))
    except Exception as e:
        console.print(f"[red]Error loading model: {str(e)}")
        return

    # Get generation settings
    settings = get_generation_settings()

    # Interactive generation loop
    while True:
        prompt = Prompt.ask("\nEnter your prompt (or 'quit' to exit)")

        if prompt.lower() in ['quit', 'exit']:
            break

        if prompt.lower() == 'settings':
            settings = get_generation_settings()
            continue

        console.print("\n[cyan]Generating...[/cyan]")

        try:
            generated_text = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                repetition_penalty=settings["repetition_penalty"]
            )

            console.print("\n[green]Generated text:[/green]")
            console.print(generated_text)

            # Show stats
            console.print("\n[dim]Generation stats:[/dim]")
            console.print(f"[dim]→ Input tokens: {len(tokenizer.encode(prompt))}[/dim]")
            console.print(f"[dim]→ Generated tokens: {len(tokenizer.encode(generated_text)) - len(tokenizer.encode(prompt))}[/dim]")

        except Exception as e:
            console.print(f"[red]Error during generation: {str(e)}")

if __name__ == "__main__":
    main()
