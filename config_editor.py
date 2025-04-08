"""
Configuration Table Editor for LightLM

This module provides an interactive configuration editor using Rich tables.
"""

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.panel import Panel
from rich import box

console = Console()

def validate_model_config(model_config):
    """
    Validate model configuration parameters and their relationships.
    Returns a list of warnings/errors.
    """
    warnings = []

    # Check num_dims is divisible by num_heads
    if model_config.num_dims % model_config.num_heads != 0:
        warnings.append(f"[bold red]ERROR:[/bold red] num_dims ({model_config.num_dims}) must be divisible by num_heads ({model_config.num_heads})")

    # Check head_dim is reasonable
    head_dim = model_config.num_dims // model_config.num_heads
    if head_dim < 32 or head_dim > 128:
        warnings.append(f"[bold yellow]WARNING:[/bold yellow] Unusual head dimension: {head_dim}. Recommended range is 32-128.")

    # Check num_kv_heads divides num_heads evenly
    if model_config.num_heads % model_config.num_kv_heads != 0:
        warnings.append(f"[bold red]ERROR:[/bold red] num_heads ({model_config.num_heads}) must be divisible by num_kv_heads ({model_config.num_kv_heads})")

    # Check MoE parameters
    if model_config.use_moe:
        if model_config.moe_active_experts > model_config.moe_num_experts:
            warnings.append(f"[bold red]ERROR:[/bold red] moe_active_experts ({model_config.moe_active_experts}) cannot be greater than moe_num_experts ({model_config.moe_num_experts})")

    # Check if model is too large for typical consumer hardware
    model_size_estimate = model_config.num_layers * model_config.num_dims * model_config.num_dims * 4 / (1024**3)  # Rough estimate in GB
    if model_size_estimate > 8:
        warnings.append(f"[bold yellow]WARNING:[/bold yellow] Estimated model size ({model_size_estimate:.1f} GB) may be too large for consumer GPUs")

    # Check if num_layers is very high
    if model_config.num_layers > 40:
        warnings.append(f"[bold yellow]WARNING:[/bold yellow] Large number of layers ({model_config.num_layers}) may cause memory issues")

    return warnings

def validate_train_config(train_config):
    """
    Validate training configuration parameters.
    Returns a list of warnings/errors.
    """
    warnings = []

    # Check batch size
    if train_config.batch_size > 32:
        warnings.append(f"[bold yellow]WARNING:[/bold yellow] Large batch size ({train_config.batch_size}) may cause memory issues")

    # Check learning rate
    if train_config.learning_rate > 0.01:
        warnings.append(f"[bold yellow]WARNING:[/bold yellow] Learning rate ({train_config.learning_rate}) is unusually high")

    # Check sequence length
    if train_config.max_seq_len > 2048:
        warnings.append(f"[bold yellow]WARNING:[/bold yellow] Long sequence length ({train_config.max_seq_len}) may cause memory issues")

    return warnings

def get_parameter_description(param_name):
    """
    Get enhanced description for parameters with valid ranges and constraints.
    """
    descriptions = {
        # Model parameters
        "vocab_size": "Vocabulary size for tokenizer (typically 32000-50000)",
        "num_dims": "Model dimension size (d_model). Must be divisible by num_heads. Common values: 512, 768, 1024",
        "num_heads": "Number of attention heads. Must divide num_dims evenly. Common values: 8, 12, 16",
        "num_kv_heads": "Number of key/value heads. Must divide num_heads evenly. Common values: 1, 2, 4, 8",
        "num_layers": "Number of transformer layers. Common values: 12, 24, 32",
        "ffn_hidden_dims": "Hidden dimension for FFN. Typically 2-4x num_dims",
        "context_len": "Maximum context length. Common values: 512, 1024, 2048, 4096",

        # Training parameters
        "batch_size": "Batch size for training. Adjust based on available GPU memory. Common values: 4, 8, 16, 32",
        "accumulation_steps": "Gradient accumulation steps. Increases effective batch size. Common values: 1, 2, 4, 8",
        "learning_rate": "Learning rate. Common values: 0.0001-0.001",
        "max_seq_len": "Maximum sequence length. Should be <= context_len. Common values: 512, 1024, 2048",
        "gradient_checkpointing": "Use gradient checkpointing to save memory at the cost of speed",
        "use_dtype": "Data type for training. Options: float32, float16, bfloat16",
    }

    return descriptions.get(param_name, "")

def edit_config_table(model_config, train_config):
    """
    Display and edit configuration parameters using Rich tables.

    Args:
        model_config: The model configuration object
        train_config: The training configuration object

    Returns:
        Tuple of (model_config, train_config) with any updates applied
    """
    while True:
        console.clear()
        console.print(Panel("[bold cyan]Configuration Editor[/bold cyan]",
                           subtitle="Use arrow keys to navigate, Enter to edit, ESC to exit"))

        # Check for configuration warnings/errors
        model_warnings = validate_model_config(model_config)
        train_warnings = validate_train_config(train_config)

        # Display warnings if any
        if model_warnings or train_warnings:
            console.print(Panel("\n".join(["[bold red]Configuration Warnings/Errors:[/bold red]"] +
                                        model_warnings + train_warnings),
                               title="[bold red]⚠️ Warning[/bold red]",
                               border_style="red"))

        # Create model configuration table
        model_table = Table(title="[bold]Model Configuration[/bold]",
                           box=box.ROUNDED,
                           show_header=True,
                           header_style="bold magenta",
                           expand=True)

        model_table.add_column("Parameter", style="cyan")
        model_table.add_column("Value", style="green")
        model_table.add_column("Description", style="yellow")

        # Add model configuration parameters with enhanced descriptions
        model_params = [
            ("vocab_size", model_config.vocab_size, get_parameter_description("vocab_size") or "Vocabulary size for tokenizer"),
            ("num_dims", model_config.num_dims, get_parameter_description("num_dims") or "Model dimension size (d_model)"),
            ("num_heads", model_config.num_heads, get_parameter_description("num_heads") or "Number of attention heads"),
            ("num_kv_heads", model_config.num_kv_heads, get_parameter_description("num_kv_heads") or "Number of key/value heads"),
            ("num_layers", model_config.num_layers, get_parameter_description("num_layers") or "Number of transformer layers"),
            ("ffn_hidden_dims", model_config.ffn_hidden_dims, get_parameter_description("ffn_hidden_dims") or "Hidden dimension for FFN"),
            ("context_len", model_config.context_len, get_parameter_description("context_len") or "Maximum context length"),
            ("use_cache", model_config.use_cache, "Enable KV-caching"),
            ("use_flash", model_config.use_flash, "Use Flash Attention"),
            ("use_moe", model_config.use_moe, "Enable mixture-of-experts"),
            ("moe_num_experts", model_config.moe_num_experts, "Total number of experts"),
            ("moe_active_experts", model_config.moe_active_experts, "Number of experts per token (top_k)"),
            ("moe_eps", model_config.moe_eps, "Epsilon for router stability"),
            ("moe_aux_loss_coef", model_config.moe_aux_loss_coef, "Coefficient for auxiliary loss"),
            ("moe_shared_experts", model_config.moe_shared_experts, "Number of shared experts"),
            ("use_lossfreebalance", model_config.use_lossfreebalance, "Use loss-free balancing strategy"),
            ("rmsnorm_eps", model_config.rmsnorm_eps, "Epsilon for RMSNorm"),
            ("rope_theta", model_config.rope_theta, "Theta for RoPE positional encoding"),
            ("activation_fn", model_config.activation_fn, "Activation function (silu, gelu, relu)"),
            ("norm_type", model_config.norm_type, "Normalization type (rmsnorm, layernorm)")
        ]

        for param, value, desc in model_params:
            model_table.add_row(param, str(value), desc)

        # Create training configuration table
        train_table = Table(title="[bold]Training Configuration[/bold]",
                           box=box.ROUNDED,
                           show_header=True,
                           header_style="bold magenta",
                           expand=True)

        train_table.add_column("Parameter", style="cyan")
        train_table.add_column("Value", style="green")
        train_table.add_column("Description", style="yellow")

        # Add training configuration parameters with enhanced descriptions
        train_params = [
            ("num_epochs", train_config.num_epochs, get_parameter_description("num_epochs") or "Number of training epochs"),
            ("use_subset", train_config.use_subset, "Use subset of dataset"),
            ("target_samples", train_config.target_samples, "Number of samples when using subset"),
            ("use_epoch_checkpoints", train_config.use_epoch_checkpoints, "Use checkpoint from previous epoch for each new epoch"),
            ("batch_size", train_config.batch_size, get_parameter_description("batch_size") or "Batch size for training"),
            ("accumulation_steps", train_config.accumulation_steps, get_parameter_description("accumulation_steps") or "Gradient accumulation steps"),
            ("max_seq_len", train_config.max_seq_len, get_parameter_description("max_seq_len") or "Maximum sequence length"),
            ("learning_rate", train_config.learning_rate, get_parameter_description("learning_rate") or "Learning rate"),
            ("weight_decay", train_config.weight_decay, "Weight decay"),
            ("warmup_ratio", train_config.warmup_ratio, "Warmup ratio"),
            ("use_lora", train_config.use_lora, "Use LoRA fine-tuning"),
            ("lora_rank", train_config.lora_rank, "LoRA rank"),
            ("lora_alpha", train_config.lora_alpha, "LoRA alpha"),
            ("use_compile", train_config.use_compile, "Use PyTorch 2.0 compile"),
            ("use_dtype", train_config.use_dtype, "Data type for training"),
            ("use_flash", train_config.use_flash, "Use Flash Attention"),
            ("gradient_checkpointing", train_config.gradient_checkpointing, "Use gradient checkpointing"),
            ("gpu_memory_utilization", train_config.gpu_memory_utilization, "GPU memory utilization"),
            ("enable_memory_tracking", train_config.enable_memory_tracking, "Enable memory tracking"),
            ("empty_cache_freq", train_config.empty_cache_freq, "Frequency of cache clearing"),
            ("auto_adjust_batch", train_config.auto_adjust_batch, "Auto-adjust batch size"),
            ("min_batch_size", train_config.min_batch_size, "Minimum batch size")
        ]

        for param, value, desc in train_params:
            train_table.add_row(param, str(value), desc)

        # Print tables
        console.print(model_table)
        console.print("\n")
        console.print(train_table)

        # Get user selection
        console.print("\n[bold]Select configuration to edit:[/bold]")
        console.print("1. Model Configuration")
        console.print("2. Training Configuration")
        console.print("3. Return to main menu")

        choice = Prompt.ask("Select option", choices=["1", "2", "3"], default="3")

        if choice == "1":
            # Edit model configuration
            param_name = Prompt.ask("Enter parameter name to edit",
                                   choices=[p[0] for p in model_params])

            # Get current value and type
            current_value = getattr(model_config, param_name)
            value_type = type(current_value)

            # Show parameter description and constraints
            desc = get_parameter_description(param_name)
            if desc:
                console.print(f"\n[yellow]Description:[/yellow] {desc}")

            # Show specific constraints based on parameter
            if param_name == "num_dims":
                console.print("[yellow]Constraint:[/yellow] Must be divisible by num_heads")
                console.print(f"[yellow]Current num_heads:[/yellow] {model_config.num_heads}")
                console.print(f"[yellow]Valid values:[/yellow] {', '.join(str(model_config.num_heads * i) for i in range(1, 9))}")
            elif param_name == "num_heads":
                console.print("[yellow]Constraint:[/yellow] Must divide num_dims evenly")
                console.print(f"[yellow]Current num_dims:[/yellow] {model_config.num_dims}")
                console.print(f"[yellow]Valid values:[/yellow] {', '.join(str(model_config.num_dims // i) for i in [1, 2, 4, 8, 16, 32, 64] if model_config.num_dims % i == 0)}")
            elif param_name == "num_kv_heads":
                console.print("[yellow]Constraint:[/yellow] Must divide num_heads evenly")
                console.print(f"[yellow]Current num_heads:[/yellow] {model_config.num_heads}")
                console.print(f"[yellow]Valid values:[/yellow] {', '.join(str(model_config.num_heads // i) for i in [1, 2, 4, 8, 16] if model_config.num_heads % i == 0)}")
            elif param_name == "moe_active_experts" and model_config.use_moe:
                console.print("[yellow]Constraint:[/yellow] Must be <= moe_num_experts")
                console.print(f"[yellow]Current moe_num_experts:[/yellow] {model_config.moe_num_experts}")
                console.print(f"[yellow]Valid values:[/yellow] 1 to {model_config.moe_num_experts}")

            # Get new value based on type
            if value_type == bool:
                new_value = Confirm.ask(f"New value for {param_name}", default=current_value)
            elif value_type == int:
                new_value = IntPrompt.ask(f"New value for {param_name}", default=current_value)
            elif value_type == float:
                new_value = FloatPrompt.ask(f"New value for {param_name}", default=current_value)
            else:
                new_value = Prompt.ask(f"New value for {param_name}", default=str(current_value))
                # Try to convert string to original type
                if value_type != str:
                    try:
                        new_value = value_type(new_value)
                    except ValueError:
                        console.print(f"[red]Error: Could not convert to {value_type.__name__}[/red]")
                        continue

            # Validate the new value
            if param_name == "num_dims" and new_value % model_config.num_heads != 0:
                console.print(f"[red]Error: num_dims ({new_value}) must be divisible by num_heads ({model_config.num_heads})[/red]")
                continue
            elif param_name == "num_heads" and model_config.num_dims % new_value != 0:
                console.print(f"[red]Error: num_heads ({new_value}) must divide num_dims ({model_config.num_dims}) evenly[/red]")
                continue
            elif param_name == "num_kv_heads" and model_config.num_heads % new_value != 0:
                console.print(f"[red]Error: num_kv_heads ({new_value}) must divide num_heads ({model_config.num_heads}) evenly[/red]")
                continue
            elif param_name == "moe_active_experts" and model_config.use_moe and new_value > model_config.moe_num_experts:
                console.print(f"[red]Error: moe_active_experts ({new_value}) cannot be greater than moe_num_experts ({model_config.moe_num_experts})[/red]")
                continue

            # Update the configuration
            setattr(model_config, param_name, new_value)
            console.print(f"[green]Updated {param_name} to {new_value}[/green]")

            # Recalculate and show derived values
            if param_name in ["num_dims", "num_heads"]:
                head_dim = model_config.num_dims // model_config.num_heads
                console.print(f"[green]Head dimension is now: {head_dim}[/green]")

        elif choice == "2":
            # Edit training configuration
            param_name = Prompt.ask("Enter parameter name to edit",
                                   choices=[p[0] for p in train_params])

            # Get current value and type
            current_value = getattr(train_config, param_name)
            value_type = type(current_value)

            # Show parameter description and constraints
            desc = get_parameter_description(param_name)
            if desc:
                console.print(f"\n[yellow]Description:[/yellow] {desc}")

            # Show specific constraints based on parameter
            if param_name == "batch_size":
                console.print("[yellow]Warning:[/yellow] Large batch sizes may cause memory issues")
                console.print("[yellow]Recommended:[/yellow] 4-16 for consumer GPUs")
            elif param_name == "max_seq_len":
                console.print("[yellow]Constraint:[/yellow] Should be <= model's context_len")
                console.print(f"[yellow]Current context_len:[/yellow] {model_config.context_len}")
                console.print("[yellow]Warning:[/yellow] Large sequence lengths may cause memory issues")
            elif param_name == "learning_rate":
                console.print("[yellow]Recommended:[/yellow] 0.0001-0.001 for most training")
            elif param_name == "target_samples" and train_config.use_subset:
                console.print("[yellow]Warning:[/yellow] Very small values may result in poor training")
                console.print("[yellow]Minimum recommended:[/yellow] 100 samples")

            # Get new value based on type
            if value_type == bool:
                new_value = Confirm.ask(f"New value for {param_name}", default=current_value)
            elif value_type == int:
                new_value = IntPrompt.ask(f"New value for {param_name}", default=current_value)
            elif value_type == float:
                new_value = FloatPrompt.ask(f"New value for {param_name}", default=current_value)
            else:
                new_value = Prompt.ask(f"New value for {param_name}", default=str(current_value))
                # Try to convert string to original type
                if value_type != str:
                    try:
                        new_value = value_type(new_value)
                    except ValueError:
                        console.print(f"[red]Error: Could not convert to {value_type.__name__}[/red]")
                        continue

            # Validate the new value
            if param_name == "batch_size" and new_value <= 0:
                console.print(f"[red]Error: batch_size must be positive[/red]")
                continue
            elif param_name == "max_seq_len" and new_value > model_config.context_len:
                console.print(f"[red]Warning: max_seq_len ({new_value}) is greater than model's context_len ({model_config.context_len})[/red]")
                if not Confirm.ask("Continue anyway?", default=False):
                    continue
            elif param_name == "learning_rate" and (new_value <= 0 or new_value > 0.1):
                console.print(f"[red]Warning: Unusual learning_rate value: {new_value}[/red]")
                if not Confirm.ask("Continue anyway?", default=False):
                    continue
            elif param_name == "target_samples" and train_config.use_subset and new_value < 20:
                console.print(f"[red]Warning: Very small target_samples ({new_value}) may result in poor training[/red]")
                if not Confirm.ask("Continue anyway?", default=False):
                    continue

            # Update the configuration
            setattr(train_config, param_name, new_value)
            console.print(f"[green]Updated {param_name} to {new_value}[/green]")

            # Show additional information for certain parameters
            if param_name == "batch_size":
                effective_batch = new_value * train_config.accumulation_steps
                console.print(f"[green]Effective batch size is now: {effective_batch}[/green]")

        elif choice == "3":
            # Return to main menu
            break

        # Pause to show the update message
        Prompt.ask("Press Enter to continue")

    return model_config, train_config
