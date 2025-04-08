import torch.nn.functional as F
from queue import Queue
import threading
import time
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from datasets import load_dataset
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from datatrove.utils.dataset import DatatroveFolderDataset

TOKENIZER_ID = "HuggingFaceTB/SmolLM-360M"  # Default tokenizer ID

# Add rich console import with fallback at the top of trainer.py
try:
    from rich.console import Console
    from rich.panel import Panel
    default_console = Console()
except ImportError:
    default_console = print
    class Panel:
        def __init__(self, text, title=None):
            self.renderable = f"\n{title if title else ''}\n{text}\n"

@dataclass
class TrainerConfig:
    vocab_size: int
    num_epochs: int

    use_ddp: bool
    use_moe: bool                   # enable mixture-of-experts
    use_lossfreebalance: bool       # use Auxiliary-loss-free load balancing strategy for mixture-of-experts from DeepSeek
    clean_cuda_cache: bool = True   # Helps prevent OOM errors during eval on large models
    use_compile: bool = True        # use torch.compile()
    use_dtype: str = "bfloat16"
    use_flash: bool = False         # use Flash Attention when available

    seed: int = 1998
    max_seq_len: int = 1024         # maximum context length for batch
    batch_size: int = 1             # number of batches
    accumulation_steps: int = 1

    # Optimizer parameters
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    learning_rate: float = 1e-3
    betas: Tuple[float, float] = (0.90, 0.95)
    update_rate: float = 1e-5       # update_rate of biases for loss-free balancing

    val_ratio: int = 0.005
    steps_for_eval: int = 20        # number of steps for evaluation
    eval_interval: int = 50
    log_interval: int = 1          # number of steps between logging updates

    checkpoints_frequency: int = 500
    path_to_checkpoints: str = "./model_testing"        # path to directory to save checkpoints
    tokenized_dataset_path: str = ""                    # path to directory with tokenized dataset
    eval_log_file: str = "logs/eval.txt"               # path to file to write eval results

    # LoRA parameters
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Threading parameters
    num_workers: int = 4
    queue_size: int = 8

    # Dynamic memory management
    gpu_memory_utilization: float = field(default_factory=lambda: None)  # Will be set dynamically
    enable_memory_tracking: bool = True
    expandable_segments: bool = True
    gradient_checkpointing: bool = False
    empty_cache_freq: int = 100
    auto_adjust_batch: bool = True
    min_batch_size: int = 1

    # Dataset parameters
    use_subset: bool = False
    target_samples: Optional[int] = None
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_text_field: Optional[str] = None

    # Checkpoint parameters
    use_epoch_checkpoints: bool = False  # Whether to use the checkpoint from the previous epoch

    def initialize_memory_settings(self, hardware_manager):
        """Initialize memory settings based on hardware capabilities"""
        if hardware_manager.gpu_available:
            # Set memory utilization based on safe limit
            self.gpu_memory_utilization = hardware_manager.gpu_memory / torch.cuda.get_device_properties(0).total_memory

            # For ultra-fast testing configurations
            if self.max_seq_len <= 128 and self.num_epochs == 1:
                self.gradient_checkpointing = False
                self.empty_cache_freq = 1000
                self.enable_memory_tracking = False
                self.gpu_memory_utilization = 0.95  # Can be more aggressive for short runs
            # Regular memory optimization logic continues...
            if hardware_manager.gpu_memory < 8 * (1024**3):  # Less than 8GB
                self.gradient_checkpointing = True
                self.empty_cache_freq = 50

            # Set initial batch size based on hardware
            if self.auto_adjust_batch:
                self.batch_size = hardware_manager.get_max_batch_size(self)


class CustomDataLoader:
    def __init__(self, config, dataset, tokenizer=None):
        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = config.batch_size

        # Calculate dataset sizes
        self.total_size = len(dataset)
        self.val_size = int(self.total_size * config.val_ratio)
        self.train_size = self.total_size - self.val_size

        # Create indices
        self.indices = torch.randperm(self.train_size).tolist()
        self.current_idx = 0

        print(f"Dataset size: total={self.total_size}, train={self.train_size}, val={self.val_size}")

    def get_batch(self, split='train'):
        # Initialize empty lists for batch
        input_ids = []
        labels = []

        # Get batch_size samples
        for _ in range(self.batch_size):
            # Get sample from dataset
            sample = self.dataset[self.current_idx]

            # Tokenize if needed
            if self.tokenizer and isinstance(sample, str):
                tokens = self.tokenizer(
                    sample,
                    truncation=True,
                    max_length=self.config.max_seq_len,
                    padding='max_length',
                    return_tensors="pt"
                )
                sample_input_ids = tokens["input_ids"][0]
            else:
                sample_input_ids = torch.tensor(sample['input_ids'])
                # Ensure consistent length
                if len(sample_input_ids) > self.config.max_seq_len:
                    sample_input_ids = sample_input_ids[:self.config.max_seq_len]

            input_ids.append(sample_input_ids)
            labels.append(sample_input_ids.clone())

            # Update index
            self.current_idx = (self.current_idx + 1) % self.train_size

        # Stack tensors instead of padding manually
        input_ids = torch.stack([x for x in input_ids])
        labels = torch.stack([x for x in labels])

        return {
            'input_ids': input_ids,
            'labels': labels
        }

    def __len__(self):
        return self.train_size // self.batch_size


class ThreadedDataLoader:
    def __init__(self, config, subset_size=None, queue_size=8, dataset=None):
        self.config = config
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

        # Check if subset_size is provided directly or via config
        if subset_size is None and hasattr(config, 'use_subset') and config.use_subset and config.target_samples is not None:
            self.subset_size = config.target_samples
            print(f"Using subset size from config: {self.subset_size} samples")
        else:
            self.subset_size = subset_size
            if self.subset_size is not None:
                print(f"Using provided subset size: {self.subset_size} samples")

        self.dataset_size = 0
        self.steps_per_epoch = 0
        self.current_step = 0

        # Initialize tokenizer automatically
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Initialized tokenizer: {TOKENIZER_ID}")
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            raise e

        # Load dataset
        try:
            # Use provided dataset if available, otherwise load default
            if dataset is not None:
                full_dataset = dataset
                print(f"Using provided dataset with {len(full_dataset)} samples")
            else:
                # Load default dataset if none provided
                full_dataset = load_dataset(
                    "wikitext",
                    "wikitext-2-raw-v1",
                    split="train",
                    cache_dir=self.config.tokenized_dataset_path
                )
                print(f"Using default wikitext dataset with {len(full_dataset)} samples")

            # Apply subsetting if specified
            if self.subset_size is not None:
                # Explicitly select only the first subset_size samples
                if self.subset_size < len(full_dataset):
                    self.dataset = full_dataset.select(range(self.subset_size))
                    print(f"Using subset mode: {self.subset_size} samples out of {len(full_dataset)}")
                else:
                    self.dataset = full_dataset
                    print(f"Warning: Requested subset size {self.subset_size} is larger than dataset size {len(full_dataset)}")
                    print(f"Using full dataset with {len(full_dataset)} samples")
                    self.subset_size = len(full_dataset)  # Adjust subset size to match dataset size
            else:
                self.dataset = full_dataset
                print(f"Using full dataset with {len(full_dataset)} samples")

            # Calculate dataset size and steps per epoch based on the actual dataset size
            self.dataset_size = len(self.dataset)

            # If subset mode is active, handle differently based on subset size
            if self.subset_size is not None:
                # Check if this is ultra-fast mode (very small subset)
                if self.subset_size <= 20:  # Ultra-fast mode with minimal samples
                    # Force small number of steps for ultra-fast testing
                    self.steps_per_epoch = max(1, min(5, self.subset_size // config.batch_size))
                    if self.subset_size % config.batch_size != 0 and self.steps_per_epoch < 5:
                        self.steps_per_epoch += 1
                    print(f"Ultra-fast mode active: Using only {self.steps_per_epoch * config.batch_size} samples with {self.steps_per_epoch} steps per epoch")
                else:
                    # Regular subset mode - use the actual subset size
                    self.steps_per_epoch = self.subset_size // config.batch_size
                    if self.subset_size % config.batch_size != 0:
                        self.steps_per_epoch += 1
                    print(f"Subset mode active: Using {self.subset_size} samples with {self.steps_per_epoch} steps per epoch")
            else:
                # Normal mode - use the full dataset
                self.steps_per_epoch = self.dataset_size // config.batch_size
                if self.dataset_size % config.batch_size != 0:
                    self.steps_per_epoch += 1
                print(f"Full dataset mode: {self.dataset_size} samples with {self.steps_per_epoch} steps per epoch")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e

        # Start loader thread
        self.loader_thread = threading.Thread(target=self._loader_worker, daemon=True)
        self.loader_thread.start()

    def __iter__(self):
        """Make the class iterable"""
        self.current_step = 0
        return self

    def __next__(self):
        """Get next batch for iteration"""
        if self.current_step >= self.steps_per_epoch:
            self.current_step = 0
            raise StopIteration

        self.current_step += 1
        return self.next_batch()

    def __len__(self):
        """Return the number of batches per epoch"""
        return self.steps_per_epoch

    def _loader_worker(self):
        while not self.stop_event.is_set():
            try:
                batch = self._load_next_batch()
                self.queue.put(batch)
            except Exception as e:
                print(f"Loader worker error: {e}")
                break

    def _load_next_batch(self):
        """Load and process the next batch of data"""
        # Get random indices for this batch, ensuring we stay within the dataset size
        # This is especially important for subset mode
        indices = torch.randint(0, self.dataset_size, (self.config.batch_size,))

        # Get samples - use the text field from config if available, otherwise default to 'text'
        text_field = getattr(self.config, 'dataset_text_field', 'text') or 'text'

        # Try to get samples using the specified text field
        try:
            samples = [self.dataset[idx.item()][text_field] for idx in indices]
        except KeyError as e:
            # If the specified field doesn't exist, try to find a suitable text field
            sample_item = self.dataset[0]
            text_fields = [field for field in sample_item.keys()
                          if isinstance(sample_item[field], str) and len(sample_item[field]) > 10]

            if text_fields:
                text_field = text_fields[0]
                print(f"Field '{e}' not found. Using '{text_field}' field instead.")
                # Update the config with the correct field
                if hasattr(self.config, 'dataset_text_field'):
                    self.config.dataset_text_field = text_field
                samples = [self.dataset[idx.item()][text_field] for idx in indices]
            else:
                print(f"Error: No suitable text field found in dataset. Available fields: {list(sample_item.keys())}")
                raise KeyError(f"No suitable text field found in dataset")

        # Tokenize
        encodings = self.tokenizer(
            samples,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt"
        )

        # Prepare input_ids and labels
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels = input_ids.clone()

        # Move to device if specified
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def next_batch(self):
        """Get the next batch from the queue"""
        return self.queue.get()

    def stop(self):
        """Stop the loader thread"""
        self.stop_event.set()
        self.loader_thread.join()


def track_memory_usage():
    """Track GPU memory usage and high points"""
    if not hasattr(track_memory_usage, 'max_allocated'):
        track_memory_usage.max_allocated = 0
        track_memory_usage.max_reserved = 0

    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
        free = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2 - allocated

        # Update max values
        track_memory_usage.max_allocated = max(track_memory_usage.max_allocated, allocated)
        track_memory_usage.max_reserved = max(track_memory_usage.max_reserved, reserved)

        return allocated, reserved, free, track_memory_usage.max_allocated
    return 0, 0, 0, 0

def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
        free = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2 - allocated
        return allocated, reserved, free
    return 0, 0, 0

def ensure_tensor_dtype(tensor, device, dtype=torch.long):
    """Ensure tensor is of correct type and on correct device"""
    if tensor is not None:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=dtype, device=device)
        else:
            tensor = tensor.to(device, dtype=dtype)
    return tensor

class DeviceManager:
    """Context manager for handling device transitions"""
    def __init__(self, model, optimizer, target_device, trainer):
        self.model = model
        self.optimizer = optimizer
        self.target_device = target_device
        self.original_device = next(model.parameters()).device if model is not None else None
        self.trainer = trainer
        self.state = None
        self.optimizer_state = None

    def __enter__(self):
        if self.original_device != self.target_device:
            self.trainer.console.print(f"[yellow]Moving model and optimizer from {self.original_device} to {self.target_device}...")
            if self.target_device == torch.device('cpu'):
                # Save states before moving to CPU
                self.state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                self.optimizer_state = self.optimizer.state_dict()

                # Move optimizer state to CPU if optimizer exists
                if self.optimizer:
                    for group in self.optimizer_state['state'].values():
                        for k, v in group.items():
                            if isinstance(v, torch.Tensor):
                                group[k] = v.cpu()

                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Show GPU memory being cleared
                # Update the UI if progress object is available
                if hasattr(self.trainer, 'progress') and self.trainer.progress is not None:
                    self.trainer.progress.update_message("Moving to CPU for memory recovery...")
                    for i in range(5):
                        allocated, _, free, _ = track_memory_usage()
                        self.trainer.progress.update_gpu_memory(allocated, allocated + free)
                        torch.cuda.empty_cache()
                        time.sleep(1)
                else:
                    # Fallback to console if UI not available
                    self.trainer.console.print("\n[yellow]GPU Memory during cooldown:")
                    for i in range(5):
                        allocated, _, free, _ = track_memory_usage()
                        self.trainer.console.print(f"→ {allocated:.0f}MB used, {free:.0f}MB free")
                        torch.cuda.empty_cache()
                        time.sleep(1)

            # Move model and all components
            self.model.to(self.target_device)
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    module.weight = torch.nn.Parameter(module.weight.to(self.target_device))
                if hasattr(module, 'rotary_emb'):
                    module.rotary_emb.inv_freq = module.rotary_emb.inv_freq.to(self.target_device)
                    if hasattr(module.rotary_emb, 'cos_saved') and module.rotary_emb.cos_saved is not None:
                        module.rotary_emb.cos_saved = module.rotary_emb.cos_saved.to(self.target_device)
                        module.rotary_emb.sin_saved = module.rotary_emb.sin_saved.to(self.target_device)

            # Move optimizer state
            if self.target_device.type == 'cuda':
                for group in self.optimizer.state.values():
                    for k, v in group.items():
                        if isinstance(v, torch.Tensor):
                            group[k] = v.to(self.target_device)

                torch.cuda.synchronize()

        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.trainer.console.print(f"[red]Error during device transition: {str(exc_val)}")
            if self.state is not None:
                self.trainer.console.print("[yellow]Restoring model and optimizer states...")
                self.model.load_state_dict(self.state)
                if self.optimizer_state:
                    self.optimizer.load_state_dict(self.optimizer_state)

class Trainer:
    def __init__(self, config, model, tokenizer, console=None):
        """Initialize trainer with optional console parameter"""
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.console = console or default_console
        self.num_epochs = config.num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = None  # Will be set when training starts

        # Set dtype based on config
        if config.use_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif config.use_dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # Print initialization info
        self.console.print(f"\n[bold cyan]Initializing Trainer:[/bold cyan]")
        self.console.print(f"→ Device: {self.device}")
        self.console.print(f"→ Model type: {type(model).__name__}")
        self.console.print(f"→ Tokenizer: {type(tokenizer).__name__}")

        # Initialize optimizer first
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        self.console.print(f"[green]✓[/green] Optimizer initialized with lr={config.learning_rate}")

        # Move model and optimizer to device using DeviceManager
        with DeviceManager(self.model, self.optimizer, self.device, self) as model_on_device:
            self.model = model_on_device
            self.console.print(f"[green]✓[/green] Model and optimizer moved to {self.device} using DeviceManager")

        # Set up DDP if enabled
        self.master_process = True
        if config.use_ddp:
            try:
                import torch.distributed as dist
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl')
                self.master_process = dist.get_rank() == 0
                self.model = DDP(self.model, device_ids=[self.device])
                self.console.print(f"→ DDP enabled, master process: {self.master_process}")
            except Exception as e:
                self.console.print(f"[yellow]Warning: DDP initialization failed: {str(e)}. Continuing without DDP.")
                config.use_ddp = False

        self.console.print(f"→ Using dtype: {self.dtype}")

        # Set up LoRA if enabled
        if config.use_lora:
            try:
                from lora import add_lora_layers
                self.model = add_lora_layers(
                    self.model,
                    rank=config.lora_rank,  # Changed from 'r' to 'rank' to match the function signature
                    alpha=config.lora_alpha,
                    target_modules=config.lora_target_modules
                )
                self.console.print("[green]✓[/green] LoRA layers added to model")
            except Exception as e:
                self.console.print(f"[yellow]Warning: LoRA initialization failed: {str(e)}. Continuing without LoRA.")
                config.use_lora = False

        # Set up model compilation if enabled
        if config.use_compile:
            try:
                self.model = torch.compile(self.model)
                self.console.print("[green]✓[/green] Model compiled successfully")
            except Exception as e:
                self.console.print(f"[yellow]Warning: Model compilation failed: {str(e)}. Continuing without compilation.")
                config.use_compile = False

        # Set up dtype
        self.dtype = getattr(torch, config.use_dtype) if hasattr(torch, config.use_dtype) else torch.float32
        self.console.print(f"→ Using dtype: {self.dtype}")

        # Print training configuration
        self.console.print("\n[bold cyan]Training Configuration:[/bold cyan]")
        self.console.print(f"→ Epochs: {config.num_epochs}")
        self.console.print(f"→ Batch size: {config.batch_size}")
        self.console.print(f"→ Accumulation steps: {config.accumulation_steps}")
        self.console.print(f"→ Effective batch size: {config.batch_size * config.accumulation_steps}")
        self.console.print(f"→ Learning rate: {config.learning_rate}")
        self.console.print(f"→ Weight decay: {config.weight_decay}")

    def step(self, data_loader, accumulation_steps: int,
              num_tokens: int, split: str = "train"):
        """
        Performs single forward/backward pass with gradient accumulation.
            Returns: (total_loss, cross_entropy_loss, number_of_processed_tokens)
        """
        x, y = data_loader.next_batch(split=split)
        x = ensure_tensor_dtype(x, self.device)
        y = ensure_tensor_dtype(y, self.device)
        num_tokens += torch.numel(x)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            logits, loss, ce_loss = self.model(x, targets=y)

        if loss is not None:
            loss = loss / accumulation_steps
            loss.backward()
            return loss, ce_loss, num_tokens
        else:
            return None, ce_loss, num_tokens


    def train(self, data_loader, progress=None):
        """Train the model"""
        try:
            # Initialize training variables
            steps_per_epoch = len(data_loader)
            total_steps = steps_per_epoch * self.num_epochs
            running_loss = 0.0
            running_ce_loss = 0.0
            last_log_step = 0
            num_tokens = 0
            self.start_time = time.time()  # Initialize start_time here

            # Initialize the training progress display if provided
            if progress and hasattr(progress, 'start_training'):
                progress.start_training(self.num_epochs, steps_per_epoch)
            else:
                # Fallback to simple console output
                self.console.print("\nStarting training:")
                self.console.print(f"→ Total steps: {total_steps}")
                self.console.print(f"→ Steps per epoch: {steps_per_epoch}")

            for epoch in range(self.num_epochs):
                # Load checkpoint from previous epoch if enabled
                if epoch > 0 and self.config.use_epoch_checkpoints:
                    prev_epoch_checkpoint_dir = os.path.join(self.config.path_to_checkpoints, f"epoch_{epoch}")
                    prev_epoch_model_path = os.path.join(prev_epoch_checkpoint_dir, "model.pt")
                    prev_epoch_optimizer_path = os.path.join(prev_epoch_checkpoint_dir, "optimizer.pt")

                    if os.path.exists(prev_epoch_model_path) and os.path.exists(prev_epoch_optimizer_path):
                        self.console.print(f"\n[yellow]Loading checkpoint from epoch {epoch}[/yellow]")
                        try:
                            # Load model state
                            checkpoint = torch.load(prev_epoch_model_path)
                            self.model.load_state_dict(checkpoint['model_state_dict'])

                            # Load optimizer state
                            optimizer_state = torch.load(prev_epoch_optimizer_path)
                            self.optimizer.load_state_dict(optimizer_state)

                            self.console.print(f"[green]✓[/green] Successfully loaded checkpoint from epoch {epoch}")
                        except Exception as e:
                            self.console.print(f"[red]Error loading checkpoint: {str(e)}. Continuing with current model state.[/red]")
                    else:
                        self.console.print(f"[yellow]No checkpoint found for epoch {epoch}. Continuing with current model state.[/yellow]")

                # Update epoch progress
                if progress and hasattr(progress, 'update_epoch'):
                    progress.update_epoch(epoch + 1)
                else:
                    self.console.print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

                for step, batch in enumerate(data_loader):
                    try:
                        # Update the progress display for each step
                        if progress and hasattr(progress, 'update_step'):
                            # Calculate epoch step (step within current epoch)
                            epoch_step = step % steps_per_epoch

                            # Update the progress display with step information
                            progress.update_step(
                                step=step+1,
                                epoch_step=epoch_step+1
                            )

                        # Check memory before processing
                        _, _, free, _ = track_memory_usage()
                        current_device = self.device
                        # Memory monitoring function
                        def monitor_gpu_memory(seconds, interval=1):
                            # Update the UI if progress object is available
                            if progress and hasattr(progress, 'update_gpu_memory'):
                                progress.update_message("Monitoring GPU memory...")
                                for _ in range(seconds):
                                    allocated, _, free, _ = track_memory_usage()
                                    progress.update_gpu_memory(allocated, allocated + free)
                                    torch.cuda.empty_cache()
                                    time.sleep(interval)
                            else:
                                self.console.print("\n[yellow]GPU Memory Status:")
                                for _ in range(seconds):
                                    allocated, _, free, _ = track_memory_usage()
                                    self.console.print(f"→ {allocated:.0f}MB used, {free:.0f}MB free")
                                    torch.cuda.empty_cache()
                                    time.sleep(interval)
                            return self.model

                        if free < 300:  # Pre-emptive memory check
                            self.console.print("\n[yellow]Low memory detected. Switching to CPU for processing...")
                            with DeviceManager(self.model, self.optimizer, torch.device('cpu'), self) as model_on_cpu:
                                for cpu_step in range(self.config.accumulation_steps):
                                    # Process multiple steps on CPU while GPU cools
                                    input_ids = ensure_tensor_dtype(batch['input_ids'], torch.device('cpu'))
                                    labels = ensure_tensor_dtype(batch['labels'], torch.device('cpu'))

                                    self.console.print(f"[cyan]CPU Processing step {cpu_step + 1}/{self.config.accumulation_steps}...")
                                    logits, loss, ce_loss = model_on_cpu(input_ids, targets=labels)

                                    if loss is not None:
                                        loss = loss / self.config.accumulation_steps
                                        loss.backward()

                                    # Update tracking variables
                                    if loss is not None:
                                        running_loss += loss.item()
                                    if ce_loss is not None:
                                        running_ce_loss += ce_loss.item()

                                # Complete the optimization step on CPU
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                        else:
                            # Process on GPU with proper memory management
                            # Process on GPU with proper memory management
                            with DeviceManager(self.model, self.optimizer, self.device, self) as model_on_gpu:
                                try:
                                    input_ids = ensure_tensor_dtype(batch['input_ids'], self.device)
                                    labels = ensure_tensor_dtype(batch['labels'], self.device)

                                    # Update UI with current sample information if progress object is available
                                    if progress and hasattr(progress, 'update_current_sample'):
                                        # Get sample text if tokenizer is available
                                        if self.tokenizer:
                                            try:
                                                sample_text = self.tokenizer.decode(batch['input_ids'][0][:50])
                                                sample_tokens = batch['input_ids'].shape[1]
                                                progress.update_current_sample(sample_text, sample_tokens)
                                                progress.update_message(f"Processing on {self.device}...")
                                            except Exception:
                                                # Silently fail if decoding fails
                                                pass
                                    else:
                                        self.console.print(f"[cyan]Processing on {self.device}...")

                                    # Forward pass
                                    with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                                        logits, loss, ce_loss = model_on_gpu(input_ids, targets=labels)

                                        if loss is not None:
                                            # Scale loss for gradient accumulation
                                            loss = loss / self.config.accumulation_steps
                                            loss.backward()

                                    # Update tracking variables
                                    if loss is not None:
                                        running_loss += loss.item() * self.config.accumulation_steps
                                    if ce_loss is not None:
                                        running_ce_loss += ce_loss.item()

                                except RuntimeError as e:
                                    if "out of memory" in str(e):
                                        raise torch.cuda.OutOfMemoryError(str(e))
                                    raise e

                        # Backward pass
                        # Monitor memory after processing
                        if torch.cuda.is_available():
                            monitor_gpu_memory(2)  # Reduced monitoring time for faster feedback

                        # Update optimizer if we've accumulated enough gradients
                        if (step + 1) % self.config.accumulation_steps == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                allocated, _, free, _ = track_memory_usage()
                                self.console.print(f"[green]Optimizer step complete - Memory: {allocated:.0f}MB used, {free:.0f}MB free")

                    except torch.cuda.OutOfMemoryError:
                        # Update the UI if progress object is available
                        if progress and hasattr(progress, 'update_message'):
                            progress.update_message("CUDA OOM caught! Emergency switch to CPU...")
                        else:
                            self.console.print("\n[red]CUDA OOM caught! Emergency switch to CPU...")

                        current_device = torch.device('cpu')
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Emergency CPU processing with context manager
                        # Emergency CPU fallback with proper gradient accumulation
                        with DeviceManager(self.model, self.optimizer, torch.device('cpu'), self) as model_on_cpu:
                            try:
                                # Update the UI if progress object is available
                                if progress and hasattr(progress, 'update_message'):
                                    progress.update_message("Initiating emergency CPU fallback...")
                                else:
                                    self.console.print("[yellow]Initiating emergency CPU fallback...")

                                # Monitor GPU memory recovery
                                if progress and hasattr(progress, 'update_gpu_memory'):
                                    progress.update_message("Monitoring GPU memory recovery...")
                                    for _ in range(5):  # Show recovery for 5 seconds
                                        allocated, _, free, _ = track_memory_usage()
                                        progress.update_gpu_memory(allocated, allocated + free)
                                        torch.cuda.empty_cache()
                                        time.sleep(1)
                                else:
                                    self.console.print("[yellow]Monitoring GPU memory recovery...")
                                    for _ in range(5):  # Show recovery for 5 seconds
                                        allocated, _, free, _ = track_memory_usage()
                                        self.console.print(f"[yellow]→ {allocated:.0f}MB used, {free:.0f}MB free")
                                        torch.cuda.empty_cache()
                                        time.sleep(1)

                                input_ids = ensure_tensor_dtype(batch['input_ids'], torch.device('cpu'))
                                labels = ensure_tensor_dtype(batch['labels'], torch.device('cpu'))

                                # Update UI with current sample information if progress object is available
                                if progress and hasattr(progress, 'update_current_sample'):
                                    # Get sample text if tokenizer is available
                                    if self.tokenizer:
                                        try:
                                            sample_text = self.tokenizer.decode(batch['input_ids'][0][:50])
                                            sample_tokens = batch['input_ids'].shape[1]
                                            progress.update_current_sample(sample_text, sample_tokens)
                                            progress.update_message("Processing on CPU...")
                                        except Exception:
                                            # Silently fail if decoding fails
                                            pass
                                else:
                                    self.console.print("[cyan]Processing on CPU...")
                                # Forward pass
                                logits, loss, ce_loss = model_on_cpu(input_ids, targets=labels)

                                if loss is not None:
                                    # Scale loss for gradient accumulation
                                    loss = loss / self.config.accumulation_steps
                                    loss.backward()

                                    # Update tracking variables
                                    running_loss += loss.item() * self.config.accumulation_steps
                                if ce_loss is not None:
                                    running_ce_loss += ce_loss.item()

                                # Update optimizer if we've accumulated enough gradients
                                if (step + 1) % self.config.accumulation_steps == 0:
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()

                                self.console.print("[green]Successfully completed CPU fallback step")

                                # Check if GPU memory has recovered
                                if torch.cuda.is_available():
                                    _, _, free, _ = track_memory_usage()
                                    if free > 1000:  # More than 1GB free
                                        self.console.print("[green]GPU memory has recovered sufficiently")
                                    else:
                                        self.console.print("[yellow]GPU memory still low - continuing on CPU")

                            except Exception as e:
                                self.console.print(f"[red]Error during CPU processing: {str(e)}")
                                raise
                    if labels is not None:
                        num_tokens += (labels != -100).sum().item()

                    # Update weights if we've accumulated enough gradients
                    if (step + 1) % self.config.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Aggressive memory management
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Clear memory every empty_cache_freq steps
                        if step % self.config.empty_cache_freq == 0:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()

                    # Log progress
                    if step - last_log_step >= self.config.log_interval or step == total_steps - 1:
                        current_time = time.time()
                        time_elapsed = current_time - self.start_time
                        avg_loss = running_loss / (step - last_log_step + 1)
                        avg_ce_loss = running_ce_loss / (step - last_log_step + 1)
                        tokens_per_sec = num_tokens / time_elapsed if time_elapsed > 0 else 0

                        # Get memory stats and check GPU health
                        allocated, _, free, max_allocated = track_memory_usage()
                        gpu_utilization = 0.0
                        if torch.cuda.is_available():
                            try:
                                gpu_utilization = torch.cuda.utilization() * 100  # Convert to percentage
                            except:
                                # Fallback if utilization() is not available
                                gpu_utilization = (allocated / (allocated + free)) * 100 if (allocated + free) > 0 else 0

                        # Memory management logic
                        if free < 500:  # High memory pressure
                            if free < 200:  # Critical memory pressure - initiate cooling
                                self.console.print(f"\n[red]Critical memory pressure: {free:.0f}MB free")
                                self.console.print("[yellow]Initiating GPU cooling period...")

                                # Save model state
                                model_state = self.model.state_dict()

                                # Move model to CPU and clear GPU memory
                                self.model = self.model.cpu()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()

                                # Wait for GPU to cool
                                cooling_time = 15  # seconds
                                self.console.print(f"Cooling for {cooling_time} seconds...")
                                time.sleep(cooling_time)

                                # Move back to GPU
                                self.model = self.model.to(self.device)
                                self.model.load_state_dict(model_state)

                                # Permanently reduce batch size
                                if self.config.batch_size > self.config.min_batch_size:
                                    old_batch = self.config.batch_size
                                    self.config.batch_size = max(self.config.min_batch_size, self.config.batch_size - 1)
                                    data_loader.batch_size = self.config.batch_size
                                    self.console.print(f"[yellow]Reducing batch size from {old_batch} to {self.config.batch_size}")
                            else:  # High but not critical - try preventive measures
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                if self.config.batch_size > self.config.min_batch_size:
                                    self.config.batch_size -= 1
                                    data_loader.batch_size = self.config.batch_size
                                    self.console.print(f"[yellow]Preventive batch size reduction to {self.config.batch_size}")

                        # Update the rich progress display if available
                        if progress and hasattr(progress, 'update_step'):
                            # Calculate epoch step (step within current epoch)
                            epoch_step = step % steps_per_epoch

                            # Update the progress display with all metrics
                            progress.update_step(
                                step=step+1,
                                epoch_step=epoch_step+1,
                                loss=avg_loss,
                                tokens_per_sec=tokens_per_sec,
                                gpu_memory=(allocated, allocated+free),
                                gpu_utilization=gpu_utilization
                            )
                        else:
                            # Fallback to simple console output
                            log_parts = [f"Step {step+1}/{total_steps}"]
                            if avg_loss > 0:
                                log_parts.append(f"Loss: {avg_loss:.4f}")
                            if avg_ce_loss > 0:
                                log_parts.append(f"CE Loss: {avg_ce_loss:.4f}")
                            if tokens_per_sec > 0:
                                log_parts.append(f"Tokens/sec: {tokens_per_sec:.2f}")
                            log_parts.append(f"Memory: {allocated:.0f}MB used ({max_allocated:.0f}MB peak), {free:.0f}MB free")

                            self.console.print(" | ".join(log_parts))

                        # Reset counters
                        running_loss = 0.0
                        running_ce_loss = 0.0
                        num_tokens = 0
                        last_log_step = step
                        self.start_time = current_time  # Reset start time for next interval

            # Save model checkpoint at end of epoch
            if (step + 1) % len(data_loader) == 0:
                checkpoint_dir = os.path.join(self.config.path_to_checkpoints, f"epoch_{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Save model state
                model_path = os.path.join(checkpoint_dir, "model.pt")
                config_path = os.path.join(checkpoint_dir, "config.json")
                optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")

                self.console.print(f"\n[green]Saving checkpoint to {checkpoint_dir}")

                # Save model in safe way
                temp_model_path = model_path + ".tmp"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                }, temp_model_path)
                os.replace(temp_model_path, model_path)

                # Save optimizer state
                temp_optimizer_path = optimizer_path + ".tmp"
                torch.save(self.optimizer.state_dict(), temp_optimizer_path)
                os.replace(temp_optimizer_path, optimizer_path)

                self.console.print("[green]✓[/green] Checkpoint saved successfully")

        except Exception as e:
            self.console.print(f"\nError in training loop:\n→ {str(e)}")
            raise e
