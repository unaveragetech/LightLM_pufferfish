# python train.py

import struct
import sys
import shutil
import json
import statistics

import datasets
from model import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, CustomDataLoader, ThreadedDataLoader
from lora import add_lora_layers

from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os
import glob
import warnings
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TransferSpeedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.console import Group
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import requests
from tqdm.auto import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List, Tuple
from enum import Enum
import threading
from queue import Queue, Empty
import psutil
import torch
import platform
import shutil
import sys

# Add rich console import with fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    default_console = console

    # Global variable to store the selected dataset
    selected_dataset = None
except ImportError:
    # Fallback if rich is not installed
    console = print
    default_console = print
    class Panel:
        def __init__(self, text, title=None):
            self.renderable = f"\n{title if title else ''}\n{text}\n"


TOKENIZER_ID = "HuggingFaceTB/SmolLM-360M"  # Define as a constant at the top
MAX_ULTRA_SHORT_SAMPLES = 20  # Maximum samples for ultra-short training

class ConstantLengthDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, seq_length):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length

    def __iter__(self):
        buffer_tokens = []
        for item in self.dataset:
            text = item['text'].strip()
            if not text:
                continue

            tokens = self.tokenizer(text, truncation=True, max_length=self.seq_length)['input_ids']
            buffer_tokens.extend(tokens)

            while len(buffer_tokens) >= self.seq_length:
                yield torch.tensor(buffer_tokens[:self.seq_length])
                buffer_tokens = buffer_tokens[self.seq_length:]

class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ComputeNode:
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    dependencies: list['ComputeNode'] = None
    priority: int = 0

class BackgroundTaskManager:
    def __init__(self, max_workers=None):
        if max_workers is None:
            cpu_count = psutil.cpu_count(logical=False)
            self.max_workers = max(1, cpu_count - 1)  # Leave one core free
        else:
            self.max_workers = max_workers

        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.tasks = {}
        self.queue = Queue()
        self.running = True
        self._start_worker_thread()

    def _start_worker_thread(self):
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _process_queue(self):
        while self.running:
            try:
                node = self.queue.get(timeout=1)
                if node.dependencies:
                    if not all(self.tasks[dep.task_id].status == NodeStatus.COMPLETED
                             for dep in node.dependencies):
                        self.queue.put(node)
                        continue

                self._execute_node(node)
            except Empty:  # Changed from Queue.Empty to Empty
                continue
            except Exception as e:
                console.print(f"[red]Error in queue processing: {str(e)}")
                continue

    def _execute_node(self, node):
        node.status = NodeStatus.RUNNING
        try:
            # Choose appropriate executor based on task type
            executor = self.process_pool if node.kwargs.get('use_process', False) else self.thread_pool
            future = executor.submit(node.func, *node.args, **node.kwargs)
            node.result = future.result()
            node.status = NodeStatus.COMPLETED
        except Exception as e:
            node.error = e
            node.status = NodeStatus.FAILED
            console.print(f"[red]Error in task {node.task_id}: {str(e)}")

    def add_task(self, task_id: str, func: Callable, *args,
                 dependencies=None, priority=0, **kwargs) -> ComputeNode:
        node = ComputeNode(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies,
            priority=priority
        )
        self.tasks[task_id] = node
        self.queue.put(node)
        return node

    def shutdown(self):
        self.running = False
        self.thread_pool.shutdown()
        self.process_pool.shutdown()

class TrainingProgress:
    def __init__(self):
        # Create a live display with multiple panels
        self.live = None
        self.console = Console()

        # Main progress bar for overall training
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
            MofNCompleteColumn(),
            TextColumn("[bold]{task.fields[status]}")
        )

        # Epoch progress bar
        self.epoch_progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40, complete_style="cyan"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )

        # Step progress bar
        self.step_progress = Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(bar_width=40, complete_style="green"),
            TaskProgressColumn(),
            TransferSpeedColumn(),
        )

        # Initialize metrics tracking
        self.metrics = {
            "loss": [],
            "tokens_per_sec": [],
            "gpu_memory": [],
            "gpu_utilization": [],
        }

        # Task IDs
        self.main_task_id = None
        self.epoch_task_id = None
        self.step_task_id = None

        # Current state
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.current_tokens_per_sec = 0.0
        self.gpu_memory_used = 0
        self.gpu_memory_total = 0
        self.gpu_utilization = 0
        self.message = ""

        # Dataset and model information
        self.current_sample = ""
        self.sample_tokens = 0
        self.dataset_name = ""
        self.dataset_size = 0
        self.subset_size = 0
        self.use_subset = False
        self.current_device = "cuda"
        self.batch_size = 0
        self.effective_batch_size = 0
        self.learning_rate = 0.0
        self.model_params = 0
        self.model_layers = 0
        self.model_dims = 0

        # UI update throttling
        self.last_update_time = 0
        self.update_interval = 0.5  # Update UI at most every 0.5 seconds
        self.pending_update = False

        # Smoothing for metrics
        self.smoothing_factor = 0.7  # Higher = more smoothing
        self.smoothed_loss = 0.0
        self.smoothed_tokens_per_sec = 0.0
        self.smoothed_gpu_memory = 0.0
        self.smoothed_gpu_utilization = 0.0

    def start_training(self, total_epochs, steps_per_epoch):
        """Initialize the live display for training"""
        self.total_epochs = total_epochs
        self.total_steps = steps_per_epoch * total_epochs

        # Create tasks
        self.main_task_id = self.progress.add_task("[bold]Training Progress", total=self.total_steps, status="")
        self.epoch_task_id = self.epoch_progress.add_task(f"Epoch 1/{total_epochs}", total=total_epochs)
        self.step_task_id = self.step_progress.add_task(f"Steps in Epoch 1", total=steps_per_epoch)

        # Create the live display
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich import box

        # Create a layout with sub-layouts
        layout = Layout(name="main")
        layout.split(
            Layout(name="progress", ratio=1),
            Layout(name="metrics", ratio=1),
            Layout(name="memory", ratio=1)
        )

        # Create a single comprehensive table for all information
        main_table = Table(show_header=True, header_style="bold blue", width=120, box=box.ROUNDED)

        # Add columns for the consolidated table
        main_table.add_column("Category", style="bold cyan", width=15)
        main_table.add_column("Metric", style="dim", width=15)
        main_table.add_column("Value", justify="right", width=20)
        main_table.add_column("Progress", justify="center", width=30)
        main_table.add_column("Details", justify="right", width=30)

        # Add progress information
        main_table.add_row(
            "Progress",
            "Overall",
            f"0/{self.total_steps} steps",
            "[0%] ░░░░░░░░░░░░░░░░░░░░",
            "ETA: --:--:--"
        )
        main_table.add_row(
            "Progress",
            "Epoch",
            f"1/{self.total_epochs}",
            "[0%] ░░░░░░░░░░░░░░░░░░░░",
            "Current: 1"
        )
        main_table.add_row(
            "Progress",
            "Step",
            f"0/{steps_per_epoch}",
            "[0%] ░░░░░░░░░░░░░░░░░░░░",
            "In current epoch"
        )

        # Add separator
        main_table.add_row("", "", "", "", "")

        # Add metrics information
        main_table.add_row(
            "Metrics",
            "Loss",
            "0.0000",
            "",
            "Min: 0.0 / Max: 0.0"
        )
        main_table.add_row(
            "Metrics",
            "Tokens/sec",
            "0.00",
            "",
            "Avg: 0.00"
        )
        main_table.add_row(
            "Metrics",
            "Learning Rate",
            "0.001000",
            "",
            ""
        )
        main_table.add_row(
            "Metrics",
            "Batch Size",
            "4",
            "",
            "Effective: 16"
        )

        # Add separator
        main_table.add_row("", "", "", "", "")

        # Add resource usage information
        main_table.add_row(
            "Resources",
            "GPU Memory",
            "0 MB / 0 MB",
            "[0%] ░░░░░░░░░░░░░░░░░░░░",
            "Peak: 0 MB"
        )
        main_table.add_row(
            "Resources",
            "GPU Utilization",
            "0%",
            "[0%] ░░░░░░░░░░░░░░░░░░░░",
            "Peak: 0%"
        )
        main_table.add_row(
            "Resources",
            "RAM Usage",
            "0 MB / 0 MB",
            "[0%] ░░░░░░░░░░░░░░░░░░░░",
            ""
        )
        main_table.add_row(
            "Resources",
            "VRAM Efficiency",
            "0.0000 tokens/MB",
            "",
            ""
        )

        # Set the main table as the progress layout content
        layout["progress"].update(Panel(main_table, title="[bold]LightLM Training Progress[/bold]", border_style="green"))

        # Initialize the metrics and memory layouts with empty panels
        metrics_table = Table(show_header=True, header_style="bold magenta", width=120)
        metrics_table.add_column("Metric", style="dim", width=20)
        metrics_table.add_column("Current Value", justify="right", width=20)
        metrics_table.add_column("Average", justify="right", width=20)
        metrics_table.add_column("Min/Max", justify="right", width=20)
        metrics_table.add_column("Trend", justify="center", width=30)
        layout["metrics"].update(Panel(metrics_table, title="Live Metrics", border_style="magenta"))

        memory_table = Table(show_header=True, header_style="bold yellow", width=120)
        memory_table.add_column("Metric", style="dim", width=20)
        memory_table.add_column("Current Value", justify="right", width=20)
        memory_table.add_column("Peak", justify="right", width=20)
        memory_table.add_column("Total", justify="right", width=20)
        memory_table.add_column("Usage Bar", justify="center", width=30)
        layout["memory"].update(Panel(memory_table, title="Resource Usage", border_style="yellow"))

        # Start the live display with a lower refresh rate for stability
        self.live = Live(
            layout,
            refresh_per_second=2,  # Lower refresh rate for stability
            console=self.console,
            auto_refresh=False    # We'll manually refresh with throttling
        )
        self.live.start()

    def update_message(self, message):
        """Update the status message in the UI"""
        # Create a new row or update existing row with the message
        self.message = message
        self._refresh_display()

    def update_gpu_memory(self, used_mb, total_mb):
        """Update the GPU memory display in the UI"""
        self.gpu_memory_used = used_mb
        self.gpu_memory_total = total_mb
        self._refresh_display()

    def update_dataset_info(self, dataset_name, dataset_size, subset_size=None, use_subset=False):
        """Update dataset information in the UI"""
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.subset_size = subset_size if subset_size is not None else dataset_size
        self.use_subset = use_subset
        self._refresh_display()

    def update_model_info(self, model_params, model_layers, model_dims):
        """Update model information in the UI"""
        self.model_params = model_params
        self.model_layers = model_layers
        self.model_dims = model_dims
        self._refresh_display()

    def update_training_info(self, batch_size, effective_batch_size, learning_rate, current_device):
        """Update training configuration information in the UI"""
        self.batch_size = batch_size
        self.effective_batch_size = effective_batch_size
        self.learning_rate = learning_rate
        self.current_device = current_device
        self._refresh_display()

    def update_current_sample(self, sample_text, sample_tokens):
        """Update information about the current sample being processed"""
        if isinstance(sample_text, str) and len(sample_text) > 100:
            # Truncate long samples for display
            sample_text = sample_text[:100] + "..."
        self.current_sample = sample_text
        self.sample_tokens = sample_tokens
        self._refresh_display()

    def update_epoch(self, epoch):
        """Update the epoch progress"""
        self.current_epoch = epoch
        self.epoch_progress.update(self.epoch_task_id, completed=epoch, description=f"Epoch {epoch}/{self.total_epochs}")
        self.step_progress.update(self.step_task_id, completed=0, description=f"Steps in Epoch {epoch}")
        self._refresh_display()

    def update_step(self, step, epoch_step, loss=None, tokens_per_sec=None, gpu_memory=None, gpu_utilization=None):
        """Update the step progress and metrics with throttling and smoothing"""
        self.current_step = step
        self.progress.update(self.main_task_id, completed=step)
        self.step_progress.update(self.step_task_id, completed=epoch_step)

        # Always refresh the display to show step progress
        self._refresh_display()

        # Update metrics if provided with exponential smoothing
        if loss is not None:
            # Apply smoothing to loss
            if self.smoothed_loss == 0.0:  # First value
                self.smoothed_loss = loss
            else:
                self.smoothed_loss = self.smoothing_factor * self.smoothed_loss + (1 - self.smoothing_factor) * loss

            self.current_loss = self.smoothed_loss
            self.metrics["loss"].append(loss)  # Store raw value in metrics

        if tokens_per_sec is not None:
            # Apply smoothing to tokens_per_sec
            if self.smoothed_tokens_per_sec == 0.0:  # First value
                self.smoothed_tokens_per_sec = tokens_per_sec
            else:
                self.smoothed_tokens_per_sec = self.smoothing_factor * self.smoothed_tokens_per_sec + (1 - self.smoothing_factor) * tokens_per_sec

            self.current_tokens_per_sec = self.smoothed_tokens_per_sec
            self.metrics["tokens_per_sec"].append(tokens_per_sec)  # Store raw value in metrics

        # Update GPU memory usage if provided
        if gpu_memory is not None:
            # Unpack GPU memory tuple (allocated, total)
            allocated_memory, total_memory = gpu_memory
            memory_usage = allocated_memory / total_memory  # Calculate usage ratio
            
            # Debug logging
            print(f"GPU Memory - Allocated: {allocated_memory:.2f}GB, Total: {total_memory:.2f}GB, Usage: {memory_usage:.2%}")
            print(f"Current smoothed_gpu_memory: {self.smoothed_gpu_memory:.2%}")
            
            # Apply smoothing to GPU memory usage ratio
            if self.smoothed_gpu_memory == 0.0:  # First value
                self.smoothed_gpu_memory = memory_usage
            else:
                self.smoothed_gpu_memory = self.smoothing_factor * self.smoothed_gpu_memory + (1 - self.smoothing_factor) * memory_usage

            self.gpu_memory_used = allocated_memory
            self.metrics["gpu_memory"].append(allocated_memory)  # Store allocated memory value

            # Try to get total GPU memory if not already set
            if self.gpu_memory_total == 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # Convert to MB
                except:
                    pass

        # Update GPU utilization if provided
        if gpu_utilization is not None:
            # Apply smoothing to GPU utilization
            if self.smoothed_gpu_utilization == 0.0:  # First value
                self.smoothed_gpu_utilization = gpu_utilization
            else:
                self.smoothed_gpu_utilization = self.smoothing_factor * self.smoothed_gpu_utilization + (1 - self.smoothing_factor) * gpu_utilization

            self.gpu_utilization = self.smoothed_gpu_utilization
            self.metrics["gpu_utilization"].append(gpu_utilization)  # Store raw value in metrics

        # Throttle UI updates to avoid excessive refreshes
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._refresh_display()
            self.last_update_time = current_time
            self.pending_update = False
        else:
            # Mark that we have pending updates
            self.pending_update = True

    def _refresh_display(self):
        """Refresh the live display with current metrics"""
        if not self.live:
            return

        # Get the existing layout
        layout = self.live.renderable

        # Create tables for each section
        from rich.table import Table
        from rich import box
        import statistics

        # 1. Create the main progress table
        main_table = Table(show_header=True, header_style="bold blue", width=120, box=box.ROUNDED)
        main_table.add_column("Category", style="bold cyan", width=15)
        main_table.add_column("Metric", style="dim", width=15)
        main_table.add_column("Value", justify="right", width=20)
        main_table.add_column("Progress", justify="center", width=30)
        main_table.add_column("Details", justify="right", width=30)

        # Add status message if present
        if hasattr(self, 'message') and self.message:
            main_table.add_row(
                "Status",
                "Message",
                "",
                "",
                f"[bold yellow]{self.message}[/bold yellow]"
            )
            main_table.add_row("", "", "", "", "")

        # Add progress information
        main_table.add_row(
            "Progress",
            "Overall",
            f"{self.current_step}/{self.total_steps} steps",
            self.progress.tasks[self.main_task_id].description,
            f"ETA: {self.progress.tasks[self.main_task_id].time_remaining or '--:--:--'}"
        )
        main_table.add_row(
            "Progress",
            "Epoch",
            f"{self.current_epoch}/{self.total_epochs}",
            self.epoch_progress.tasks[self.epoch_task_id].description,
            f"Current: {self.current_epoch}"
        )
        main_table.add_row(
            "Progress",
            "Step",
            f"{self.step_progress.tasks[self.step_task_id].completed}/{self.step_progress.tasks[self.step_task_id].total}",
            self.step_progress.tasks[self.step_task_id].description,
            "In current epoch"
        )

        # Update the main progress panel
        layout["progress"].update(Panel(main_table, title="[bold]LightLM Training Progress[/bold]", border_style="green"))

        # 2. Create the metrics table
        metrics_table = Table(show_header=True, header_style="bold magenta", width=120)
        metrics_table.add_column("Metric", style="dim", width=20)
        metrics_table.add_column("Current Value", justify="right", width=20)
        metrics_table.add_column("Average", justify="right", width=20)
        metrics_table.add_column("Min/Max", justify="right", width=20)
        metrics_table.add_column("Trend", justify="center", width=30)

        # Add dataset information section
        if self.dataset_name:
            metrics_table.add_row(
                "[bold cyan]Dataset[/bold cyan]",
                f"{self.dataset_name}",
                "-",
                "-",
                ""
            )

            # Show dataset size information
            if self.use_subset and self.subset_size < self.dataset_size:
                metrics_table.add_row(
                    "Dataset Size",
                    f"{self.subset_size:,} samples",
                    "-",
                    f"Full: {self.dataset_size:,}",
                    f"[yellow]{self.subset_size/self.dataset_size*100:.1f}% of full[/yellow]"
                )
            else:
                metrics_table.add_row(
                    "Dataset Size",
                    f"{self.dataset_size:,} samples",
                    "-",
                    "-",
                    "[green]Full dataset[/green]"
                )

        # Add current sample information if available
        if self.current_sample:
            sample_display = self.current_sample
            if len(sample_display) > 30:
                sample_display = sample_display[:30] + "..."
            metrics_table.add_row(
                "Current Sample",
                f"{sample_display}",
                "-",
                f"Tokens: {self.sample_tokens}",
                ""
            )

        # Add model information if available
        if self.model_params > 0:
            metrics_table.add_row(
                "[bold cyan]Model[/bold cyan]",
                f"Layers: {self.model_layers}",
                f"Dims: {self.model_dims}",
                f"Params: {self.model_params/1e6:.1f}M",
                ""
            )

        # Add training configuration information
        if self.batch_size > 0:
            metrics_table.add_row(
                "[bold cyan]Training[/bold cyan]",
                f"Device: {self.current_device}",
                "-",
                "-",
                ""
            )
            metrics_table.add_row(
                "Batch Size",
                f"{self.batch_size}",
                "-",
                f"Effective: {self.effective_batch_size}",
                ""
            )
            metrics_table.add_row(
                "Learning Rate",
                f"{self.learning_rate:.6f}",
                "-",
                "-",
                ""
            )

        # Add separator
        metrics_table.add_row(
            "[bold cyan]Metrics[/bold cyan]",
            "",
            "",
            "",
            ""
        )

        # Add loss row with trend and statistics
        if self.metrics["loss"]:
            loss_trend = self._get_trend(self.metrics["loss"], 5)
            loss_avg = statistics.mean(self.metrics["loss"])
            loss_min = min(self.metrics["loss"])
            loss_max = max(self.metrics["loss"])
            metrics_table.add_row(
                "Loss",
                f"{self.current_loss:.4f}",
                f"{loss_avg:.4f}",
                f"{loss_min:.4f} / {loss_max:.4f}",
                loss_trend
            )
        else:
            metrics_table.add_row(
                "Loss",
                "0.0000",
                "0.0000",
                "0.0000 / 0.0000",
                ""
            )

        # Add tokens/sec row with trend and statistics
        if self.metrics["tokens_per_sec"]:
            tokens_trend = self._get_trend(self.metrics["tokens_per_sec"], 5)
            tokens_avg = statistics.mean(self.metrics["tokens_per_sec"])
            tokens_min = min(self.metrics["tokens_per_sec"])
            tokens_max = max(self.metrics["tokens_per_sec"])
            metrics_table.add_row(
                "Tokens/sec",
                f"{self.current_tokens_per_sec:.2f}",
                f"{tokens_avg:.2f}",
                f"{tokens_min:.2f} / {tokens_max:.2f}",
                tokens_trend
            )
        else:
            metrics_table.add_row(
                "Tokens/sec",
                "0.00",
                "0.00",
                "0.00 / 0.00",
                ""
            )

        # Add gradient norm information (placeholder)
        metrics_table.add_row(
            "Gradient Norm",
            f"0.0",  # Placeholder - would need to get from trainer
            "-",
            "-",
            ""
        )

        # Update the metrics panel
        layout["metrics"].update(Panel(metrics_table, title="Live Metrics", border_style="magenta"))

        # 3. Create the resource usage table
        memory_table = Table(show_header=True, header_style="bold yellow", width=120)
        memory_table.add_column("Metric", style="dim", width=20)
        memory_table.add_column("Current Value", justify="right", width=20)
        memory_table.add_column("Peak", justify="right", width=20)
        memory_table.add_column("Total", justify="right", width=20)
        memory_table.add_column("Usage Bar", justify="center", width=30)

        # Add GPU memory usage with bar
        if isinstance(self.gpu_memory_used, tuple) and isinstance(self.gpu_memory_total, tuple):
            # Handle the case when both are tuples
            if len(self.gpu_memory_used) > 0 and len(self.gpu_memory_total) > 0 and self.gpu_memory_total[0] > 0:
                memory_percent = (self.gpu_memory_used[0] / self.gpu_memory_total[0]) * 100
                memory_bar = self._create_usage_bar(memory_percent)
                peak_memory = max(self.metrics["gpu_memory"]) if self.metrics["gpu_memory"] else self.gpu_memory_used[0]
                memory_table.add_row(
                    "GPU Memory",
                    f"{self.gpu_memory_used[0]:.0f} MB",
                    f"{peak_memory:.0f} MB",
                    f"{self.gpu_memory_total[0]:.0f} MB",
                    f"{memory_bar} [{memory_percent:.1f}%]"
                )
            else:
                memory_table.add_row("GPU Memory", "N/A", "N/A", "N/A", "")
        elif isinstance(self.gpu_memory_used, tuple):
            # Handle the case when only gpu_memory_used is a tuple
            if len(self.gpu_memory_used) > 0 and self.gpu_memory_total > 0:
                memory_percent = (self.gpu_memory_used[0] / self.gpu_memory_total) * 100
                memory_bar = self._create_usage_bar(memory_percent)
                # Handle the case when peak_memory might be a tuple
                if self.metrics["gpu_memory"]:
                    peak_memory_value = max(self.metrics["gpu_memory"])
                    if isinstance(peak_memory_value, tuple) and len(peak_memory_value) > 0:
                        peak_memory_value = peak_memory_value[0]
                else:
                    peak_memory_value = self.gpu_memory_used[0]

                # Handle the case when gpu_memory_total might be a tuple
                if isinstance(self.gpu_memory_total, tuple) and len(self.gpu_memory_total) > 0:
                    total_memory = self.gpu_memory_total[0]
                else:
                    total_memory = self.gpu_memory_total

                memory_table.add_row(
                    "GPU Memory",
                    f"{self.gpu_memory_used[0]:.0f} MB",
                    f"{peak_memory_value:.0f} MB",
                    f"{total_memory:.0f} MB",
                    f"{memory_bar} [{memory_percent:.1f}%]"
                )
            else:
                memory_table.add_row("GPU Memory", "N/A", "N/A", "N/A", "")
        elif isinstance(self.gpu_memory_total, tuple):
            # Handle the case when only gpu_memory_total is a tuple
            if len(self.gpu_memory_total) > 0 and self.gpu_memory_used > 0:
                memory_percent = (self.gpu_memory_used / self.gpu_memory_total[0]) * 100
                memory_bar = self._create_usage_bar(memory_percent)

                # Handle the case when peak_memory might be a tuple
                if self.metrics["gpu_memory"]:
                    peak_memory_value = max(self.metrics["gpu_memory"])
                    if isinstance(peak_memory_value, tuple) and len(peak_memory_value) > 0:
                        peak_memory_value = peak_memory_value[0]
                else:
                    peak_memory_value = self.gpu_memory_used

                # Make sure peak_memory_value is not a tuple
                if isinstance(peak_memory_value, tuple) and len(peak_memory_value) > 0:
                    peak_memory_value = peak_memory_value[0]

                memory_table.add_row(
                    "GPU Memory",
                    f"{self.gpu_memory_used:.0f} MB",
                    f"{peak_memory_value:.0f} MB",
                    f"{self.gpu_memory_total[0]:.0f} MB",
                    f"{memory_bar} [{memory_percent:.1f}%]"
                )
            else:
                memory_table.add_row("GPU Memory", "N/A", "N/A", "N/A", "")
        elif self.gpu_memory_total > 0 and self.gpu_memory_used > 0:
            # Handle the case when both are numbers
            memory_percent = (self.gpu_memory_used / self.gpu_memory_total) * 100
            memory_bar = self._create_usage_bar(memory_percent)

            # Handle the case when peak_memory might be a tuple
            if self.metrics["gpu_memory"]:
                peak_memory_value = max(self.metrics["gpu_memory"])
                if isinstance(peak_memory_value, tuple) and len(peak_memory_value) > 0:
                    peak_memory_value = peak_memory_value[0]
            else:
                peak_memory_value = self.gpu_memory_used

            # Make sure peak_memory_value is not a tuple
            if isinstance(peak_memory_value, tuple) and len(peak_memory_value) > 0:
                peak_memory_value = peak_memory_value[0]

            memory_table.add_row(
                "GPU Memory",
                f"{self.gpu_memory_used:.0f} MB",
                f"{peak_memory_value:.0f} MB",
                f"{self.gpu_memory_total:.0f} MB",
                f"{memory_bar} [{memory_percent:.1f}%]"
            )
        else:
            memory_table.add_row("GPU Memory", "N/A", "N/A", "N/A", "")

        # Add GPU utilization with bar
        if isinstance(self.gpu_utilization, tuple):
            # Handle the case when gpu_utilization is a tuple
            if len(self.gpu_utilization) > 0:
                utilization = self.gpu_utilization[0]
                utilization_bar = self._create_usage_bar(utilization)
                peak_utilization = max(self.metrics["gpu_utilization"]) if self.metrics["gpu_utilization"] else utilization
                memory_table.add_row(
                    "GPU Utilization",
                    f"{utilization:.1f}%",
                    f"{peak_utilization:.1f}%",
                    "100%",
                    f"{utilization_bar} [{utilization:.1f}%]"
                )
            else:
                memory_table.add_row("GPU Utilization", "N/A", "N/A", "100%", "")
        else:
            # Handle the case when gpu_utilization is a number
            utilization_bar = self._create_usage_bar(self.gpu_utilization)
            peak_utilization = max(self.metrics["gpu_utilization"]) if self.metrics["gpu_utilization"] else self.gpu_utilization
            memory_table.add_row(
                "GPU Utilization",
                f"{self.gpu_utilization:.1f}%",
                f"{peak_utilization:.1f}%",
                "100%",
                f"{utilization_bar} [{self.gpu_utilization:.1f}%]"
            )

        # Add RAM usage information
        try:
            import psutil
            ram = psutil.virtual_memory()
            ram_used = ram.used / (1024**2)  # Convert to MB
            ram_total = ram.total / (1024**2)  # Convert to MB
            ram_percent = ram.percent
            ram_bar = self._create_usage_bar(ram_percent)
            memory_table.add_row(
                "RAM Usage",
                f"{ram_used:.0f} MB",
                "-",
                f"{ram_total:.0f} MB",
                f"{ram_bar} [{ram_percent:.1f}%]"
            )
        except Exception:
            memory_table.add_row("RAM Usage", "N/A", "-", "N/A", "")

        # Add VRAM efficiency (tokens processed per MB of VRAM)
        if self.current_tokens_per_sec > 0:
            # Handle the case when gpu_memory_used is a tuple
            if isinstance(self.gpu_memory_used, tuple):
                # Use the first value of the tuple if it's not empty
                if len(self.gpu_memory_used) > 0 and self.gpu_memory_used[0] > 0:
                    vram_efficiency = self.current_tokens_per_sec / self.gpu_memory_used[0]
                    memory_table.add_row(
                        "VRAM Efficiency",
                        f"{vram_efficiency:.4f} tokens/MB",
                        "-",
                        "-",
                        ""
                    )
                else:
                    memory_table.add_row("VRAM Efficiency", "0 tokens/MB", "-", "-", "")
            # Handle the case when gpu_memory_used is a number
            elif self.gpu_memory_used > 0:
                vram_efficiency = self.current_tokens_per_sec / self.gpu_memory_used
                memory_table.add_row(
                    "VRAM Efficiency",
                    f"{vram_efficiency:.4f} tokens/MB",
                    "-",
                    "-",
                    ""
                )
            else:
                memory_table.add_row("VRAM Efficiency", "0 tokens/MB", "-", "-", "")
        else:
            memory_table.add_row("VRAM Efficiency", "0 tokens/MB", "-", "-", "")

        # Update the memory panel
        layout["memory"].update(Panel(memory_table, title="Resource Usage", border_style="yellow"))

        # Manually refresh the display
        try:
            self.live.refresh()
        except Exception as e:
            # Gracefully handle any refresh errors
            pass

    def _get_trend(self, values, window=5):
        """Create a trend indicator based on recent values"""
        if len(values) < 2:
            return ""

        # Get the most recent values within the window
        recent = values[-window:] if len(values) >= window else values

        # Calculate the trend direction
        if len(recent) >= 2:
            first_avg = sum(recent[:len(recent)//2]) / (len(recent)//2)
            second_avg = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

            if second_avg < first_avg * 0.98:  # Decreasing trend (good for loss)
                return "[green]↓ Decreasing[/green]"
            elif second_avg > first_avg * 1.02:  # Increasing trend (good for tokens/sec)
                return "[green]↑ Increasing[/green]"
            else:
                return "[yellow]→ Stable[/yellow]"

        return ""

    def _create_usage_bar(self, percentage, width=20):
        """Create a visual bar representing usage percentage"""
        if not isinstance(percentage, (int, float)):
            return ""

        filled = int((percentage / 100) * width)
        empty = width - filled

        # Color coding based on usage level
        if percentage < 50:
            color = "green"
            filled_char = "█"  # Full block
        elif percentage < 80:
            color = "yellow"
            filled_char = "█"  # Full block
        else:
            color = "red"
            filled_char = "█"  # Full block

        # Use different characters for empty space to make it more visible
        empty_char = "░"  # Light shade

        return f"[{color}]{filled_char * filled}[/{color}]{empty_char * empty}"

    def create_task(self, description, total):
        """Create a task in the main progress bar"""
        return self.progress.add_task(description, total=total, status="")

    def update(self, task_id, advance=None, status=None):
        """Update a task in the main progress bar"""
        if advance is not None:
            self.progress.update(task_id, advance=advance)
        if status is not None:
            self.progress.update(task_id, status=status)

    def stop(self):
        """Stop the live display"""
        # Process any pending updates before stopping
        if self.pending_update:
            self._refresh_display()

        if self.live:
            self.live.stop()

class SystemCheck:
    def __init__(self):
        self.console = default_console

    def check_python_version(self):
        version = sys.version_info
        min_version = (3, 8)
        recommended_version = (3, 10)

        if version < min_version:
            return False, f"Python {version.major}.{version.minor} detected. Minimum required: 3.8"
        elif version < recommended_version:
            return True, f"Python {version.major}.{version.minor} detected. Recommended: 3.10+"
        return True, f"Python {version.major}.{version.minor} detected ✓"

    def check_gpu(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"GPU: {gpu_name} ({memory:.1f}GB) ✓"
        return False, "No GPU detected. Training will be slow on CPU"

    def check_disk_space(self, min_space_gb=10):
        path = "."
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        if free_gb < min_space_gb:
            return False, f"Low disk space: {free_gb:.1f}GB free. Minimum recommended: {min_space_gb}GB"
        return True, f"Disk space: {free_gb:.1f}GB free ✓"

    def check_ram(self, min_ram_gb=8):
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        if ram_gb < min_ram_gb:
            return False, f"RAM: {ram_gb:.1f}GB. Minimum recommended: {min_ram_gb}GB"
        return True, f"RAM: {ram_gb:.1f}GB ✓"

    def run_all_checks(self):
        self.console.print("\n[bold cyan]System Check:[/bold cyan]")

        checks = [
            ("Python Version", self.check_python_version()),
            ("GPU Status", self.check_gpu()),
            ("Disk Space", self.check_disk_space()),
            ("RAM", self.check_ram())
        ]

        all_passed = True
        warnings = []

        for check_name, (passed, message) in checks:
            if passed:
                self.console.print(f"[green]✓[/green] {check_name}: {message}")
            else:
                all_passed = False
                self.console.print(f"[yellow]![/yellow] {check_name}: {message}")
                warnings.append(message)

        return all_passed, warnings

class HardwareManager:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory if self.gpu_available else 0
        self.cpu_count = psutil.cpu_count(logical=False)
        self.ram = psutil.virtual_memory().total

        # Cache GPU capabilities
        self.gpu_capabilities = self._get_gpu_capabilities() if self.gpu_available else {}

    def get_gpu_compute_capability(self):
        """Get GPU compute capability as a float (e.g., 7.5)"""
        if not self.gpu_available:
            return 0.0
        try:
            return float(self.gpu_capabilities['compute_capability'])
        except (KeyError, ValueError, TypeError):
            return 0.0

    def _get_gpu_capabilities(self):
        """Get detailed GPU capabilities"""
        props = torch.cuda.get_device_properties(0)
        return {
            'name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory': props.total_memory,
            'multi_processor_count': props.multi_processor_count,
        }

    def get_gpu_specs(self):
        """Get GPU specifications for performance estimation"""
        if not self.gpu_available:
            return None
        return self.gpu_capabilities

    def supports_flash_attention(self):
        """Check if GPU supports Flash Attention"""
        if not self.gpu_available:
            return False
        try:
            return float(self.gpu_capabilities['compute_capability']) >= 7.5
        except (KeyError, ValueError, TypeError):
            return False

    @property
    def flash_attention_available(self):
        """Property to check if Flash Attention is available"""
        return self.supports_flash_attention()

    def get_max_batch_size(self, config):
        """Estimate maximum batch size based on available memory"""
        if not self.gpu_available:
            return 1

        # Rough estimation of memory per sample
        memory_per_sample = (
            config.max_seq_len * config.vocab_size * 4  # 4 bytes per float
            * (2 if config.use_flash else 3)  # Memory multiplier based on attention implementation
            * (1.5 if config.gradient_checkpointing else 2)  # Memory for gradients
        )

        # Leave some memory for other operations
        usable_memory = self.gpu_memory * config.gpu_memory_utilization
        max_batch = max(1, int(usable_memory / memory_per_sample))

        return max_batch

    def optimize_config(self, config, target_time_hours=None):
        """Optimize training configuration based on hardware and target time"""
        if target_time_hours is not None:
            # Ultra-short training configuration
            if target_time_hours < 0.01:  # Less than 36 seconds
                config.num_epochs = 1
                config.max_seq_len = 128
                config.batch_size = min(12, self.get_max_batch_size(config))
                config.gradient_checkpointing = False
                config.empty_cache_freq = 1000
                config.enable_memory_tracking = False
                config.gpu_memory_utilization = 0.95
                config.checkpoints_frequency = 1
                config.log_interval = 1
                config.eval_interval = 1
                config.use_flash = self.flash_attention_available
                return config

        # Standard optimization for regular training
        if self.gpu_available:
            # Calculate optimal batch size based on available GPU memory
            available_memory = self.gpu_memory * config.gpu_memory_utilization

            # Enable memory optimizations for limited memory
            if available_memory < 8 * (1024**3):  # Less than 8GB
                config.gradient_checkpointing = True
                config.empty_cache_freq = 50
                config.enable_memory_tracking = True
                config.empty_cache_freq = 50

            # Adjust batch size based on available memory
            if config.auto_adjust_batch:
                # Simple heuristic: adjust batch size based on available memory
                memory_factor = available_memory / (8 * (1024**3))  # normalize to 8GB
                config.batch_size = max(
                    config.min_batch_size,
                    min(config.batch_size, int(config.batch_size * memory_factor))
                )
        else:
            # CPU optimizations
            config.batch_size = max(1, config.batch_size // 2)
            config.gradient_checkpointing = True
            config.use_compile = False

        return config

def setup_training_environment():
    """Initialize training environment and background task manager"""
    # Run system checks first
    system_check = SystemCheck()
    checks_passed, warnings = system_check.run_all_checks()

    if not checks_passed:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"• {warning}")

        # Ask user if they want to continue
        if not get_user_confirmation("Continue despite warnings?"):
            console.print("\n[red]Setup aborted by user[/red]")
            sys.exit(1)

    # Initialize hardware manager
    hardware_manager = HardwareManager()

    # Setup environment
    setup_environment()

    # Initialize task manager and progress
    task_manager = BackgroundTaskManager()
    training_progress = TrainingProgress()

    return hardware_manager, task_manager, training_progress

def get_user_confirmation(prompt):
    """Get yes/no confirmation from user"""
    while True:
        response = input(f"\n{prompt} (y/n): ").lower().strip()
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        console.print("[yellow]Please answer 'y' or 'n'[/yellow]")

def setup_environment():
    """Setup environment and handle Windows-specific configurations"""
    # Disable symlinks warning
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    # Check if running on Windows
    if os.name == 'nt':
        # Check if running as administrator on Windows
        import ctypes
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            is_admin = False

        # Check if Developer Mode is enabled
        dev_mode = False
        try:
            import winreg
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock",
                    0,
                    winreg.KEY_READ | winreg.KEY_WOW64_64KEY
                )
                dev_mode = winreg.QueryValueEx(key, "AllowDevelopmentWithoutDevLicense")[0] == 1
                winreg.CloseKey(key)
            except FileNotFoundError:
                # Registry key doesn't exist, which means Developer Mode was never enabled
                dev_mode = False
            except Exception:
                # Any other registry-related error
                dev_mode = False
        except Exception:
            # winreg import failed or other error
            dev_mode = False

        if not (is_admin or dev_mode):
            console.print("\n[yellow]NOTICE: For better performance on Windows, consider either:")
            console.print("1. Running Python as administrator")
            console.print("2. Enabling Developer Mode in Windows settings:")
            console.print("   Settings -> Privacy & Security -> For Developers -> Developer Mode")
            console.print("\nContinuing with degraded caching performance...\n")
    else:
        # Unix-like systems (Linux, macOS)
        try:
            is_admin = os.getuid() == 0
        except AttributeError:
            is_admin = False

    # Create cache directories if they don't exist
    home = os.path.expanduser("~")
    cache_paths = [
        os.path.join(home, ".cache", "huggingface"),
        os.path.join(home, ".cache", "torch"),
        os.path.join(home, ".cache", "datasets"),
        os.path.join(home, ".cache", "transformers"),
        # Windows-specific paths
        os.path.join(home, "AppData", "Local", "huggingface"),
        os.path.join(home, "AppData", "Local", "torch")
    ]

    for path in cache_paths:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create cache directory {path}: {str(e)}")

def display_parameter_help(param_name):
    """Display help text for each configurable parameter"""
    help_texts = {
        "num_dims": """
        Model dimension size (d_model). Higher values increase model capacity but require more memory.
        Requirements:
        - Must be divisible by 64
        - Recommended ranges:
          • Small models: 512-768
          • Medium models: 1024-1536
          • Large models: 2048+
        Common values: 512, 768, 1024, 1536, 2048
        """,
        "num_heads": """
        Number of attention heads. Should typically divide num_dims evenly.
        - Common values: 8, 16, 32
        - Rule of thumb: num_dims / num_heads should be 64-128
        """,
        "num_kv_heads": """
        Number of key/value heads for grouped-query attention (GQA).
        - Should be <= num_heads and divide num_heads evenly
        - Lower values reduce memory usage
        - Common values: 4, 8
        """,
        "num_layers": """
        Number of transformer layers. More layers = deeper model = more capacity.
        - Small models: 12-24
        - Medium models: 24-36
        - Large models: 32+
        """,
        "max_seq_length": """
        Maximum sequence length for training. Affects memory usage significantly.
        - Longer sequences require more memory
        - Common values: 1024, 2048, 4096
        - Must be <= model's context_len
        """,
        "batch_size": """
        Number of sequences processed in parallel. Affects training speed and memory.
        - Larger values generally give better training stability
        - Must fit in GPU memory along with model
        - Common values: 8-32
        """,
        "learning_rate": """
        Learning rate for optimizer. Critical for training stability.
        - Too high: training may diverge
        - Too low: training may be too slow
        - Common values: 1e-4 to 6e-4
        """,
        "num_epochs": """
        Number of times to process the entire dataset.
        - More epochs = longer training
        - Watch for overfitting
        - Common values: 1-3
        """,
        # LoRA parameters
        "use_lora": """
        Enable Low-Rank Adaptation (LoRA) for efficient fine-tuning.
        - Reduces memory usage and training time
        - Preserves base model weights
        - Recommended for fine-tuning tasks
        """,
        "lora_rank": """
        Rank of LoRA adaptation matrices. Controls capacity vs efficiency trade-off.
        - Lower values = smaller, faster, less expressive
        - Higher values = larger, slower, more expressive
        - Common values: 4, 8, 16, 32
        - Rule of thumb: start small (8) and increase if needed
        """,
        "lora_alpha": """
        Scaling factor for LoRA. Controls contribution of LoRA vs base weights.
        - Higher values = stronger LoRA influence
        - Typically set to lora_rank * 4
        - Common values: 16, 32, 64
        """,
        "lora_target_modules": """
        Which model layers to apply LoRA to.
        Default targets attention query and value projections.
        Options: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        """,

        # MoE parameters
        "use_moe": """
        Enable Mixture of Experts (MoE) for conditional computation.
        - Increases model capacity without proportional compute cost
        - Requires more memory but can improve performance
        """,
        "moe_num_experts": """
        Total number of expert networks in MoE layers.
        - More experts = higher capacity but more memory
        - Common values: 4, 8, 16
        """,
        "moe_active_experts": """
        Number of experts to use per token (top-k routing).
        - Must be <= moe_num_experts
        - Higher values increase computation
        - Common values: 1, 2, 4
        """,
        "use_lossfreebalance": """
        Use Auxiliary-loss-free load balancing for MoE (from DeepSeek paper).
        - Improves expert utilization
        - Reduces training instability
        - Recommended when using MoE
        """,
        "moe_shared_experts": """
        Number of shared experts across all MoE layers.
        - From DeepSeekMoE architecture
        - Helps prevent expert collapse
        - Common values: 0, 1, 2
        """
    }
    return help_texts.get(param_name, "No help available for this parameter.")

def get_user_input_with_help(param_name, default_value, param_type=str, moe_num_experts=None):
    """Get user input with help text and validation"""
    while True:
        console.print(f"\n[cyan]Parameter: [bold]{param_name}[/bold][/cyan]")
        console.print(display_parameter_help(param_name))
        console.print(f"Default value: [green]{default_value}[/green]")

        # Show quick hints based on parameter
        if param_name == "num_dims":
            console.print("[italic]Hint: Value must be divisible by 64 (e.g., 512, 768, 1024)[/italic]")
        elif param_name == "num_heads":
            console.print("[italic]Hint: Value should evenly divide num_dims (typical ratio: num_dims/num_heads = 64-128)[/italic]")
        elif param_name == "lora_rank":
            console.print("[italic]Hint: Lower values are more efficient, higher values are more expressive[/italic]")
        elif param_name == "lora_alpha":
            console.print(f"[italic]Hint: Recommended value for current lora_rank: {default_value * 4}[/italic]")
        elif param_name == "moe_active_experts":
            console.print("[italic]Hint: Should be less than or equal to moe_num_experts[/italic]")

        console.print("Enter value, [yellow]?[/yellow] for help, or [yellow]q[/yellow] to quit: ", end="")

        user_input = input().strip()

        if user_input.lower() == 'q':
            if confirm_quit():
                cleanup_and_exit()
            continue

        if user_input == '?':
            continue

        if user_input == "":
            return default_value

        try:
            value = param_type(user_input)
            if validate_parameter(param_name, value):
                # Additional validation for related parameters
                if param_name == "num_dims":
                    console.print(f"[green]✓ Valid value. This will support {value//64} attention heads per layer.[/green]")
                elif param_name == "moe_active_experts" and moe_num_experts and value > moe_num_experts:
                    console.print(f"[red]moe_active_experts ({value}) cannot be greater than moe_num_experts ({moe_num_experts})[/red]")
                    continue
                return value
        except ValueError:
            console.print(f"[red]Invalid input. Please enter a {param_type.__name__} value.[/red]")

def validate_parameter(param_name, value):
    """Validate parameter values with detailed error messages"""
    validation_rules = {
        "num_dims": {
            "check": lambda x: x > 0 and x % 64 == 0,
            "error": "num_dims must be positive and divisible by 64. Suggested values: 512, 768, 1024, 1536, 2048"
        },
        "num_heads": {
            "check": lambda x: x > 0 and x % 2 == 0,
            "error": "num_heads must be positive and even. Common values: 8, 16, 32"
        },
        "num_kv_heads": {
            "check": lambda x: x > 0,
            "error": "num_kv_heads must be positive and should divide num_heads evenly"
        },
        "num_layers": {
            "check": lambda x: x > 0,
            "error": "num_layers must be positive. Common values: 12, 24, 32"
        },
        "max_seq_length": {
            "check": lambda x: x > 0,
            "error": "max_seq_length must be positive. Common values: 1024, 2048, 4096"
        },
        "batch_size": {
            "check": lambda x: x > 0,
            "error": "batch_size must be positive. Common values: 8, 16, 32"
        },
        "learning_rate": {
            "check": lambda x: 0 < x < 1,
            "error": "learning_rate must be between 0 and 1. Common values: 1e-4 to 6e-4"
        },
        "num_epochs": {
            "check": lambda x: x > 0,
            "error": "num_epochs must be positive"
        },
        # LoRA validation
        "use_lora": {
            "check": lambda x: isinstance(x, bool),
            "error": "use_lora must be True or False"
        },
        "lora_rank": {
            "check": lambda x: x > 0 and x <= 256,
            "error": "lora_rank must be positive and <= 256. Common values: 4, 8, 16, 32"
        },
        "lora_alpha": {
            "check": lambda x: x > 0,
            "error": "lora_alpha must be positive. Typically set to lora_rank * 4"
        },

        # MoE validation
        "use_moe": {
            "check": lambda x: isinstance(x, bool),
            "error": "use_moe must be True or False"
        },
        "moe_num_experts": {
            "check": lambda x: x > 0,
            "error": "moe_num_experts must be positive. Common values: 4, 8, 16"
        },
        "moe_active_experts": {
            "check": lambda x: x > 0,
            "error": "moe_active_experts must be positive and <= moe_num_experts"
        },
        "use_lossfreebalance": {
            "check": lambda x: isinstance(x, bool),
            "error": "use_lossfreebalance must be True or False"
        },
        "moe_shared_experts": {
            "check": lambda x: x >= 0,
            "error": "moe_shared_experts must be >= 0. Common values: 0, 1, 2"
        }
    }

    rule = validation_rules.get(param_name)
    if rule:
        if not rule["check"](value):
            console.print(f"[red]Invalid value: {rule['error']}[/red]")
            return False
    return True

def confirm_quit():
    """Confirm if user wants to quit"""
    return input("Are you sure you want to quit? (y/N): ").lower() == 'y'

def cleanup_and_exit(code=0):
    """Clean up resources and exit"""
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Remove temporary files if they exist
        temp_dirs = ['./temp', './cache']
        for dir_path in temp_dirs:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    except Exception as e:
        console.print(f"[yellow]Warning during cleanup: {str(e)}")
    finally:
        sys.exit(code)

def cleanup_checkpoints():
    """Clean up model checkpoints"""
    checkpoint_dir = "./model_testing"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

def cleanup_cache():
    """Clean up cache files"""
    cache_dir = "./data/dataset_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def cleanup_logs():
    """Clean up log files"""
    log_dir = "./log"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

def get_dataset_splits(dataset_name, dataset_config=None):
    """Get available splits for a dataset"""
    cache_dir = os.path.join("data", "dataset_cache")
    try:
        # Try to get dataset info without downloading
        try:
            info = load_dataset(dataset_name, dataset_config, download_mode='reuse_dataset_if_exists')
            if hasattr(info, 'splits') and info.splits:
                return list(info.splits.keys())
            return []
        except Exception as e:
            error_msg = str(e)
            # Check if the error is about missing config
            if "Config name is missing" in error_msg and "available configs" in error_msg:
                # Extract available configs from the error message
                import re

                # Try to extract configs using a simple pattern
                configs = re.findall(r"'([^']+)'", error_msg)
                configs = [c for c in configs if c]  # Remove empty strings

                if configs and dataset_config in configs:
                    # Try to get splits for the specific config
                    try:
                        info = load_dataset(dataset_name, dataset_config, download_mode='reuse_dataset_if_exists')
                        if hasattr(info, 'splits') and info.splits:
                            return list(info.splits.keys())
                    except Exception:
                        pass
            raise e
    except Exception as e:
        console.print(f"[yellow]Error getting splits: {str(e)}")
        return []

def get_dataset_config(dataset_name):
    """Get available configs for dataset and let user choose one"""
    cache_dir = os.path.join("data", "dataset_cache")
    try:
        console.print(f"\n[yellow]Checking dataset: {dataset_name}")

        # Try to get dataset info without downloading
        try:
            # First check available splits
            splits = get_dataset_splits(dataset_name)
            split_to_use = "train" if "train" in splits else (splits[0] if splits else "train")

            dataset_info = load_dataset(
                dataset_name,
                split=split_to_use,
                cache_dir=cache_dir
            )
            return None, len(dataset_info)

        except Exception as e:
            error_msg = str(e)
            # Check for multiple configurations error
            if "Config name is missing" in error_msg or "multiple configurations" in error_msg.lower():
                # Extract available configs from the error message
                configs = []

                # Try to parse configs from error message
                if "Please pick one among the available configs:" in error_msg:
                    # Extract the config list from the error message
                    import re
                    config_match = re.search(r"available configs: \['([^']+)'(?:, '([^']+)')*\]", error_msg)
                    if config_match:
                        configs = re.findall(r"'([^']+)'(?:, )?|", error_msg)
                        configs = [c for c in configs if c]  # Remove empty strings

                # If we couldn't extract from error message, try to load dataset info
                if not configs:
                    try:
                        info = load_dataset(dataset_name, download_mode='reuse_dataset_if_exists')
                        configs = list(info.builder_configs.keys())
                    except Exception as config_error:
                        console.print(f"[red]Error getting configurations: {str(config_error)}")
                        return None, None

                console.print("\n[yellow]Available configurations:")
                for i, config in enumerate(configs, 1):
                    console.print(f"{i}. {config}")

                choice = get_user_input("\nSelect configuration (number): ", [str(i) for i in range(1, len(configs) + 1)])
                try:
                    selected_config = configs[int(choice) - 1]
                    console.print(f"[green]✓[/green] Selected configuration: {selected_config}")
                    return selected_config, None
                except Exception as choice_error:
                    console.print(f"[yellow]Error selecting configuration: {str(choice_error)}. Using first config.")
                    return configs[0], None  # Default to first config
            else:
                raise e
    except Exception as e:
        console.print(f"[red]Error checking dataset: {str(e)}")
        return None, None

def create_progress():
    """Create a progress bar instance"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=50),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TransferSpeedColumn(),
        console=console
    )

def download_file(args):
    """Worker function to download a single file"""
    url, filename, retry_count = args
    for _ in range(retry_count):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if _ == retry_count - 1:  # Last retry
                print(f"Failed to download {url}: {str(e)}")
    return False

def parallel_download_dataset(dataset_name, dataset_config, num_files, cache_dir, task_manager, progress):
    """Download dataset using parallel workers with background tasks"""
    try:
        # Calculate optimal number of workers based on file count and CPU cores
        num_workers = min(cpu_count(), 4)  # Reduced number of workers for stability

        download_task = progress.create_task(
            f"[cyan]Loading dataset...",
            total=100
        )

        console.print(f"[yellow]Attempting to load dataset: {dataset_name}")
        console.print(f"[yellow]Config: {dataset_config}")

        # Get available splits
        splits = get_dataset_splits(dataset_name, dataset_config)
        if splits:
            console.print(f"[yellow]Available splits: {', '.join(splits)}")
            split_to_use = "train" if "train" in splits else splits[0]
            console.print(f"[yellow]Using split: {split_to_use}")
        else:
            split_to_use = "train"  # Default to train if we can't determine splits

        # Try loading with streaming first
        try:
            dataset_info = load_dataset(
                dataset_name,
                dataset_config,
                split=split_to_use,
                streaming=True,
                cache_dir=cache_dir,
                num_proc=num_workers
            )
            # Convert streaming dataset to regular dataset with specific size
            dataset_info = dataset_info.take(num_files)
            dataset_info = list(dataset_info)
        except Exception as e:
            console.print(f"[yellow]Streaming load failed, trying regular load: {str(e)}")
            # Fall back to regular loading with smaller chunk
            try:
                dataset_info = load_dataset(
                    dataset_name,
                    dataset_config,
                    split=f"{split_to_use}[:{min(num_files, 1000000)}]",  # Load smaller chunk
                    cache_dir=cache_dir,
                    num_proc=num_workers
                )
            except Exception as e2:
                console.print(f"[yellow]Regular load failed: {str(e2)}")
                # Try using the config itself as the split if applicable
                if "Unknown split" in str(e2) and dataset_config:
                    console.print(f"[yellow]Trying to use '{dataset_config}' as the split directly")
                    try:
                        dataset_info = load_dataset(
                            dataset_name,
                            split=dataset_config,
                            streaming=True,
                            cache_dir=cache_dir,
                            num_proc=num_workers
                        )
                        # Convert streaming dataset to regular dataset with specific size
                        dataset_info = dataset_info.take(num_files)
                        dataset_info = list(dataset_info)
                    except Exception as e3:
                        console.print(f"[yellow]All loading attempts failed: {str(e3)}")
                        raise e3
                else:
                    raise e2

        progress.update(download_task, completed=100)
        console.print(f"\n[green]Successfully loaded {len(dataset_info)} samples")
        return dataset_info

    except Exception as e:
        console.print(f"\n[red]Error loading dataset: {str(e)}")
        console.print("[yellow]Falling back to wikitext-2-raw-v1 dataset")

        try:
            return load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="train",
                cache_dir=cache_dir,
                num_proc=num_workers
            )
        except Exception as fallback_error:
            console.print(f"[red]Fallback dataset also failed: {str(fallback_error)}")
            console.print("[red]Available options:")
            console.print("1. Try a different dataset")
            console.print("2. Check your internet connection")
            console.print("3. Clear the cache and try again")

            if input("\nWould you like to try a different dataset? (y/n): ").lower() == 'y':
                new_dataset = input("Enter dataset name (e.g., 'wikitext', 'bookcorpus'): ")
                return parallel_download_dataset(new_dataset, None, num_files, cache_dir, task_manager, progress)
            else:
                cleanup_and_exit()

def process_and_save_text(text: str, filename: str, max_seq_len: int, tokenizer) -> bool:
    """
    Process and save tokenized text to a binary file.

    Args:
        text: Input text to tokenize
        filename: Path to save the tokenized data
        max_seq_len: Maximum sequence length
        tokenizer: The tokenizer instance to use

    Returns:
        bool: True if processing and saving succeeded, False otherwise
    """
    try:
        # Tokenize the text with status indicator
        with console.status("[bold blue]Tokenizing text...", spinner="dots") as status:
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt"
            )["input_ids"][0]

        # Save tokens to binary file
        with open(filename, 'wb') as f:
            # Save number of tokens as uint32
            f.write(struct.pack('I', len(tokens)))
            # Save each token as uint16
            for token in tokens:
                f.write(struct.pack('H', token))
        return True

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return False

def process_chunk(args):
    """Worker function to process a chunk of dataset"""
    chunk, config, tokenizer, text_column = args
    successful = 0
    results = []

    for i, item in enumerate(chunk):
        text = item.get(text_column, "").strip()
        if text:
            filename = os.path.join(config.tokenized_dataset_path, f"sample_{i}.ds")
            if process_and_save_text(text, filename, config.max_seq_len, tokenizer):
                successful += 1
                results.append((i, filename))

    return successful, results

def parallel_process_dataset(dataset, train_config, tokenizer, text_column, task_manager, progress):
    """Process dataset using parallel workers with background tasks and cleanup"""
    cleanup_manager = DataCleanupManager(retention_time=train_config.data_retention_time)
    chunk_size = len(dataset) // task_manager.max_workers + 1
    chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

    process_task = progress.create_task(
        f"[cyan]Processing samples with {task_manager.max_workers} workers",
        total=len(dataset)
    )

    def process_chunk_task(chunk, config, tokenizer, text_column):
        successful = 0
        results = []
        temp_files = []

        for i, item in enumerate(chunk):
            text = item.get(text_column, "").strip()
            if text:
                filename = os.path.join(config.tokenized_dataset_path, f"sample_{i}.ds")
                if process_and_save_text(text, filename, config.max_seq_len):
                    successful += 1
                    results.append((i, filename))
                    temp_files.append(filename)

        # Schedule processed files for cleanup
        for file in temp_files:
            cleanup_manager.schedule_cleanup(file)

        return successful, results

    # Create processing tasks
    processing_nodes = []
    for i, chunk in enumerate(chunks):
        node = task_manager.add_task(
            f"process_chunk_{i}",
            process_chunk_task,
            chunk,
            train_config,
            tokenizer,
            text_column,
            use_process=True
        )
        processing_nodes.append(node)

    return processing_nodes, cleanup_manager

def setup_model(config, train_config):
    model = Transformer(config)

    # Apply LoRA if enabled
    if train_config.use_lora:
        model = add_lora_layers(
            model,
            rank=train_config.lora_rank,
            alpha=train_config.lora_alpha,
            target_modules=train_config.lora_target_modules
        )

        # Freeze base model parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

    return model

def cleanup_old_datasets():
    """Clean up old dataset files and cache"""
    import shutil

    with console.status("[bold yellow]Cleaning up old datasets...") as status:
        # Clean up tokenized datasets
        if os.path.exists("data/test_dataset"):
            shutil.rmtree("data/test_dataset")
            console.print("[green]✓[/green] Cleaned tokenized datasets")

        # Clean up HuggingFace dataset cache
        if os.path.exists("data/dataset_cache"):
            shutil.rmtree("data/dataset_cache")
            console.print("[green]✓[/green] Cleaned dataset cache")

        # Clean up HuggingFace general cache
        cache_path = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            console.print("[green]✓[/green] Cleaned HuggingFace cache")

    console.print("[bold green]Dataset cleanup completed!")

def cleanup_all_datasets():
    """Clean up ALL dataset files and caches from the system"""
    import shutil

    with console.status("[bold yellow]Performing complete dataset cleanup...") as status:
        # Project-specific cleanup
        paths_to_clean = [
            ("Project datasets", "data/test_dataset"),
            ("Project cache", "data/dataset_cache"),
            ("Project models", "model_testing"),
            ("Log files", "log"),
        ]

        # System-wide ML framework caches
        home = os.path.expanduser("~")
        system_paths = [
            ("HuggingFace cache", os.path.join(home, ".cache", "huggingface")),
            ("Torch cache", os.path.join(home, ".cache", "torch")),
            ("HF datasets cache", os.path.join(home, ".cache", "datasets")),
            ("HF transformers cache", os.path.join(home, ".cache", "transformers")),
            # Windows-specific paths
            ("Windows HF cache", os.path.join(home, "AppData", "Local", "huggingface")),
            ("Windows Torch cache", os.path.join(home, "AppData", "Local", "torch")),
        ]

        paths_to_clean.extend(system_paths)

        total_space_freed = 0
        for name, path in paths_to_clean:
            if os.path.exists(path):
                try:
                    size = sum(os.path.getsize(os.path.join(dirpath,filename))
                             for dirpath, dirnames, filenames in os.walk(path)
                             for filename in filenames)
                    shutil.rmtree(path)
                    total_space_freed += size
                    console.print(f"[green]✓[/green] Cleaned {name} ({size/1024/1024:.1f} MB)")
                except Exception as e:
                    console.print(f"[red]✗[/red] Error cleaning {name}: {str(e)}")

        console.print(f"[bold green]Total space freed: {total_space_freed/1024/1024:.1f} MB")

    console.print("[bold green]Complete system cleanup finished!")

class DataCleanupManager:
    def __init__(self, retention_time=3600):  # default 1 hour retention
        self.cleanup_queue = Queue()
        self.retention_time = retention_time
        self.running = True
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def _cleanup_worker(self):
        while self.running:
            try:
                # Get (path, timestamp) tuple from queue
                path, timestamp = self.cleanup_queue.get(timeout=60)

                # Skip if file was already removed
                if not os.path.exists(path):
                    continue

                current_time = time.time()
                if current_time - timestamp >= self.retention_time:
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                        console.print(f"[dim]Cleaned up: {path}[/dim]")
                    except Exception as e:
                        console.print(f"[red]Error cleaning up {path}: {str(e)}[/red]")
                else:
                    # If not ready for cleanup, put back in queue
                    self.cleanup_queue.put((path, timestamp))
            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]Cleanup worker error: {str(e)}[/red]")

    def schedule_cleanup(self, path):
        """Schedule a file or directory for cleanup"""
        self.cleanup_queue.put((path, time.time()))

    def shutdown(self):
        """Shutdown the cleanup manager"""
        self.running = False
        self.cleanup_thread.join(timeout=5)

# TrainerConfig is imported from trainer.py

def get_user_input(prompt, valid_options=None):
    """Helper function to get stable user input"""
    while True:
        try:
            print(prompt, end='', flush=True)  # Use print instead of input
            user_input = sys.stdin.readline().strip()
            if valid_options is None or user_input in valid_options:
                return user_input
        except KeyboardInterrupt:
            raise
        except:
            continue

def subset_dataset(dataset, num_samples):
    """Create a subset of the dataset with the specified number of samples"""
    if num_samples >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset))[:num_samples]
    return torch.utils.data.Subset(dataset, indices)

def start_training(model_config, train_config, dataset, task_manager, progress, console=None):
    """Start the training process"""
    console = console or default_console
    try:
        # Initialize model
        model = setup_model(model_config, train_config)

        # Apply dataset subsetting if needed
        if hasattr(train_config, 'use_subset') and train_config.use_subset:
            orig_size = len(dataset)
            dataset = subset_dataset(dataset, train_config.target_samples)
            console.print(f"\n[yellow]Dataset optimization:[/yellow]")
            console.print(f"→ Original size: {orig_size:,} samples")
            console.print(f"→ Reduced size: {len(dataset):,} samples")
            console.print(f"→ Reduction ratio: {len(dataset)/orig_size*100:.1f}%")

        # Make sure progress is properly initialized
        if progress and not hasattr(progress, 'stop'):
            # Create a new TrainingProgress instance with enhanced visualization
            progress = TrainingProgress()
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

        # Set padding token and strategy
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'right'

        # Initialize tokenizer and data loader
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'right'

        # Create data loader with the dataset
        data_loader = CustomDataLoader(
            config=train_config,
            dataset=dataset,
            tokenizer=tokenizer
        )

        # Initialize trainer with console
        trainer = Trainer(train_config, model, tokenizer, console=console)

        console.print("\n[green]Starting training process...")
        trainer.train(data_loader, progress)

        console.print("\n[green]Training completed successfully!")

    except Exception as e:
        console.print(f"\nError during training: {str(e)}")
        raise e

def get_model_config(tokenizer):
    """Get model configuration from user input"""
    print("\nModel Configuration:")
    num_dims = int(get_user_input_with_help("num_dims", default_value=512, param_type=int))
    num_heads = int(get_user_input_with_help("num_heads", default_value=16, param_type=int))
    num_layers = int(get_user_input_with_help("num_layers", default_value=32, param_type=int))
    context_len = int(get_user_input_with_help("context_len", default_value=1536, param_type=int))
    use_flash = get_user_input("Use Flash Attention? (Y/n): ").lower() != 'n'
    use_moe = get_user_input("Use Mixture of Experts? (y/N): ").lower() == 'y'

    moe_config = {}
    if use_moe:
        # First get num_experts, then pass it to active_experts validation
        moe_num_experts = int(get_user_input_with_help("moe_num_experts", default_value=2, param_type=int))
        moe_active_experts = int(get_user_input_with_help("moe_active_experts", default_value=2, param_type=int, moe_num_experts=moe_num_experts))
        use_lossfreebalance = get_user_input("Use loss-free balancing? (y/N): ").lower() == 'y'

        moe_config.update({
            'moe_num_experts': moe_num_experts,
            'moe_active_experts': moe_active_experts,
            'use_lossfreebalance': use_lossfreebalance
        })

    return ModelConfig(
        vocab_size=tokenizer.vocab_size,
        num_dims=num_dims,
        num_heads=num_heads,
        num_kv_heads=num_heads // 4,  # Default to num_heads/4
        num_layers=num_layers,
        ffn_hidden_dims=num_dims * 4,
        context_len=context_len,
        use_cache=True,
        use_flash=use_flash,
        use_moe=use_moe,
        **moe_config
    )

def get_training_config(tokenizer, subset_mode=False):
    """Get training configuration with optional subset mode"""
    config = TrainerConfig(
        vocab_size=tokenizer.vocab_size,
        num_epochs=1 if subset_mode else 3,
        use_ddp=False,
        use_moe=False,
        use_lossfreebalance=False,
        use_flash=False,
        gradient_checkpointing=False,
        # Add memory management parameters
        gpu_memory_utilization=0.85,
        enable_memory_tracking=True,
        expandable_segments=True,
        empty_cache_freq=100,
        auto_adjust_batch=True,
        min_batch_size=1,
        # Configure subset mode if enabled
        use_subset=subset_mode,
        target_samples=20 if subset_mode else None,
        # Configure checkpoint usage
        use_epoch_checkpoints=False,

        # Adjust parameters for subset mode
        batch_size=4 if subset_mode else 1,
        learning_rate=5e-4 if subset_mode else 1e-3,
        max_seq_len=128 if subset_mode else 1024,
        use_compile=False if subset_mode else True,
        use_dtype="float16" if subset_mode else "bfloat16"
    )
    return config

def load_existing_config(tokenizer, dataset_size=None, config_file='last_config.json'):
    """Load existing configuration from file"""
    try:
        # Load configuration file
        with open(config_file, 'r') as f:
            saved_config = json.load(f)

        # Get subsetting parameters
        use_subset = saved_config.get('train_config', {}).get('use_subset', False)
        target_samples = saved_config.get('train_config', {}).get('target_samples')

        # Debug print for loading configuration
        console.print(f"[yellow]Debug: Loading configuration with use_subset={use_subset}, target_samples={target_samples}")

        if use_subset and target_samples:
            console.print(f"[yellow]Loading configuration with dataset subset: {target_samples} samples")
            dataset_size = target_samples

        # Extract training config parameters
        train_config = TrainerConfig(
            vocab_size=tokenizer.vocab_size,
            num_epochs=saved_config.get('train_config', {}).get('num_epochs', 1),
            use_subset=saved_config.get('train_config', {}).get('use_subset', False),  # Explicitly set use_subset
            target_samples=saved_config.get('train_config', {}).get('target_samples', None),  # Explicitly set target_samples
            use_ddp=saved_config.get('train_config', {}).get('use_ddp', False),
            use_moe=saved_config.get('model_config', {}).get('use_moe', False),
            use_lossfreebalance=saved_config.get('model_config', {}).get('use_lossfreebalance', False),
            clean_cuda_cache=saved_config.get('train_config', {}).get('clean_cuda_cache', True),
            use_compile=saved_config.get('train_config', {}).get('use_compile', False),
            use_dtype=saved_config.get('train_config', {}).get('use_dtype', "float16"),
            use_flash=saved_config.get('model_config', {}).get('use_flash', False),
            seed=saved_config.get('train_config', {}).get('seed', 1),
            max_seq_len=saved_config.get('train_config', {}).get('max_seq_len', 512),
            batch_size=saved_config.get('train_config', {}).get('batch_size', 4),
            accumulation_steps=saved_config.get('train_config', {}).get('accumulation_steps', 4),
            weight_decay=saved_config.get('train_config', {}).get('weight_decay', 0.1),
            warmup_ratio=saved_config.get('train_config', {}).get('warmup_ratio', 0.01),
            learning_rate=saved_config.get('train_config', {}).get('learning_rate', 1e-3),
            betas=saved_config.get('train_config', {}).get('betas', (0.90, 0.95)),
            update_rate=saved_config.get('train_config', {}).get('update_rate', 1e-5),
            val_ratio=saved_config.get('train_config', {}).get('val_ratio', 0.005),
            steps_for_eval=saved_config.get('train_config', {}).get('steps_for_eval', 20),
            eval_interval=saved_config.get('train_config', {}).get('eval_interval', 100),
            log_interval=saved_config.get('train_config', {}).get('log_interval', 1),
            checkpoints_frequency=saved_config.get('train_config', {}).get('checkpoints_frequency', 1),
            path_to_checkpoints=saved_config.get('train_config', {}).get('path_to_checkpoints', "./model_testing"),
            tokenized_dataset_path=saved_config.get('train_config', {}).get('tokenized_dataset_path', "fineweb-edu_tok-10BT"),
            eval_log_file=saved_config.get('train_config', {}).get('eval_log_file', "log/eval.txt"),
            use_lora=saved_config.get('train_config', {}).get('use_lora', False),
            lora_rank=saved_config.get('train_config', {}).get('lora_rank', 8),
            lora_alpha=saved_config.get('train_config', {}).get('lora_alpha', 32),
            lora_target_modules=saved_config.get('train_config', {}).get('lora_target_modules', ["q_proj", "v_proj"]),
            use_epoch_checkpoints=saved_config.get('train_config', {}).get('use_epoch_checkpoints', False),
            num_workers=saved_config.get('train_config', {}).get('num_workers', 4),
            queue_size=saved_config.get('train_config', {}).get('queue_size', 8),
            gradient_checkpointing=saved_config.get('train_config', {}).get('gradient_checkpointing', False),
            gpu_memory_utilization=saved_config.get('train_config', {}).get('gpu_memory_utilization', 0.85),
            enable_memory_tracking=saved_config.get('train_config', {}).get('enable_memory_tracking', True),
            expandable_segments=saved_config.get('train_config', {}).get('expandable_segments', True),
            empty_cache_freq=saved_config.get('train_config', {}).get('empty_cache_freq', 100),
            auto_adjust_batch=saved_config.get('train_config', {}).get('auto_adjust_batch', True),
            min_batch_size=saved_config.get('train_config', {}).get('min_batch_size', 1)
        )

        # Create model config with loaded parameters
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            num_dims=saved_config['model_config']['num_dims'],
            num_heads=saved_config['model_config']['num_heads'],
            num_kv_heads=saved_config['model_config']['num_kv_heads'],
            num_layers=saved_config['model_config']['num_layers'],
            context_len=saved_config['model_config']['context_len'],
            use_flash=saved_config['model_config']['use_flash'],
            use_moe=saved_config['model_config']['use_moe'],
            moe_num_experts=saved_config['model_config'].get('moe_num_experts', 2),
            moe_active_experts=saved_config['model_config'].get('moe_active_experts', 2),
            use_lossfreebalance=saved_config['model_config']['use_lossfreebalance'],
            ffn_hidden_dims=saved_config['model_config']['num_dims'] * 4,
            use_cache=False
        )

        # Set dataset attributes after creating the TrainerConfig instance
        train_config.dataset_name = saved_config.get('train_config', {}).get('dataset_name', None)
        train_config.dataset_config = saved_config.get('train_config', {}).get('dataset_config', None)
        train_config.dataset_text_field = saved_config.get('train_config', {}).get('dataset_text_field', None)

        console.print(f"[yellow]Loaded dataset info: {train_config.dataset_name}, {train_config.dataset_config}")

        return model_config, train_config

    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")

def get_training_configs(console=None):
    """Get model and training configurations from user input"""
    console = console or default_console
    global model_config  # Move global declaration to the start of the function

    try:
        # First, check if there's a previous config
        print("\nConfiguration options:")
        print("1. Use previous configuration from last_config.json")
        print("2. Create new configuration")
        print("")

        choice = get_user_input("Select option (1-2): ", ['1', '2'])

        if choice == "1":
            try:
                with open('last_config.json', 'r') as f:
                    saved_config = json.load(f)

                print("\nLoading previous configuration:")
                print(json.dumps(saved_config, indent=2))

                confirm = get_user_input("\nUse this configuration? (Y/n): ").lower()
                if confirm != 'n':
                    # Initialize tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
                    tokenizer.pad_token = tokenizer.eos_token

                    # Convert saved config to ModelConfig and TrainerConfig objects
                    model_config = ModelConfig(
                        vocab_size=tokenizer.vocab_size,
                        num_dims=saved_config['model_config']['num_dims'],
                        num_heads=saved_config['model_config']['num_heads'],
                        num_kv_heads=saved_config['model_config']['num_kv_heads'],
                        num_layers=saved_config['model_config']['num_layers'],
                        context_len=saved_config['model_config']['context_len'],
                        use_flash=saved_config['model_config']['use_flash'],
                        use_moe=saved_config['model_config']['use_moe'],
                        moe_num_experts=saved_config['model_config'].get('moe_num_experts', 2),
                        moe_active_experts=saved_config['model_config'].get('moe_active_experts', 2),
                        use_lossfreebalance=saved_config['model_config']['use_lossfreebalance'],
                        ffn_hidden_dims=saved_config['model_config']['num_dims'] * 4,  # Standard multiplier
                        use_cache=False
                    )

                    train_config = TrainerConfig(
                        vocab_size=tokenizer.vocab_size,
                        batch_size=saved_config['train_config']['batch_size'],
                        num_epochs=saved_config['train_config']['num_epochs'],
                        learning_rate=saved_config['train_config']['learning_rate'],
                        max_seq_len=saved_config['train_config']['max_seq_len'],
                        use_ddp=saved_config['train_config']['use_ddp'],
                        use_compile=saved_config['train_config']['use_compile'],
                        use_dtype=saved_config['train_config']['use_dtype'],
                        use_moe=saved_config['model_config']['use_moe'],
                        use_lossfreebalance=saved_config['model_config']['use_lossfreebalance'],
                        use_flash=saved_config['model_config'].get('use_flash', False)  # Add this line
                    )

                    # Load dataset
                    print("\nLoading dataset: wikitext")
                    dataset = load_dataset(
                        "wikitext",
                        "wikitext-2-raw-v1",
                        split="train",
                        cache_dir="data/dataset_cache"
                    )
                    print("Raw dataset features:", dataset.features)

                    # Filter out empty texts
                    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
                    print("Dataset size after filtering:", len(dataset))

                    # Apply subset if configured
                    if train_config.use_subset and train_config.target_samples > 0:
                        if train_config.target_samples < len(dataset):
                            dataset = dataset.select(range(train_config.target_samples))
                            print(f"Using subset of {train_config.target_samples} samples")
                    print("First non-empty item:", dataset[0])

                    # Map dataset to create input_ids
                    # Process dataset with proper tokenization
                    def tokenize_function(examples):
                        return tokenizer(
                            examples["text"],
                            truncation=True,
                            padding='max_length',
                            max_length=train_config.max_seq_len,
                            return_tensors=None  # Don't convert to tensors yet
                        )

                    # Process dataset
                    dataset = dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=4,
                        remove_columns=dataset.column_names,
                    )
                    print("\nTokenized dataset features:", dataset.features)
                    print("First tokenized item:", dataset[0])
                    print(f"Successfully loaded {len(dataset)} samples")

                    return model_config, train_config, dataset
            except FileNotFoundError:
                console.print("No previous configuration found. Creating new configuration...")
            except Exception as e:
                console.print(f"Error loading previous configuration: {str(e)}")
                console.print("Creating new configuration...")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        tokenizer.pad_token = tokenizer.eos_token

        # Dataset selection
        print("\nDataset Configuration:")
        print("1. Use default dataset (wikitext-2-raw-v1)")
        print("2. Use custom HuggingFace dataset")
        print("")

        choice = get_user_input("Select option (1-2): ", ['1', '2'])

        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"

        if choice == "2":
            dataset_name = get_user_input("\nEnter dataset name (e.g., 'wikitext', 'bookcorpus'): ")
            dataset_config, _ = get_dataset_config(dataset_name)

        # Load the dataset
        print(f"\nLoading dataset: {dataset_name}")
        if dataset_config:
            print(f"Config: {dataset_config}")

        try:
            # Special case for nvidia/OpenCodeReasoning dataset
            if dataset_name == "nvidia/OpenCodeReasoning":
                print(f"Detected nvidia/OpenCodeReasoning dataset, using special handling")
                # For this dataset, the split name is the same as the config name
                dataset = load_dataset(
                    dataset_name,
                    dataset_config,
                    split=dataset_config,  # Use config name as split
                    cache_dir="data/dataset_cache"
                )
                print(f"Successfully loaded {len(dataset)} samples using {dataset_config} as both config and split")
            else:
                # Standard handling for other datasets
                # Get available splits
                splits = get_dataset_splits(dataset_name, dataset_config)
                if splits:
                    print(f"Available splits: {', '.join(splits)}")
                    split_to_use = "train" if "train" in splits else splits[0]
                    print(f"Using split: {split_to_use}")
                else:
                    # If we couldn't determine splits, try loading with the config name as the split
                    # This is a common pattern for some datasets
                    if dataset_config:
                        split_to_use = dataset_config
                        print(f"No splits found, trying to use config name as split: {split_to_use}")
                    else:
                        split_to_use = "train"  # Default to train if we can't determine splits
                        print(f"No splits found, using default 'train' split")

                try:
                    dataset = load_dataset(
                        dataset_name,
                        dataset_config,
                        split=split_to_use,
                        cache_dir="data/dataset_cache"
                    )
                except Exception as split_error:
                    if "Unknown split" in str(split_error) and dataset_config:
                        # Try using the config itself as the split
                        print(f"Error with split '{split_to_use}', trying to use '{dataset_config}' as the split directly")
                        dataset = load_dataset(
                            dataset_name,
                            split=dataset_config,
                            cache_dir="data/dataset_cache"
                        )
                    else:
                        raise split_error

            print(f"Successfully loaded {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Falling back to wikitext-2-raw-v1 dataset")
            dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="train",
                cache_dir="data/dataset_cache"
            )

        # Get configurations with proper subset mode
        # Default to False for is_quick_run since target_hours is not defined in this context
        is_quick_run = False
        model_config = get_model_config(tokenizer)
        train_config = get_training_config(tokenizer, subset_mode=is_quick_run)

        return model_config, train_config, dataset

    except Exception as e:
        console.print(f"\nError in configuration: {str(e)}")
        cleanup_and_exit()

def display_config_summary(model_config, train_config, console=None):
    """Display a summary of the configurations"""
    console = console or default_console

    model_summary = f"""
    Model Configuration:
    - Dimensions: {model_config.num_dims}
    - Attention Heads: {model_config.num_heads}
    - KV Heads: {model_config.num_kv_heads}
    - Layers: {model_config.num_layers}
    - Context Length: {model_config.context_len}
    - Flash Attention: {model_config.use_flash}
    - MoE: {model_config.use_moe}
    """

    training_summary = f"""
    Training Configuration:
    - Batch Size: {train_config.batch_size}
    - Learning Rate: {train_config.learning_rate}
    - Epochs: {train_config.num_epochs}
    - Max Sequence Length: {train_config.max_seq_len}
    - DDP: {train_config.use_ddp}
    - Compile: {train_config.use_compile}
    - Data Type: {train_config.use_dtype}
    - Dataset Path: {train_config.tokenized_dataset_path}
    - Use Epoch Checkpoints: {train_config.use_epoch_checkpoints}
    """

    console.print(Panel(model_summary, title="[bold blue]Model Configuration"))
    console.print(Panel(training_summary, title="[bold green]Training Configuration"))

def save_config(model_config, train_config):
    """Save configurations to a file"""
    # Ensure use_subset and target_samples are properly set
    if not hasattr(train_config, 'use_subset'):
        train_config.use_subset = False
    if not hasattr(train_config, 'target_samples'):
        train_config.target_samples = None

    # Calculate effective dataset size
    effective_size = train_config.target_samples if train_config.use_subset else None

    # Debug print for saving configuration
    console.print(f"[yellow]Debug: Saving configuration with use_subset={train_config.use_subset}, target_samples={train_config.target_samples}")

    config_dict = {
        'model_config': {
            'vocab_size': model_config.vocab_size,
            'num_dims': model_config.num_dims,
            'num_heads': model_config.num_heads,
            'num_kv_heads': model_config.num_kv_heads,
            'num_layers': model_config.num_layers,
            'effective_dataset_size': effective_size,  # Save effective dataset size
            'ffn_hidden_dims': model_config.ffn_hidden_dims,
            'context_len': model_config.context_len,
            'use_cache': model_config.use_cache,
            'use_flash': model_config.use_flash,
            'use_moe': model_config.use_moe,
            'moe_num_experts': model_config.moe_num_experts,
            'moe_active_experts': model_config.moe_active_experts,
            'moe_eps': model_config.moe_eps,
            'moe_aux_loss_coef': model_config.moe_aux_loss_coef,
            'moe_shared_experts': model_config.moe_shared_experts,
            'use_lossfreebalance': model_config.use_lossfreebalance,
            'rmsnorm_eps': model_config.rmsnorm_eps,
            'rope_theta': model_config.rope_theta,
            'activation_fn': model_config.activation_fn,
            'norm_type': model_config.norm_type
        },
        'train_config': {
            'vocab_size': train_config.vocab_size,
            'num_epochs': train_config.num_epochs,
            'use_subset': train_config.use_subset,  # Save subset flag
            'target_samples': train_config.target_samples,  # Save target samples
            'use_ddp': train_config.use_ddp,
            'use_moe': train_config.use_moe,
            'use_lossfreebalance': train_config.use_lossfreebalance,
            'clean_cuda_cache': train_config.clean_cuda_cache,
            'use_compile': train_config.use_compile,
            'use_dtype': train_config.use_dtype,
            'use_flash': train_config.use_flash,
            'seed': train_config.seed,
            'max_seq_len': train_config.max_seq_len,
            'batch_size': train_config.batch_size,
            'accumulation_steps': train_config.accumulation_steps,
            'weight_decay': train_config.weight_decay,
            'warmup_ratio': train_config.warmup_ratio,
            'learning_rate': train_config.learning_rate,
            'betas': train_config.betas,
            'update_rate': train_config.update_rate,
            'val_ratio': train_config.val_ratio,
            'steps_for_eval': train_config.steps_for_eval,
            'eval_interval': train_config.eval_interval,
            'checkpoints_frequency': train_config.checkpoints_frequency,
            'path_to_checkpoints': train_config.path_to_checkpoints,
            'tokenized_dataset_path': train_config.tokenized_dataset_path,
            'eval_log_file': train_config.eval_log_file,
            'use_lora': train_config.use_lora,
            'lora_rank': train_config.lora_rank,
            'lora_alpha': train_config.lora_alpha,
            'lora_target_modules': train_config.lora_target_modules,
            'num_workers': train_config.num_workers,
            'queue_size': train_config.queue_size,
            'gpu_memory_utilization': train_config.gpu_memory_utilization,
            'enable_memory_tracking': train_config.enable_memory_tracking,
            'expandable_segments': train_config.expandable_segments,
            'gradient_checkpointing': train_config.gradient_checkpointing,
            'empty_cache_freq': train_config.empty_cache_freq,
            'auto_adjust_batch': train_config.auto_adjust_batch,
            'min_batch_size': train_config.min_batch_size,
            # Save dataset information
            'dataset_name': getattr(train_config, 'dataset_name', None),
            'dataset_config': getattr(train_config, 'dataset_config', None),
            'dataset_text_field': getattr(train_config, 'dataset_text_field', None),
            # Save checkpoint usage setting
            'use_epoch_checkpoints': getattr(train_config, 'use_epoch_checkpoints', False)
        }
    }

    with open('last_config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_config(tokenizer):
    """Load configuration from file with proper defaults"""
    try:
        with open('last_config.json', 'r') as f:
            saved_config = json.load(f)

        # Create ModelConfig with all required parameters
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            num_dims=saved_config['model_config']['num_dims'],
            num_heads=saved_config['model_config']['num_heads'],
            num_kv_heads=saved_config['model_config']['num_kv_heads'],
            num_layers=saved_config['model_config']['num_layers'],
            ffn_hidden_dims=saved_config['model_config']['ffn_hidden_dims'],
            context_len=saved_config['model_config']['context_len'],
            use_cache=saved_config['model_config']['use_cache'],
            use_flash=saved_config['model_config']['use_flash'],
            use_moe=saved_config['model_config']['use_moe'],
            moe_num_experts=saved_config['model_config']['moe_num_experts'],
            moe_active_experts=saved_config['model_config']['moe_active_experts'],
            moe_eps=saved_config['model_config'].get('moe_eps', 1e-6),
            moe_aux_loss_coef=saved_config['model_config'].get('moe_aux_loss_coef', 0.01),
            moe_shared_experts=saved_config['model_config'].get('moe_shared_experts', 0),
            use_lossfreebalance=saved_config['model_config'].get('use_lossfreebalance', False),
            rmsnorm_eps=saved_config['model_config'].get('rmsnorm_eps', 1e-6),
            rope_theta=saved_config['model_config'].get('rope_theta', 1e5),
            activation_fn=saved_config['model_config'].get('activation_fn', 'silu'),
            norm_type=saved_config['model_config'].get('norm_type', 'rmsnorm')
        )

        # Create TrainerConfig with all required parameters
        train_config = TrainerConfig(
            vocab_size=tokenizer.vocab_size,
            **saved_config['train_config']
        )

        return model_config, train_config

    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")

def estimate_training_time(config, dataset_size, gpu_memory, gpu_specs, target_hours=None, hardware_manager=None):
    """Estimate training time based on configuration and hardware"""
    # Calculate basic processing metrics
    tokens_per_batch = config.batch_size * config.max_seq_len
    batches_per_epoch = dataset_size // config.batch_size
    total_batches = batches_per_epoch * config.num_epochs

    # Calculate base processing speed with GPU scaling
    base_tokens_per_second = (gpu_memory / (1024**3)) * 2000  # Base speed for GPU memory in GB
    speedup_multiplier = 1.0

    # Apply feature-based speed adjustments
    if config.use_flash and hardware_manager and hardware_manager.flash_attention_available:
        speedup_multiplier *= 1.3
    if hardware_manager and hardware_manager.get_gpu_compute_capability() >= 7.0:
        speedup_multiplier *= 1.2
    if config.gradient_checkpointing:
        speedup_multiplier *= 0.7

    # Calculate expected processing speed
    effective_tokens_per_second = base_tokens_per_second * speedup_multiplier

    # If target time is specified, calculate optimal dataset size
    optimal_samples = None
    if target_hours is not None:
        target_seconds = target_hours * 3600
        overhead_seconds = 5  # Fixed startup overhead
        compute_seconds = max(0.001, target_seconds - overhead_seconds)

        # For ultra-short runs, use a fixed minimal dataset
        if target_hours < 0.01:  # Less than 36 seconds
            optimal_samples = 20  # Ultra minimal dataset
        else:
            # For regular training, calculate optimal dataset size based on target time
            # Scale the dataset size to match the target time
            target_tokens = compute_seconds * effective_tokens_per_second

            # For small datasets, always use the full dataset
            if dataset_size <= 1000:
                # For very small datasets, use the full dataset
                optimal_samples = dataset_size
                console.print(f"[yellow]Dataset is small ({dataset_size} samples), using full dataset for quality[/yellow]")
            else:
                # Calculate a reasonable subset size based on target time
                # For longer training times, use a smaller percentage of the dataset
                if target_hours < 0.1:  # Less than 6 minutes
                    percentage = 0.05  # 5% of dataset
                elif target_hours < 0.5:  # Less than 30 minutes
                    percentage = 0.1  # 10% of dataset
                elif target_hours < 1.0:  # Less than 1 hour
                    percentage = 0.2  # 20% of dataset
                elif target_hours < 3.0:  # Less than 3 hours
                    percentage = 0.3  # 30% of dataset
                elif target_hours < 6.0:  # Less than 6 hours
                    percentage = 0.5  # 50% of dataset
                elif target_hours < 12.0:  # Less than 12 hours
                    percentage = 0.7  # 70% of dataset
                else:  # 12+ hours
                    percentage = 0.9  # 90% of dataset

                # Calculate the optimal sample count based on percentage
                optimal_samples = int(max(100, min(dataset_size, dataset_size * percentage)))

        # Update batches based on optimal samples
        batches_per_epoch = (optimal_samples + config.batch_size - 1) // config.batch_size
        total_batches = batches_per_epoch * config.num_epochs

    # Calculate total processing time
    total_tokens = total_batches * tokens_per_batch
    processing_seconds = total_tokens / effective_tokens_per_second
    estimated_seconds = processing_seconds + (5 if target_hours and target_hours < 0.01 else 0)  # Add overhead for ultra-short runs

    # Return both the estimated time and the optimal samples count
    return estimated_seconds / 3600, optimal_samples  # Convert to hours

def optimize_for_time_target(train_config, model_config, dataset_size, target_hours, hardware_manager):
    """Optimize configuration to meet target training time with comprehensive parameter analysis"""
    # Create copies of the original configs for comparison
    import copy
    original_train_config = copy.deepcopy(train_config)
    original_model_config = copy.deepcopy(model_config)
    console.print("\n[bold cyan]Starting Comprehensive Configuration Optimization[/bold cyan]")
    console.print(f"Target training time: {target_hours:.6f} hours")

    # Get hardware capabilities
    gpu_memory = hardware_manager.gpu_memory
    gpu_specs = hardware_manager.get_gpu_specs()
    max_batch_size = hardware_manager.get_max_batch_size(train_config)

    console.print("\n[bold]Hardware Capabilities:[/bold]")
    console.print(f"→ Maximum batch size: {max_batch_size}")
    console.print(f"→ Available GPU memory: {gpu_memory / (1024**3):.1f}GB")
    console.print(f"→ Flash Attention support: {hardware_manager.flash_attention_available}")
    console.print(f"→ GPU Compute Capability: {hardware_manager.get_gpu_compute_capability()}")

    # Get initial time estimate and optimal sample count
    current_estimate, optimal_samples = estimate_training_time(train_config, dataset_size, gpu_memory, gpu_specs, target_hours, hardware_manager)

    # For small datasets, use the full dataset and increase epochs
    if dataset_size <= 1000:
        # Use the full dataset for small datasets
        train_config.use_subset = False
        train_config.target_samples = dataset_size
        console.print(f"\n[yellow]Dataset is small ({dataset_size} samples), using full dataset for quality[/yellow]")

        # Calculate how many epochs we can fit in the target time
        tokens_per_epoch = dataset_size * train_config.max_seq_len
        tokens_per_second = 32000  # Estimated processing speed
        seconds_per_epoch = tokens_per_epoch / tokens_per_second

        # Calculate maximum epochs that fit in target time (with some margin)
        max_epochs = int((target_hours * 3600 * 0.8) / seconds_per_epoch)

        # Set a reasonable number of epochs (at least 3, at most 20)
        train_config.num_epochs = max(3, min(max_epochs, 20))
        console.print(f"[yellow]Increasing epochs to {train_config.num_epochs} for better quality.[/yellow]")
    else:
        # For larger datasets, use subsetting based on the optimal sample count
        train_config.use_subset = True
        train_config.target_samples = optimal_samples

    # For very quick testing (less than 1 second)
    if target_hours < 0.01:  # Less than 36 seconds
        console.print("\n[bold yellow]Optimizing for ultra-fast testing mode (sub-second)[/bold yellow]")
        console.print("\n[bold]Speed Optimizations:[/bold]")

        # Display dataset subsetting information
        console.print(f"→ Dataset: Using minimal subset ({train_config.target_samples} samples)")
        console.print("→ Device transfers: Minimized for speed")
        console.print("→ Memory tracking: Disabled for faster execution")

        # Ultra-minimal model for very short runs
        model_config.num_layers = 1
        model_config.num_dims = 64
        model_config.ffn_hidden_dims = 256
        model_config.context_len = 128  # Very short context
        model_config.num_heads = 4
        model_config.num_kv_heads = 2
        model_config.ffn_hidden_dims = 512
        console.print("→ Minimized model architecture:")
        console.print(f"  • Layers: {model_config.num_layers}")
        console.print(f"  • Dimensions: {model_config.num_dims}")
        console.print(f"  • Heads: {model_config.num_heads}")

        # Ultra-minimal training settings
        train_config.num_epochs = 1
        train_config.max_seq_len = 64  # Even shorter sequences
        train_config.batch_size = 4
        train_config.accumulation_steps = 1
        train_config.gradient_checkpointing = False  # Disable to speed up
        train_config.empty_cache_freq = 1000  # Reduce cache clearing
        train_config.clean_cuda_cache = True
        train_config.enable_memory_tracking = False  # Disable for speed
        train_config.use_compile = False  # Skip compilation for faster startup
        train_config.use_flash = hardware_manager.flash_attention_available
        train_config.eval_interval = 1  # No intermediate evals
    else:
        # Display dataset optimization information
        console.print(f"\n[bold yellow]Dataset Optimization:[/bold yellow]")

        if dataset_size <= 1000:
            console.print(f"→ Using full dataset: {dataset_size:,} samples (small dataset)")
            console.print(f"→ Training for {train_config.num_epochs} epochs to improve quality")
            console.print(f"→ Estimated tokens per epoch: {dataset_size * train_config.max_seq_len:,}")
            console.print(f"→ Total tokens: {dataset_size * train_config.max_seq_len * train_config.num_epochs:,}")
        else:
            # Calculate the percentage of the dataset being used
            percentage_used = optimal_samples / dataset_size * 100
            console.print(f"→ Using optimized dataset size: {optimal_samples:,} samples (from full size of {dataset_size:,})")
            console.print(f"→ Reduction ratio: {percentage_used:.1f}% of full dataset")
            console.print(f"→ Estimated tokens: {optimal_samples * train_config.max_seq_len:,}")

        console.print(f"→ Target training time: {target_hours:.2f} hours")
        # For regular training, use more appropriate settings
        # Adjust model parameters based on target time to better utilize the available time
        if target_hours < 0.5:  # Less than 30 minutes - use smaller model
            model_config.num_layers = int(max(6, model_config.num_layers // 2))
            model_config.num_heads = int(max(8, model_config.num_heads // 2))
            # Ensure num_dims is divisible by num_heads
            head_dim = 64  # Standard head dimension
            model_config.num_dims = model_config.num_heads * head_dim

            # Ensure num_kv_heads divides num_heads evenly
            # Find the largest divisor of num_heads that is <= 8
            possible_kv_heads = [i for i in [1, 2, 4, 8] if model_config.num_heads % i == 0]
            model_config.num_kv_heads = max(possible_kv_heads) if possible_kv_heads else 1

            # Set training parameters for quick training
            train_config.num_epochs = max(5, train_config.num_epochs // 2)
            train_config.batch_size = 4
            train_config.accumulation_steps = 1
            train_config.learning_rate = 5e-4
            train_config.lora_rank = 4

            # Additional optimization parameters for quick training
            train_config.warmup_ratio = 0.01  # Short warmup
            train_config.weight_decay = 0.05  # Reduced weight decay
            train_config.eval_interval = 10  # More frequent evaluation
            train_config.empty_cache_freq = 200  # Less frequent cache clearing

            # Adjust model-specific parameters
            model_config.ffn_hidden_dims = model_config.num_dims * 3  # Smaller FFN
            model_config.rmsnorm_eps = 1e-5  # Standard normalization epsilon

            console.print("\n[bold]Model Configuration for Short Training Time:[/bold]")
            console.print(f"→ Using smaller model architecture:")
            console.print(f"  • Layers: {model_config.num_layers}")
            console.print(f"  • Dimensions: {model_config.num_dims}")
            console.print(f"  • Heads: {model_config.num_heads}")
            console.print(f"  • Epochs: {train_config.num_epochs}")
            console.print(f"  • Learning Rate: {train_config.learning_rate}")

        elif target_hours < 2.0:  # Less than 2 hours - use medium model
            model_config.num_layers = int(max(12, int(model_config.num_layers // 1.5)))
            model_config.num_heads = int(max(12, int(model_config.num_heads // 1.5)))
            # Ensure num_dims is divisible by num_heads
            head_dim = 64  # Standard head dimension
            model_config.num_dims = model_config.num_heads * head_dim

            # Ensure num_kv_heads divides num_heads evenly
            # Find the largest divisor of num_heads that is <= 8
            possible_kv_heads = [i for i in [1, 2, 4, 8] if model_config.num_heads % i == 0]
            model_config.num_kv_heads = max(possible_kv_heads) if possible_kv_heads else 1

            # Set training parameters for medium-length training
            train_config.num_epochs = max(10, train_config.num_epochs)
            train_config.batch_size = 4
            train_config.accumulation_steps = 2
            train_config.learning_rate = 3e-4
            train_config.lora_rank = 8
            train_config.max_seq_len = min(768, model_config.context_len)

            # Additional optimization parameters for medium training
            train_config.warmup_ratio = 0.015  # Medium warmup
            train_config.weight_decay = 0.08  # Medium weight decay
            train_config.eval_interval = 25  # Medium evaluation frequency
            train_config.empty_cache_freq = 100  # Standard cache clearing

            # Adjust model-specific parameters
            model_config.ffn_hidden_dims = model_config.num_dims * 3  # Medium FFN
            model_config.rmsnorm_eps = 1e-6  # Standard normalization epsilon
            model_config.rope_theta = 10000.0  # Standard RoPE theta

            console.print("\n[bold]Model Configuration for Medium Training Time:[/bold]")
            console.print(f"→ Using medium-sized model architecture:")
            console.print(f"  • Layers: {model_config.num_layers}")
            console.print(f"  • Dimensions: {model_config.num_dims}")
            console.print(f"  • Heads: {model_config.num_heads}")
            console.print(f"  • Epochs: {train_config.num_epochs}")
            console.print(f"  • Sequence Length: {train_config.max_seq_len}")

        elif target_hours < 5.0:  # Less than 5 hours - use standard model with extended training
            # Adjust model architecture for better quality
            model_config.num_layers = int(24)  # Standard size
            model_config.num_heads = int(16)  # Standard size
            head_dim = 48  # Standard head dimension
            model_config.num_dims = model_config.num_heads * head_dim

            # Ensure num_kv_heads divides num_heads evenly
            # Find the largest divisor of num_heads that is <= 8
            possible_kv_heads = [i for i in [1, 2, 4, 8] if model_config.num_heads % i == 0]
            model_config.num_kv_heads = max(possible_kv_heads) if possible_kv_heads else 1

            # Set training parameters for extended training
            train_config.num_epochs = max(30, train_config.num_epochs * 2)  # Increase epochs significantly
            train_config.batch_size = 4
            train_config.accumulation_steps = 4  # Use gradient accumulation
            train_config.learning_rate = 2e-4
            train_config.lora_rank = 16  # Higher LoRA rank for better adaptation
            train_config.max_seq_len = min(1024, model_config.context_len)  # Longer sequences
            train_config.gradient_checkpointing = True  # Enable gradient checkpointing for memory efficiency

            # Additional optimization parameters
            train_config.warmup_ratio = 0.02  # Increased warmup for stability
            train_config.weight_decay = 0.1  # Standard weight decay
            train_config.eval_interval = 50  # Less frequent evaluation for speed
            train_config.empty_cache_freq = 50  # More frequent cache clearing

            # Adjust model-specific parameters
            model_config.ffn_hidden_dims = model_config.num_dims * 4  # Standard FFN size
            model_config.rmsnorm_eps = 1e-6  # Standard normalization epsilon

            console.print("\n[bold]Model Configuration for Standard Training Time:[/bold]")
            console.print(f"→ Using standard model architecture with extended training:")
            console.print(f"  • Layers: {model_config.num_layers}")
            console.print(f"  • Dimensions: {model_config.num_dims}")
            console.print(f"  • Heads: {model_config.num_heads}")
            console.print(f"  • Epochs: {train_config.num_epochs}")
            console.print(f"  • Sequence Length: {train_config.max_seq_len}")
            console.print(f"  • LoRA Rank: {train_config.lora_rank}")

        else:  # 5+ hours - use larger model with comprehensive training
            # Increase model size for longer training times, but be more conservative
            model_config.num_layers = int(min(32, int(model_config.num_layers * 1.2)))
            model_config.num_heads = int(min(20, int(model_config.num_heads * 1.1)))
            # Ensure num_dims is divisible by num_heads
            head_dim = 64  # Larger head dimension for extended training
            model_config.num_dims = model_config.num_heads * head_dim

            # Ensure num_kv_heads divides num_heads evenly
            # Find the largest divisor of num_heads that is <= 8
            possible_kv_heads = [i for i in [1, 2, 4, 8] if model_config.num_heads % i == 0]
            model_config.num_kv_heads = max(possible_kv_heads) if possible_kv_heads else 1

            # Set training parameters for comprehensive training - more reasonable values
            train_config.num_epochs = max(30, train_config.num_epochs * 2)  # Increase epochs but not excessively
            train_config.batch_size = 2  # Smaller batch size for larger model
            train_config.accumulation_steps = 4  # Moderate gradient accumulation
            train_config.learning_rate = 2e-4  # Moderate learning rate
            train_config.lora_rank = 24  # Higher LoRA rank for better adaptation
            train_config.max_seq_len = min(1024, model_config.context_len)  # Longer sequence length
            train_config.gradient_checkpointing = True  # Enable gradient checkpointing
            train_config.warmup_ratio = 0.02  # Moderate warmup for stability

            # Advanced optimization parameters
            train_config.weight_decay = 0.1  # Standard weight decay
            train_config.eval_interval = 100  # Less frequent evaluation for speed
            train_config.empty_cache_freq = 25  # More frequent cache clearing for stability
            train_config.betas = [0.9, 0.95]  # Optimizer momentum parameters

            # Advanced model parameters
            model_config.ffn_hidden_dims = model_config.num_dims * 4  # Standard FFN size
            model_config.rmsnorm_eps = 1e-6  # Standard normalization epsilon
            model_config.rope_theta = 10000.0  # Standard RoPE theta

            # Consider using mixture-of-experts for very long training (8+ hours)
            if target_hours > 8.0 and gpu_memory > 12 * (1024**3):  # Only for 12GB+ GPUs
                model_config.use_moe = True
                model_config.moe_num_experts = 4
                model_config.moe_active_experts = 2
                model_config.moe_eps = 1e-6
                model_config.moe_aux_loss_coef = 0.01
                console.print("→ Enabling Mixture-of-Experts for enhanced model capacity")

            console.print("\n[bold]Model Configuration for Extended Training:[/bold]")
            console.print(f"→ Using larger model architecture with comprehensive training:")
            console.print(f"  • Layers: {model_config.num_layers}")
            console.print(f"  • Dimensions: {model_config.num_dims}")
            console.print(f"  • Heads: {model_config.num_heads}")
            console.print(f"  • Epochs: {train_config.num_epochs}")
            console.print(f"  • Sequence Length: {train_config.max_seq_len}")
            console.print(f"  • LoRA Rank: {train_config.lora_rank}")

        # Set appropriate training parameters
        train_config.log_interval = 10  # Regular logging
        train_config.checkpoints_frequency = 1  # Save at end of training
        train_config.use_moe = model_config.use_moe  # Use MoE if configured in model
        train_config.use_lora = True  # Enable LoRA for better training
        train_config.gpu_memory_utilization = 0.85  # Standard memory utilization
        train_config.expandable_segments = True  # Enable expandable segments
        train_config.enable_memory_tracking = True  # Enable memory tracking

        # Standard optimization settings
        train_config.learning_rate = 1e-3  # Standard learning rate
        train_config.warmup_ratio = 0.01  # Standard warmup
        train_config.weight_decay = 0.1  # Standard weight decay

        # Performance monitoring
        train_config.eval_steps = 20  # Regular evaluation
        train_config.val_ratio = 0.005  # Standard validation ratio

        console.print("\n[bold]Training Parameters:[/bold]")
        console.print(f"→ Sequence length: {train_config.max_seq_len}")
        console.print(f"→ Batch size: {train_config.batch_size}")
        console.print(f"→ Flash Attention: {train_config.use_flash}")
        console.print(f"→ Memory tracking: {train_config.enable_memory_tracking}")
        console.print(f"→ Gradient checkpointing: {train_config.gradient_checkpointing}")
        console.print(f"→ LoRA: {train_config.use_lora} (rank={train_config.lora_rank})")

        # Memory management for standard training
        if hardware_manager.gpu_available:
            gpu_memory_gb = hardware_manager.gpu_memory / (1024**3)
            if gpu_memory_gb < 8:  # Less than 8GB
                train_config.batch_size = 2  # Reduce batch size
                train_config.gradient_checkpointing = True  # Enable gradient checkpointing
                console.print("[yellow]Limited GPU memory detected - using memory optimizations")
            else:
                train_config.batch_size = min(8, train_config.batch_size)  # Cap at 8

            console.print(f"→ Using batch size: {train_config.batch_size}")
            console.print(f"→ Available GPU memory: {gpu_memory_gb:.1f}GB")

        # Standard training time estimation
        tokens_per_batch = train_config.batch_size * train_config.max_seq_len
        batches_per_epoch = max(1, dataset_size // train_config.batch_size)
        total_batches = batches_per_epoch * train_config.num_epochs

        # More accurate speed estimation based on model size and training parameters
        # Calculate model complexity factor - this has the biggest impact on speed
        # Base reference model: 24 layers, 768 dims, 16 heads
        layers_factor = model_config.num_layers / 24
        dims_factor = (model_config.num_dims / 768) ** 2  # Square because impact is non-linear
        heads_factor = model_config.num_heads / 16

        # Combined model architecture factor
        model_size_factor = layers_factor * dims_factor * heads_factor

        # Calculate sequence length factor - longer sequences are slower
        seq_len_factor = (train_config.max_seq_len / 512) ** 1.5  # Non-linear impact

        # Calculate LoRA complexity factor - higher rank means more computation
        lora_factor = 0.8 + (0.2 * train_config.lora_rank / 16)  # LoRA has less impact

        # Calculate epochs factor - linear scaling with epochs
        epochs_factor = train_config.num_epochs / 20

        # Calculate total complexity factor with appropriate weighting
        # Model size and sequence length have the biggest impact
        complexity_factor = (model_size_factor * 0.6) + (seq_len_factor * 0.3) + (lora_factor * 0.1)
        complexity_factor = complexity_factor * epochs_factor

        # Base tokens per second for reference hardware (GTX 1070 8GB)
        # A standard model (24/768/16) processes ~3000 tokens/sec on this GPU
        base_tokens_per_second = 3000 * (gpu_memory / (8 * 1024**3))  # Scale by available VRAM

        # Apply complexity factor to base speed
        tokens_per_second = base_tokens_per_second / complexity_factor

        # Apply additional adjustments
        if train_config.use_flash: tokens_per_second *= 1.3
        if hardware_manager and hardware_manager.get_gpu_compute_capability() >= 7.0: tokens_per_second *= 1.2
        if train_config.gradient_checkpointing: tokens_per_second *= 0.7

        # Batch size efficiency curve (smaller batches are less efficient)
        effective_batch_size = train_config.batch_size * train_config.accumulation_steps
        batch_efficiency = 0.5 + (0.5 * min(1.0, effective_batch_size / 8))
        tokens_per_second *= batch_efficiency

        # Final adjustment to ensure reasonable estimates
        tokens_per_second = max(50, min(10000, tokens_per_second))  # Reasonable bounds

        # Account for minimal overhead
        overhead_seconds = 5  # Startup time
        compute_seconds = (total_batches * tokens_per_batch) / tokens_per_second
        total_seconds = overhead_seconds + compute_seconds
        estimated_hours = total_seconds / 3600

        console.print("\n[bold yellow]Training Time Estimation:[/bold yellow]")
        console.print(f"→ Startup overhead: {overhead_seconds:.1f} seconds")
        console.print(f"→ Computation time: {compute_seconds:.1f} seconds")
        console.print(f"→ Total estimated time: {total_seconds:.1f} seconds")
        console.print(f"→ Processing speed: {tokens_per_second:,.0f} tokens/second")

        console.print("\n[bold]Processing Details:[/bold]")
        console.print(f"→ Tokens per batch: {tokens_per_batch:,}")
        console.print(f"→ Batches per epoch: {batches_per_epoch:,}")
        console.print(f"→ Total tokens: {total_batches * tokens_per_batch:,}")
        console.print(f"→ Estimated time: {estimated_hours:.3f} hours ({total_seconds:.1f} seconds)")

        return train_config, model_config

    # Regular optimization logic continues here...
    current_estimate, _ = estimate_training_time(train_config, dataset_size, gpu_memory, gpu_specs)
    changes = []

    console.print("\n[bold]Current Configuration Analysis:[/bold]")
    console.print(f"→ Current estimated training time: {current_estimate:.1f} hours")
    console.print(f"→ Target training time: {target_hours:.1f} hours")

    # Calculate adjustment factor
    time_ratio = current_estimate / target_hours

    if time_ratio < 0.9:  # If we're training too fast
        console.print("\n[bold yellow]Note: Current training time is significantly shorter than target[/bold yellow]")
        console.print("Adjusting parameters to extend training duration while maximizing quality...")

        # 1. Increase epochs significantly to reach target time
        target_epochs = min(100, int(train_config.num_epochs * (target_hours / current_estimate)))
        console.print(f"→ Increasing epochs from {train_config.num_epochs} to {target_epochs}")
        train_config.num_epochs = target_epochs
        changes.append("Increased number of epochs for better training")

        # 2. Optimize model architecture for quality
        new_layers = min(48, int(model_config.num_layers * 1.5))  # Increase layers more conservatively
        console.print(f"→ Increasing model layers from {model_config.num_layers} to {new_layers}")
        model_config.num_layers = new_layers

        # Increase model width for better capacity
        new_dims = min(2048, model_config.num_dims * 2)
        console.print(f"→ Increasing model dimensions from {model_config.num_dims} to {new_dims}")
        model_config.num_dims = new_dims
        changes.append("Increased model capacity (layers and dimensions)")

        # 3. Enable and optimize LoRA
        train_config.use_lora = True
        train_config.lora_rank = 32  # Higher rank for better adaptation
        train_config.lora_alpha = 64  # Increased alpha for stronger updates
        train_config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Target more modules
        console.print("→ Enhanced LoRA configuration for better adaptation")
        console.print(f"  • Rank: {train_config.lora_rank}")
        console.print(f"  • Alpha: {train_config.lora_alpha}")
        console.print(f"  • Target modules: {train_config.lora_target_modules}")
        changes.append("Enhanced LoRA configuration")

        # 4. Enable and optimize MoE
        model_config.use_moe = True
        model_config.moe_num_experts = 16  # More experts for better specialization
        model_config.moe_active_experts = 4  # Increase active experts
        model_config.use_lossfreebalance = True
        console.print("→ Enhanced Mixture of Experts configuration")
        console.print(f"  • Number of experts: {model_config.moe_num_experts}")
        console.print(f"  • Active experts: {model_config.moe_active_experts}")
        console.print("  • Using loss-free balancing")
        changes.append("Enhanced MoE configuration")

        # 5. Optimize training parameters for stability
        train_config.learning_rate = 2e-4  # Lower learning rate for stability
        train_config.warmup_ratio = 0.1  # Longer warmup
        train_config.weight_decay = 0.2  # Increased regularization
        train_config.accumulation_steps = min(32, train_config.accumulation_steps * 2)
        console.print("→ Optimized training parameters for stability")
        console.print(f"  • Learning rate: {train_config.learning_rate}")
        console.print(f"  • Warmup ratio: {train_config.warmup_ratio}")
        console.print(f"  • Weight decay: {train_config.weight_decay}")
        console.print(f"  • Accumulation steps: {train_config.accumulation_steps}")
        changes.append("Optimized training parameters")

        # 6. Memory optimizations for larger model
        if gpu_memory < 16 * (1024**3):  # Less than 16GB
            train_config.gradient_checkpointing = True
            train_config.enable_memory_tracking = True
            train_config.empty_cache_freq = 50
            console.print("→ Enabled memory optimizations for larger model")
            changes.append("Enabled memory optimizations")

        # 7. Adjust evaluation frequency for better monitoring
        train_config.eval_interval = max(50, train_config.eval_interval // 2)
        train_config.steps_for_eval = max(10, train_config.steps_for_eval // 2)
        console.print(f"→ Increased evaluation frequency (interval: {train_config.eval_interval})")
        changes.append("Increased evaluation frequency")

    # Final estimation and summary
    final_estimate, _ = estimate_training_time(train_config, dataset_size, gpu_memory, gpu_specs)

    # Create a dictionary of original values for comparison
    original_values = {
        "Model Architecture": {
            "Layers": original_model_config.num_layers,
            "Dimensions": original_model_config.num_dims,
            "Heads": original_model_config.num_heads,
            "FFN Hidden Dims": original_model_config.ffn_hidden_dims,
            "KV Heads": original_model_config.num_kv_heads,
            "MoE": original_model_config.use_moe,
        },
        "Training Parameters": {
            "Epochs": original_train_config.num_epochs,
            "Sequence Length": original_train_config.max_seq_len,
            "Batch Size": original_train_config.batch_size,
            "Accumulation Steps": original_train_config.accumulation_steps,
            "Learning Rate": original_train_config.learning_rate,
            "LoRA Rank": original_train_config.lora_rank,
            "Warmup Ratio": original_train_config.warmup_ratio,
            "Weight Decay": original_train_config.weight_decay,
            "Gradient Checkpointing": original_train_config.gradient_checkpointing,
        }
    }

    # Create a dictionary of new values
    new_values = {
        "Model Architecture": {
            "Layers": model_config.num_layers,
            "Dimensions": model_config.num_dims,
            "Heads": model_config.num_heads,
            "FFN Hidden Dims": model_config.ffn_hidden_dims,
            "KV Heads": model_config.num_kv_heads,
            "MoE": model_config.use_moe,
        },
        "Training Parameters": {
            "Epochs": train_config.num_epochs,
            "Sequence Length": train_config.max_seq_len,
            "Batch Size": train_config.batch_size,
            "Accumulation Steps": train_config.accumulation_steps,
            "Learning Rate": train_config.learning_rate,
            "LoRA Rank": train_config.lora_rank,
            "Warmup Ratio": train_config.warmup_ratio,
            "Weight Decay": train_config.weight_decay,
            "Gradient Checkpointing": train_config.gradient_checkpointing,
        }
    }

    console.print("\n[bold]Optimization Summary:[/bold]")
    for change in changes:
        console.print(f"→ {change}")

    console.print("\n[bold cyan]Configuration Changes:[/bold cyan]")

    for category, params in new_values.items():
        console.print(f"\n[bold yellow]{category}:[/bold yellow]")
        for param_name, new_value in params.items():
            old_value = original_values[category][param_name]
            if new_value != old_value:
                console.print(f"→ {param_name}: [red]{old_value}[/red] → [green]{new_value}[/green]")
            else:
                console.print(f"→ {param_name}: {new_value} (unchanged)")

    # Display dataset information prominently
    if train_config.use_subset and train_config.target_samples is not None:
        percentage_used = train_config.target_samples / dataset_size * 100
        console.print(f"\n[bold yellow]Dataset Configuration:[/bold yellow]")
        console.print(f"→ Using optimized subset: {train_config.target_samples:,} samples out of {dataset_size:,}")
        console.print(f"→ Reduction ratio: {percentage_used:.1f}% of full dataset")
        console.print(f"→ Estimated tokens: {train_config.target_samples * train_config.max_seq_len:,}")
        console.print(f"→ Target training time: {target_hours:.2f} hours")
    else:
        console.print(f"\n[bold yellow]Dataset Configuration:[/bold yellow]")
        console.print(f"→ Using full dataset: {dataset_size:,} samples")
        console.print(f"→ Estimated tokens: {dataset_size * train_config.max_seq_len:,}")
        console.print(f"→ Target training time: {target_hours:.2f} hours")

    console.print(f"\n→ New estimated training time: {final_estimate:.1f} hours")

    # Ask if user wants to optimize for quality
    quality_choice = get_user_confirmation("\nWould you like to further optimize for quality? (This may increase training time)")

    if quality_choice:
        # Enhance model for quality
        console.print("\n[bold cyan]Enhancing Configuration for Quality:[/bold cyan]")

        # 1. Increase model capacity
        old_layers = model_config.num_layers
        model_config.num_layers = min(48, int(model_config.num_layers * 1.5))
        console.print(f"→ Increasing model layers: [red]{old_layers}[/red] → [green]{model_config.num_layers}[/green]")

        old_dims = model_config.num_dims
        model_config.num_dims = min(1536, int(model_config.num_dims * 1.25))
        # Ensure dimensions are divisible by heads
        model_config.num_dims = model_config.num_heads * (model_config.num_dims // model_config.num_heads)
        console.print(f"→ Increasing model dimensions: [red]{old_dims}[/red] → [green]{model_config.num_dims}[/green]")

        # 2. Enhance LoRA configuration
        old_rank = train_config.lora_rank
        train_config.lora_rank = min(64, train_config.lora_rank * 2)
        console.print(f"→ Increasing LoRA rank: [red]{old_rank}[/red] → [green]{train_config.lora_rank}[/green]")

        # 3. Increase sequence length for better context
        old_seq_len = train_config.max_seq_len
        train_config.max_seq_len = min(2048, train_config.max_seq_len * 2)
        console.print(f"→ Increasing sequence length: [red]{old_seq_len}[/red] → [green]{train_config.max_seq_len}[/green]")

        # 4. Adjust training parameters for stability
        old_lr = train_config.learning_rate
        train_config.learning_rate = train_config.learning_rate * 0.75  # Lower learning rate for stability
        console.print(f"→ Adjusting learning rate: [red]{old_lr}[/red] → [green]{train_config.learning_rate}[/green]")

        old_warmup = train_config.warmup_ratio
        train_config.warmup_ratio = min(0.1, train_config.warmup_ratio * 2)  # Longer warmup
        console.print(f"→ Increasing warmup ratio: [red]{old_warmup}[/red] → [green]{train_config.warmup_ratio}[/green]")

        # 5. Enable memory optimizations if needed
        if not train_config.gradient_checkpointing:
            train_config.gradient_checkpointing = True
            console.print("→ Enabling gradient checkpointing for memory efficiency")

        # Recalculate training time
        enhanced_estimate, _ = estimate_training_time(train_config, dataset_size, gpu_memory, gpu_specs)
        console.print(f"\n→ New estimated training time with quality enhancements: [bold]{enhanced_estimate:.1f} hours[/bold]")
        console.print(f"→ Quality-enhanced configuration will take approximately {enhanced_estimate/final_estimate:.1f}x longer")

        # Ask for confirmation of quality enhancements
        apply_quality = get_user_confirmation("Apply these quality enhancements?")
        if not apply_quality:
            # Revert to previous configuration
            console.print("[yellow]Reverting to standard optimized configuration[/yellow]")
            # Restore the values we changed
            model_config.num_layers = old_layers
            model_config.num_dims = old_dims
            train_config.lora_rank = old_rank
            train_config.max_seq_len = old_seq_len
            train_config.learning_rate = old_lr
            train_config.warmup_ratio = old_warmup
            if not original_train_config.gradient_checkpointing:
                train_config.gradient_checkpointing = False

    return train_config, model_config

# Import the configuration editor
from config_editor import edit_config_table

def manage_dataset_and_config(current_model_config, current_train_config, tokenizer):
    """Manage dataset and configuration settings"""
    dataset_size = None  # Track dataset size for subsetting

    while True:
        console.print("\n[bold cyan]Dataset and Configuration Management[/bold cyan]")
        console.print("1. Change dataset")
        console.print("2. Load existing configuration")
        console.print("3. Create new configuration")
        console.print("4. Save current configuration")
        console.print("5. Optimize for training time")
        console.print("6. Edit configuration table")
        console.print("7. Continue with current settings")
        console.print("8. Exit")

        if hasattr(current_train_config, 'use_subset') and current_train_config.use_subset:
            console.print(f"\n[yellow]Current dataset: Using {current_train_config.target_samples} samples (subset mode)")

        choice = get_user_input("\nSelect option (1-8): ", ['1', '2', '3', '4', '5', '6', '7', '8'])

        if choice == '1':
            try:
                # Show dataset options
                console.print("\n[bold cyan]Dataset Selection[/bold cyan]")
                console.print("1. Default dataset (wikitext-2-raw-v1)")
                console.print("2. Custom Hugging Face dataset by name")
                console.print("3. Hugging Face dataset by URL/ID")

                dataset_choice = get_user_input("\nSelect dataset option (1-3): ", ['1', '2', '3'])

                dataset_name = "wikitext"
                dataset_config = "wikitext-2-raw-v1"

                if dataset_choice == "1":
                    # Use default dataset
                    console.print("[yellow]Using default dataset: wikitext-2-raw-v1")

                elif dataset_choice == "2":
                    # Custom dataset by name
                    dataset_name = get_user_input("\nEnter dataset name (e.g., 'wikitext', 'bookcorpus'): ")
                    dataset_config, dataset_size_info = get_dataset_config(dataset_name)
                    console.print(f"[yellow]Using dataset: {dataset_name} with config: {dataset_config if dataset_config else 'default'}")

                elif dataset_choice == "3":
                    # Dataset by URL/ID
                    dataset_id = get_user_input("\nEnter Hugging Face dataset ID (e.g., 'HuggingFaceTB/cosmopedia'): ")
                    dataset_name = dataset_id
                    dataset_config = None
                    console.print(f"[yellow]Using dataset from: {dataset_id}")

                # Load the full dataset
                console.print(f"\n[yellow]Loading dataset: {dataset_name}")
                if dataset_config:
                    console.print(f"[yellow]Config: {dataset_config}")

                try:
                    # Debug information
                    console.print(f"[yellow]Debug: dataset_name={dataset_name}, dataset_config={dataset_config}")

                    # Special case for nvidia/OpenCodeReasoning dataset
                    if dataset_name == "nvidia/OpenCodeReasoning":
                        console.print(f"[yellow]Detected nvidia/OpenCodeReasoning dataset, using special handling")
                        # For this dataset, the split name is the same as the config name
                        full_dataset = load_dataset(
                            dataset_name,
                            dataset_config,
                            split=dataset_config,  # Use config name as split
                            cache_dir=current_train_config.tokenized_dataset_path
                        )
                        console.print(f"[green]Successfully loaded {len(full_dataset)} samples using {dataset_config} as both config and split")
                    else:
                        # Standard handling for other datasets
                        full_dataset = load_dataset(
                            dataset_name,
                            dataset_config,
                            split="train",
                            cache_dir=current_train_config.tokenized_dataset_path
                        )
                    console.print(f"[green]✓[/green] Dataset loaded successfully")

                    # Check if the dataset has the expected text field
                    sample_item = full_dataset[0]
                    if 'text' not in sample_item:
                        # Try to find a suitable text field
                        text_fields = [field for field in sample_item.keys()
                                      if isinstance(sample_item[field], str) and len(sample_item[field]) > 50]

                        if text_fields:
                            text_field = text_fields[0]
                            console.print(f"[yellow]Dataset doesn't have 'text' field. Using '{text_field}' field instead.")
                            # Map the dataset to use the identified field as 'text'
                            full_dataset = full_dataset.map(lambda x: {'text': x[text_field]})
                        else:
                            console.print(f"[red]Dataset doesn't have a suitable text field. Available fields: {list(sample_item.keys())}")
                            raise ValueError("Dataset doesn't have a suitable text field for training")

                    console.print(f"[yellow]Dataset size: {len(full_dataset)} samples")
                    console.print(f"[yellow]Sample text: {full_dataset[0]['text'][:100]}...")

                except Exception as e:
                    console.print(f"[red]Error loading specified dataset: {str(e)}")
                    console.print("[yellow]Falling back to default dataset (wikitext-2-raw-v1)")
                    full_dataset = load_dataset(
                        "wikitext",
                        "wikitext-2-raw-v1",
                        split="train",
                        cache_dir=current_train_config.tokenized_dataset_path
                    )

                # Apply subsetting if configured
                if hasattr(current_train_config, 'use_subset') and current_train_config.use_subset and current_train_config.target_samples is not None:
                    if current_train_config.target_samples < len(full_dataset):
                        dataset = full_dataset.select(range(current_train_config.target_samples))
                        dataset_size = current_train_config.target_samples
                        console.print(f"[green]✓[/green] Loaded dataset with {len(full_dataset)} samples, using subset of {dataset_size} samples")
                    else:
                        dataset = full_dataset
                        dataset_size = len(dataset)
                        console.print(f"[yellow]Warning: Requested subset size {current_train_config.target_samples} is larger than dataset size {len(full_dataset)}")
                        console.print(f"[green]✓[/green] Loaded dataset with {dataset_size} samples")
                else:
                    dataset = full_dataset
                    dataset_size = len(dataset)
                    console.print(f"[green]✓[/green] Loaded dataset with {dataset_size} samples")

                return dataset, current_model_config, current_train_config, dataset_size
            except Exception as e:
                console.print(f"[red]Error loading dataset: {str(e)}")

        elif choice == '2':
            try:
                model_config, train_config = load_existing_config(tokenizer)
                console.print("[green]✓[/green] Configuration loaded successfully")
                return None, model_config, train_config, dataset_size
            except Exception as e:
                console.print(f"[red]Error loading configuration: {str(e)}")
                continue

        elif choice == '3':
            try:
                model_config = get_model_config(tokenizer)
                train_config = get_training_config(tokenizer)
                console.print("[green]✓[/green] New configuration created")
                return None, model_config, train_config, None
            except Exception as e:
                console.print(f"[red]Error creating configuration: {str(e)}")

        elif choice == '4':
            try:
                save_config(current_model_config, current_train_config)
                console.print("[green]✓[/green] Configuration saved successfully")
            except Exception as e:
                console.print(f"[red]Error saving configuration: {str(e)}")

        elif choice == '5':
            try:
                # Get target training time from user
                while True:
                    target_hours = float(get_user_input(
                        "\nEnter desired training time in hours (e.g., 24.0): "
                    ))
                    if target_hours > 0:
                        break
                    console.print("[red]Please enter a positive number[/red]")

                # Initialize hardware manager if not already done
                hardware_manager = HardwareManager()

                # Get dataset size and create subset capability
                # For ultra-short training, use minimal dataset
                if target_hours < 0.01:  # Less than 36 seconds
                    target_samples = MAX_ULTRA_SHORT_SAMPLES  # Use constant for quick runs
                    current_train_config.use_subset = True
                    current_train_config.target_samples = target_samples
                    dataset_size = target_samples
                else:
                    # For regular training, get full dataset
                    # Ask user for dataset selection
                    console.print("\n[bold cyan]Dataset Selection for Optimization[/bold cyan]")
                    console.print("1. Default dataset (wikitext-2-raw-v1)")
                    console.print("2. Custom Hugging Face dataset by name")
                    console.print("3. Hugging Face dataset by URL/ID")

                    dataset_choice = get_user_input("\nSelect dataset option (1-3): ", ['1', '2', '3'])

                    dataset_name = "wikitext"
                    dataset_config = "wikitext-2-raw-v1"

                    if dataset_choice == "1":
                        # Use default dataset
                        console.print("[yellow]Using default dataset: wikitext-2-raw-v1")

                    elif dataset_choice == "2":
                        # Custom dataset by name
                        dataset_name = get_user_input("\nEnter dataset name (e.g., 'wikitext', 'bookcorpus'): ")
                        dataset_config, dataset_size_info = get_dataset_config(dataset_name)
                        console.print(f"[yellow]Using dataset: {dataset_name} with config: {dataset_config if dataset_config else 'default'}")

                    elif dataset_choice == "3":
                        # Dataset by URL/ID
                        dataset_id = get_user_input("\nEnter Hugging Face dataset ID (e.g., 'HuggingFaceTB/cosmopedia'): ")
                        dataset_name = dataset_id
                        dataset_config = None
                        console.print(f"[yellow]Using dataset from: {dataset_id}")

                    # Load the full dataset
                    console.print(f"\n[yellow]Loading dataset: {dataset_name}")
                    if dataset_config:
                        console.print(f"[yellow]Config: {dataset_config}")

                    try:
                        # Special case for nvidia/OpenCodeReasoning dataset
                        if dataset_name == "nvidia/OpenCodeReasoning":
                            console.print(f"[yellow]Detected nvidia/OpenCodeReasoning dataset, using special handling")
                            # For this dataset, the split name is the same as the config name
                            full_dataset = load_dataset(
                                dataset_name,
                                dataset_config,
                                split=dataset_config,  # Use config name as split
                                cache_dir=current_train_config.tokenized_dataset_path
                            )
                            console.print(f"[green]Successfully loaded {len(full_dataset)} samples using {dataset_config} as both config and split")
                        else:
                            # Standard handling for other datasets
                            # Get available splits
                            splits = get_dataset_splits(dataset_name, dataset_config)
                            if splits:
                                console.print(f"[yellow]Available splits: {', '.join(splits)}")
                                split_to_use = "train" if "train" in splits else splits[0]
                                console.print(f"[yellow]Using split: {split_to_use}")
                            else:
                                # If we couldn't determine splits, try loading with the config name as the split
                                # This is a common pattern for some datasets
                                if dataset_config:
                                    split_to_use = dataset_config
                                    console.print(f"[yellow]No splits found, trying to use config name as split: {split_to_use}")
                                else:
                                    split_to_use = "train"  # Default to train if we can't determine splits
                                    console.print(f"[yellow]No splits found, using default 'train' split")

                            try:
                                full_dataset = load_dataset(
                                    dataset_name,
                                    dataset_config,
                                    split=split_to_use,
                                    cache_dir=current_train_config.tokenized_dataset_path
                                )
                            except Exception as split_error:
                                if "Unknown split" in str(split_error) and dataset_config:
                                    # Try using the config itself as the split
                                    console.print(f"[yellow]Error with split '{split_to_use}', trying to use '{dataset_config}' as the split directly")
                                    full_dataset = load_dataset(
                                        dataset_name,
                                        split=dataset_config,
                                        cache_dir=current_train_config.tokenized_dataset_path
                                    )
                                else:
                                    raise split_error

                        # Check if the dataset has the expected text field
                        sample_item = full_dataset[0]
                        if 'text' not in sample_item:
                            # Try to find a suitable text field
                            text_fields = [field for field in sample_item.keys()
                                          if isinstance(sample_item[field], str) and len(sample_item[field]) > 50]

                            if text_fields:
                                text_field = text_fields[0]
                                console.print(f"[yellow]Dataset doesn't have 'text' field. Using '{text_field}' field instead.")
                                # Map the dataset to use the identified field as 'text'
                                full_dataset = full_dataset.map(lambda x: {'text': x[text_field]})
                            else:
                                console.print(f"[red]Dataset doesn't have a suitable text field. Available fields: {list(sample_item.keys())}")
                                raise ValueError("Dataset doesn't have a suitable text field for training")

                    except Exception as e:
                        console.print(f"[red]Error loading specified dataset: {str(e)}")
                        console.print("[yellow]Falling back to default dataset (wikitext-2-raw-v1)")
                        full_dataset = load_dataset(
                            "wikitext",
                            "wikitext-2-raw-v1",
                            split="train",
                            cache_dir=current_train_config.tokenized_dataset_path
                        )

                    dataset_size = len(full_dataset)
                    console.print(f"[green]✓[/green] Dataset loaded with {dataset_size} samples")

                    # Store dataset information in the training configuration
                    current_train_config.dataset_name = dataset_name
                    current_train_config.dataset_config = dataset_config

                    # Determine text field
                    sample_item = full_dataset[0]

                    # First check if 'text' field exists
                    if "text" in sample_item:
                        text_field = "text"
                    else:
                        # Find the first string field that looks like text (longer than 10 chars)
                        text_fields = [k for k in sample_item.keys()
                                     if isinstance(sample_item[k], str) and len(sample_item[k]) > 10]

                        if text_fields:
                            text_field = text_fields[0]
                            console.print(f"Dataset doesn't have 'text' field. Using '{text_field}' field instead.")
                        else:
                            # If no long text field, just use the first string field
                            text_fields = [k for k in sample_item.keys() if isinstance(sample_item[k], str)]
                            text_field = text_fields[0] if text_fields else None
                            console.print(f"Using '{text_field}' as the text field.")

                    current_train_config.dataset_text_field = text_field

                    console.print(f"[green]✓[/green] Dataset info stored in config: {dataset_name}, {dataset_config}, field: {text_field}")

                # Optimize configuration
                optimized_train_config, optimized_model_config = optimize_for_time_target(
                    current_train_config,
                    current_model_config,
                    dataset_size,
                    target_hours,
                    hardware_manager
                )

                if get_user_confirmation("Apply these optimizations?"):
                    # Preserve dataset information when applying optimized configuration
                    dataset_name = current_train_config.dataset_name
                    dataset_config = current_train_config.dataset_config
                    dataset_text_field = current_train_config.dataset_text_field

                    # Apply optimized configuration
                    current_train_config = optimized_train_config
                    current_model_config = optimized_model_config

                    # Restore dataset information
                    current_train_config.dataset_name = dataset_name
                    current_train_config.dataset_config = dataset_config
                    current_train_config.dataset_text_field = dataset_text_field

                    console.print("[green]✓[/green] Configuration updated with preserved dataset info")

            except Exception as e:
                console.print(f"[red]Error optimizing configuration: {str(e)}")

        elif choice == '6':
            # Edit configuration table
            try:
                # Call the configuration editor
                updated_model_config, updated_train_config = edit_config_table(current_model_config, current_train_config)
                console.print("[green]✓[/green] Configuration updated successfully")
                return None, updated_model_config, updated_train_config, dataset_size
            except Exception as e:
                console.print(f"[red]Error editing configuration: {str(e)}[/red]")
                continue

        elif choice == '7':
            # Continue with current settings
            effective_size = dataset_size if current_train_config.use_subset else None
            return None, current_model_config, current_train_config, effective_size

        elif choice == '8':
            # Exit
            cleanup_and_exit()

def main():
    # Declare global variables
    global selected_dataset

    # Initialize environment
    hardware_manager, task_manager, training_progress = setup_training_environment()

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        console.print("[green]✓[/green] Tokenizer loaded successfully")

        # Initialize model configuration
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            num_dims=768,
            num_heads=16,
            num_kv_heads=4,
            num_layers=30,
            ffn_hidden_dims=1024,
            rmsnorm_eps=1e-6,
            rope_theta=1e5,
            context_len=2048,
            use_cache=False,
            use_flash=False,
            use_moe=False,
            moe_num_experts=4,
            moe_active_experts=4,
            moe_eps=1e-6,
            moe_aux_loss_coef=0.01,
            moe_shared_experts=1,
            use_lossfreebalance=False,
        )

        # Initialize training configuration with default mode
        train_config = TrainerConfig(
            vocab_size=tokenizer.vocab_size,
            num_epochs=1,  # Default for normal mode
            use_ddp=False,
            use_moe=False,
            use_lossfreebalance=False,
            clean_cuda_cache=True,
            use_compile=False,
            use_dtype="float16",
            use_flash=False,
            gradient_checkpointing=False,
            seed=1,
            max_seq_len=512,
            batch_size=4,
            accumulation_steps=4,
            weight_decay=0.1,
            warmup_ratio=0.01,
            learning_rate=1e-3,
            betas=(0.90, 0.95),
            update_rate=1e-5,
            val_ratio=0.005,
            steps_for_eval=20,
            eval_interval=100,
            log_interval=1,
            checkpoints_frequency=1,
            path_to_checkpoints="./model_testing",
            tokenized_dataset_path="fineweb-edu_tok-10BT",
            eval_log_file="log/eval.txt",
            use_lora=False,
            lora_rank=8,
            lora_alpha=32,
            lora_target_modules=["q_proj", "v_proj"],
            num_workers=4,
            queue_size=8,
            # Add memory management parameters
            gpu_memory_utilization=0.85,
            enable_memory_tracking=True,
            expandable_segments=True,
            empty_cache_freq=100,
            auto_adjust_batch=True,
            min_batch_size=1,
            # Configure subset mode
            use_subset=False,
            target_samples=None,
            # Configure checkpoint usage
            use_epoch_checkpoints=False
        )

        while True:
            console.print("\n[bold cyan]Training Setup[/bold cyan]")
            console.print("1. Start training")
            console.print("2. Manage dataset/configuration")
            console.print("3. Exit")

            choice = get_user_input("\nSelect option (1-3): ", ['1', '2', '3'])

            if choice == '1':
                # Initialize model
                model = Transformer(model_config)
                console.print("[green]✓[/green] Model initialized successfully")

                # Initialize data loader with proper subsetting
                try:
                    # Debug print statements
                    console.print(f"[yellow]Debug: train_config.use_subset = {train_config.use_subset if hasattr(train_config, 'use_subset') else 'Not defined'}")
                    console.print(f"[yellow]Debug: train_config.target_samples = {train_config.target_samples if hasattr(train_config, 'target_samples') else 'Not defined'}")

                    # Ensure use_subset and target_samples are properly set
                    if not hasattr(train_config, 'use_subset'):
                        train_config.use_subset = False
                    if not hasattr(train_config, 'target_samples'):
                        train_config.target_samples = None

                    # Determine subset size based on configuration
                    subset_size = train_config.target_samples if train_config.use_subset else None

                    if subset_size:
                        console.print(f"[yellow]Using subset of {subset_size} samples")
                    else:
                        console.print(f"[yellow]Using full dataset (no subsetting)")

                    # Initialize dataset variable
                    dataset = None

                    # Check if we have a dataset from the optimization phase
                    if selected_dataset is not None:
                        console.print(f"[green]✓[/green] Using stored dataset with {len(selected_dataset)} samples")
                        dataset = selected_dataset
                    else:
                        # Try to load dataset from training config
                        if hasattr(train_config, 'dataset_name') and train_config.dataset_name is not None:
                            try:
                                dataset_name = train_config.dataset_name
                                dataset_config = train_config.dataset_config if hasattr(train_config, 'dataset_config') else None
                                text_field = train_config.dataset_text_field if hasattr(train_config, 'dataset_text_field') else None

                                console.print(f"[yellow]Loading dataset from config: {dataset_name}")
                                if dataset_config:
                                    console.print(f"[yellow]Config: {dataset_config}")

                                # Load the dataset
                                dataset = load_dataset(
                                    dataset_name,
                                    dataset_config,
                                    split="train",
                                    cache_dir=train_config.tokenized_dataset_path
                                )

                                console.print(f"[green]✓[/green] Dataset loaded with {len(dataset)} samples")

                                # Check if we need to map a field to 'text'
                                if text_field is not None and text_field != "text":
                                    console.print(f"[yellow]Mapping '{text_field}' field to 'text'")
                                    dataset = dataset.map(lambda x: {"text": x[text_field]})
                            except Exception as e:
                                console.print(f"[red]Error loading dataset from config: {str(e)}")
                                console.print("[yellow]Falling back to default dataset")
                                dataset = load_dataset(
                                    "wikitext",
                                    "wikitext-2-raw-v1",
                                    split="train",
                                    cache_dir="data/dataset_cache"
                                )
                                console.print(f"[green]✓[/green] Loaded default dataset with {len(dataset)} samples")
                        else:
                            console.print("[yellow]No dataset info in config, loading default wikitext dataset")
                            dataset = load_dataset(
                                "wikitext",
                                "wikitext-2-raw-v1",
                                split="train",
                                cache_dir="data/dataset_cache"
                            )
                            console.print(f"[green]✓[/green] Default dataset loaded with {len(dataset)} samples")

                    # If we got here, we have a dataset
                    if dataset is not None:
                        console.print(f"[green]✓[/green] Using dataset with {len(dataset)} samples")

                    # Create data loader with subset size and dataset
                    data_loader = ThreadedDataLoader(train_config, subset_size=subset_size, dataset=dataset)
                    console.print(f"[green]✓[/green] Data loader initialized with {len(data_loader)} steps per epoch")
                except Exception as e:
                    console.print(f"[red]Error initializing data loader: {str(e)}")
                    raise e

                # Initialize trainer with dataset size awareness
                trainer = Trainer(train_config, model, tokenizer, console=console)

                # Update the UI with dataset and model information
                if training_progress:
                    # Get dataset information
                    dataset_name = getattr(train_config, 'dataset_name', 'Custom Dataset')
                    dataset_size = len(data_loader) * train_config.batch_size
                    subset_size = train_config.target_samples if hasattr(train_config, 'use_subset') and train_config.use_subset else dataset_size
                    use_subset = getattr(train_config, 'use_subset', False)

                    # Get model information
                    model_params = sum(p.numel() for p in model.parameters())
                    model_layers = model_config.num_layers
                    model_dims = model_config.num_dims

                    # Get training configuration
                    batch_size = train_config.batch_size
                    effective_batch_size = batch_size * train_config.accumulation_steps
                    learning_rate = train_config.learning_rate
                    current_device = str(trainer.device)

                    # Update the UI
                    training_progress.update_dataset_info(dataset_name, dataset_size, subset_size, use_subset)
                    training_progress.update_model_info(model_params, model_layers, model_dims)
                    training_progress.update_training_info(batch_size, effective_batch_size, learning_rate, current_device)
                    training_progress.update_message("Starting training process...")

                console.print("\n[green]Starting training process...")
                effective_size = train_config.target_samples if train_config.use_subset else len(data_loader)
                trainer.train(data_loader, training_progress)

                # Stop the progress display if it has a stop method
                if training_progress and hasattr(training_progress, 'stop'):
                    training_progress.stop()

                console.print(f"\n[green]Training completed successfully! Used {effective_size} samples")
                break

            elif choice == '2':
                # Manage dataset and configuration
                dataset, new_model_config, new_train_config, subset_size = manage_dataset_and_config(
                    model_config, train_config, tokenizer
                )

                # Store the dataset in our global variable and save dataset info in the config
                if dataset is not None:
                    selected_dataset = dataset
                    console.print(f"[green]✓[/green] Dataset stored for training: {len(selected_dataset)} samples")

                    # Store dataset info in the training config
                    try:
                        # Get dataset name and config from the dataset
                        dataset_name = getattr(dataset, "_info", {}).get("dataset_name", "custom_dataset")
                        dataset_config = getattr(dataset, "_info", {}).get("config_name", None)
                        text_field = "text" if "text" in dataset[0] else next((k for k in dataset[0].keys() if isinstance(dataset[0][k], str)), None)

                        # Update the training config with dataset information
                        train_config.dataset_name = dataset_name
                        train_config.dataset_config = dataset_config
                        train_config.dataset_text_field = text_field
                        console.print(f"[green]✓[/green] Dataset information stored in config: {dataset_name}")

                        # Update the training config with dataset info
                        if new_train_config is not None:
                            new_train_config.dataset_name = dataset_name
                            new_train_config.dataset_config = dataset_config
                            new_train_config.dataset_text_field = text_field
                            console.print(f"[green]✓[/green] Dataset info saved to config: {dataset_name} (config: {dataset_config})")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not save dataset info to config: {str(e)}")

                if new_model_config is not None:
                    model_config = new_model_config
                if new_train_config is not None:
                    train_config = new_train_config

            else:
                cleanup_and_exit()

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.")
        cleanup_and_exit(1)
    except Exception as e:
        console.print(f"\n[red]Error during training: {str(e)}")
        console.print_exception()
        cleanup_and_exit(1)

if __name__ == "__main__":
    main()
