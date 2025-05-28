"""
Enhanced logging configuration for HyperSloth with improved formatting and organization.
"""

import os
import sys
import time
from typing import Optional, Dict, Any
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from rich.text import Text
import tabulate


class StepTimer:
    """Helper class to track timing for individual steps."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def finish(self) -> float:
        """Finish timing and return duration."""
        self.end_time = time.time()
        return self.duration

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class HyperSlothLogger:
    """Enhanced logger for HyperSloth with better formatting and GPU-aware logging."""

    def __init__(self, gpu_id: Optional[str] = None, log_level: str = "INFO"):
        self.gpu_id = gpu_id or os.environ.get("HYPERSLOTH_LOCAL_RANK", "0")
        self.log_level = log_level.upper()
        self.console = Console()

        # Timing tracking
        self.step_timers: Dict[str, StepTimer] = {}
        self.step_durations: Dict[str, list] = {}  # Store history of durations
        self.total_training_start: Optional[float] = None

        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup loguru logger with enhanced formatting."""
        # Create a separate logger instance to avoid conflicts
        from loguru import logger as base_logger

        # Create a new logger instance by copying the base logger
        self.logger = base_logger.bind(gpu_id=self.gpu_id)

        # Only remove handlers if this is the first GPU (to avoid conflicts)
        if self.gpu_id == "0":
            # Remove default handler only once
            try:
                base_logger.remove()
            except ValueError:
                # Handler already removed
                pass

        # Custom format with GPU info and better spacing
        log_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>GPU{extra[gpu_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # Only add handlers if this is the first setup or in single GPU mode
        if (
            self.gpu_id == "0"
            or len(os.environ.get("HYPERSLOTH_GPUS", "0").split(",")) == 1
        ):
            # Console handler with colors
            base_logger.add(
                sys.stderr,
                format=log_format,
                level=self.log_level,
                colorize=True,
                enqueue=True,
                filter=lambda record: record["extra"].get("gpu_id") is not None,
            )

        # File handler for individual GPU logs (always add these)
        log_dir = ".log"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/gpu_{self.gpu_id}.log"

        # Remove existing log file
        if os.path.exists(log_file):
            os.remove(log_file)

        base_logger.add(
            log_file,
            format=log_format,
            level="DEBUG",
            rotation="10 MB",
            retention="1 week",
            enqueue=True,
            filter=lambda record: record["extra"].get("gpu_id") == self.gpu_id,
        )

        # Master log file (all GPUs write here)
        if self.gpu_id == "0":
            master_log = f"{log_dir}/master.log"
            if os.path.exists(master_log):
                os.remove(master_log)

            base_logger.add(
                master_log,
                format=log_format,
                level="INFO",
                rotation="50 MB",
                retention="1 week",
                enqueue=True,
                filter=lambda record: record["extra"].get("gpu_id") is not None,
            )

    def log_config_table(
        self, config_dict: Dict[str, Any], title: str = "Configuration"
    ) -> None:

        # Organize config into sections
        sections = {
            "Model & Training": [
                "model_name",
                "max_seq_length",
                "load_in_4bit",
                "loss_type",
                "num_train_epochs",
            ],
            "Data": [
                "dataset_name_or_path",
                "num_samples",
                "test_ratio",
                "instruction_part",
                "response_part",
            ],
            "LoRA": ["r", "lora_alpha", "lora_dropout", "bias"],
            "Optimization": [
                "learning_rate",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "optim",
                "weight_decay",
            ],
            "Scheduling": [
                "lr_scheduler_type",
                "warmup_steps",
                "logging_steps",
                "eval_steps",
            ],
            "Hardware": ["gpus", "bf16", "fp16", "packing"],
            "Output": ["output_dir", "save_total_limit", "eval_strategy"],
        }

        if self.gpu_id == "0":  # Only master GPU logs config
            self.console.print(f"\n[bold blue]{title}[/bold blue]")

            for section_name, keys in sections.items():
                table = Table(
                    title=f"[bold cyan]{section_name}[/bold cyan]",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Parameter", style="cyan", width=30)
                table.add_column("Value", style="green", width=50)

                section_items = []
                for key in keys:
                    if key in config_dict:
                        value = config_dict[key]
                        # Format value based on type
                        if isinstance(value, (list, tuple)):
                            value_str = f"[{', '.join(map(str, value))}]"
                        elif isinstance(value, str) and len(value) > 60:
                            value_str = f"{value[:60]}..."
                        else:
                            value_str = str(value)
                        section_items.append((key, value_str))

                # Add remaining items to "Other" section if this is the last iteration
                if section_name == list(sections.keys())[-1]:
                    other_items = [
                        (k, str(v))
                        for k, v in config_dict.items()
                        if k
                        not in [
                            item for sublist in sections.values() for item in sublist
                        ]
                    ]
                    if other_items:
                        if not section_items:  # If current section is empty, rename it
                            table.title = "[bold cyan]Other Parameters[/bold cyan]"
                        section_items.extend(other_items)

                if section_items:
                    for param, value in section_items:
                        table.add_row(param, value)
                    self.console.print(table)

            self.console.print()  # Add spacing

    def log_training_start(
        self,
        num_examples: int,
        num_epochs: int,
        batch_size: int,
        total_batch_size: int,
        gradient_accumulation_steps: int,
        max_steps: int,
        output_dir: str,
    ) -> None:
        """Log training start information in a formatted way."""

        if self.gpu_id == "0":  # Only master GPU logs this
            panel_content = Text()
            panel_content.append("🚀 Training Started", style="bold green")
            panel_content.append("\n\n")

            panel_content.append("📊 ", style="")
            panel_content.append("Dataset Info:", style="bold cyan")
            panel_content.append(f"\n   • Examples: {num_examples:,}")
            panel_content.append(f"\n   • Epochs: {num_epochs}")
            panel_content.append("\n\n")

            panel_content.append("⚙️  ", style="")
            panel_content.append("Batch Configuration:", style="bold cyan")
            panel_content.append(f"\n   • Per Device Batch Size: {batch_size}")
            panel_content.append(f"\n   • Total Batch Size: {total_batch_size:,}")
            panel_content.append(
                f"\n   • Gradient Accumulation: {gradient_accumulation_steps}"
            )
            panel_content.append("\n\n")

            panel_content.append("🎯 ", style="")
            panel_content.append("Training Steps:", style="bold cyan")
            panel_content.append(f"\n   • Total Steps: {max_steps:,}")
            panel_content.append("\n\n")

            panel_content.append("💾 ", style="")
            panel_content.append("Output:", style="bold cyan")
            panel_content.append(f"\n   • Directory: {output_dir}")

            panel = Panel(
                panel_content,
                title="[bold blue]Training Information[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
            self.console.print(panel)

    def log_gpu_info(self, gpu: int, world_size: int, model_name: str = "") -> None:
        """Log GPU-specific information."""
        rank_info = f"GPU {gpu} (Rank {self.gpu_id}/{world_size-1})"

        if model_name:
            rank_info += f" | Model: {model_name}"

        self.logger.info(f"🔧 {rank_info}")

    def log_progress_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        epoch: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
    ) -> None:
        """Log training step with enhanced formatting."""

        # Create a formatted progress message
        progress_parts = [f"Step {step:>6}"]

        if epoch is not None:
            progress_parts.append(f"Epoch {epoch:.2f}")

        progress_parts.append(f"Loss {loss:.4f}")
        progress_parts.append(f"LR {lr:.2e}")

        if grad_norm is not None:
            progress_parts.append(f"GradNorm {grad_norm:.3f}")

        if tokens_per_sec is not None:
            progress_parts.append(f"{tokens_per_sec:.0f} tok/s")

        progress_msg = " | ".join(progress_parts)
        self.logger.info(f"📈 {progress_msg}")

    def log_model_info(self, model_name: str, num_params: Optional[int] = None) -> None:
        """Log model information."""
        model_info = f"🤖 Model: [bold cyan]{model_name}[/bold cyan]"

        if num_params is not None:
            if num_params >= 1_000_000_000:
                param_str = f"{num_params / 1_000_000_000:.1f}B"
            elif num_params >= 1_000_000:
                param_str = f"{num_params / 1_000_000:.1f}M"
            else:
                param_str = f"{num_params:,}"
            model_info += f" | Parameters: [green]{param_str}[/green]"

        self.logger.info(model_info)

    def log_dataset_info(
        self,
        train_size: int,
        eval_size: Optional[int] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        """Log dataset information."""
        dataset_info = f"📚 Training samples: [green]{train_size:,}[/green]"

        if eval_size is not None:
            dataset_info += f" | Eval samples: [yellow]{eval_size:,}[/yellow]"

        if cache_path:
            dataset_info += f" | Cache: [cyan]{cache_path}[/cyan]"

        self.logger.info(dataset_info)

    def log_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Log performance metrics in a formatted table."""
        if self.gpu_id == "0":  # Only master logs final metrics
            table = Table(
                title="[bold green]🏁 Training Complete - Performance Metrics[/bold green]",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Metric", style="cyan", width=25)
            table.add_column("Value", style="green", width=20)

            # Format metrics nicely
            for key, value in metrics.items():
                if "loss" in key.lower():
                    formatted_value = f"{value:.4f}"
                elif "time" in key.lower() or "second" in key.lower():
                    formatted_value = f"{value:.2f}s"
                elif "token" in key.lower() and isinstance(value, (int, float)):
                    if value >= 1_000_000:
                        formatted_value = f"{value/1_000_000:.1f}M"
                    elif value >= 1_000:
                        formatted_value = f"{value/1_000:.1f}K"
                    else:
                        formatted_value = f"{value:,.0f}"
                else:
                    formatted_value = f"{value}"

                table.add_row(key.replace("_", " ").title(), formatted_value)

            self.console.print(table)

    def log_error(self, error_msg: str, exc_info: bool = False) -> None:
        """Log error with enhanced formatting."""
        self.logger.error(f"❌ {error_msg}", exc_info=exc_info)

    def log_warning(self, warning_msg: str) -> None:
        """Log warning with enhanced formatting."""
        self.logger.warning(f"⚠️  {warning_msg}")

    def log_success(self, success_msg: str) -> None:
        """Log success message with enhanced formatting."""
        self.logger.success(f"✅ {success_msg}")

    # === TIMING METHODS ===
    def start_timing(self, step_name: str) -> None:
        """Start timing a major step."""
        self.step_timers[step_name] = StepTimer(step_name)
        if step_name not in self.step_durations:
            self.step_durations[step_name] = []

        self.logger.debug(f"⏱️  Started timing: {step_name}")

    def finish_timing(self, step_name: str, log_result: bool = True) -> float:
        """Finish timing a step and optionally log the result."""
        if step_name not in self.step_timers:
            self.logger.warning(f"⚠️  Timer '{step_name}' was not started")
            return 0.0

        timer = self.step_timers[step_name]
        duration = timer.finish()
        self.step_durations[step_name].append(duration)

        if log_result:
            self.log_step_duration(step_name, duration)

        # Clean up the timer
        del self.step_timers[step_name]
        return duration

    def log_step_duration(self, step_name: str, duration: float) -> None:
        """Log the duration of a completed step."""
        if duration < 0.1:
            duration_str = f"{duration*1000:.1f}ms"
        elif duration < 60:
            duration_str = f"{duration:.2f}s"
        elif duration < 3600:
            duration_str = f"{duration/60:.1f}m"
        else:
            duration_str = f"{duration/3600:.1f}h"

        self.logger.info(f"⏱️  {step_name}: {duration_str}")

    def start_total_training_timer(self) -> None:
        """Start the total training timer."""
        self.total_training_start = time.time()
        self.logger.info("🚀 Starting total training timer")

    def log_training_summary(self) -> None:
        """Log a summary of all timing information."""
        if not self.step_durations:
            return

        if self.gpu_id == "0":  # Only master GPU logs summary
            table = Table(
                title="[bold green]⏱️  Training Step Timing Summary[/bold green]",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Step", style="cyan", width=25)
            table.add_column("Count", style="yellow", width=8)
            table.add_column("Avg Duration", style="green", width=12)
            table.add_column("Total Duration", style="blue", width=12)
            table.add_column("Min/Max", style="magenta", width=15)

            total_time = 0.0
            for step_name, durations in self.step_durations.items():
                if not durations:
                    continue

                count = len(durations)
                avg_duration = sum(durations) / count
                total_duration = sum(durations)
                min_duration = min(durations)
                max_duration = max(durations)

                total_time += total_duration

                # Format durations
                def format_duration(dur: float) -> str:
                    if dur < 0.1:
                        return f"{dur*1000:.1f}ms"
                    elif dur < 60:
                        return f"{dur:.2f}s"
                    elif dur < 3600:
                        return f"{dur/60:.1f}m"
                    else:
                        return f"{dur/3600:.1f}h"

                table.add_row(
                    step_name,
                    str(count),
                    format_duration(avg_duration),
                    format_duration(total_duration),
                    f"{format_duration(min_duration)}/{format_duration(max_duration)}",
                )

            # Add total training time if available
            if self.total_training_start:
                total_training_time = time.time() - self.total_training_start
                table.add_row(
                    "[bold]TOTAL TRAINING[/bold]",
                    "-",
                    "-",
                    f"[bold]{self._format_duration(total_training_time)}[/bold]",
                    "-",
                )

            self.console.print(table)

    def log_step_timing_progress(
        self, step_name: str, current_step: int, total_steps: int
    ) -> None:
        """Log timing progress for steps showing average and estimated remaining time."""
        if step_name not in self.step_durations or not self.step_durations[step_name]:
            return

        durations = self.step_durations[step_name]
        avg_duration = sum(durations) / len(durations)
        remaining_steps = total_steps - current_step
        estimated_remaining = avg_duration * remaining_steps

        progress_msg = (
            f"📊 {step_name} Progress: {current_step}/{total_steps} "
            f"(Avg: {self._format_duration(avg_duration)}, "
            f"ETA: {self._format_duration(estimated_remaining)})"
        )

        if current_step % 10 == 0 or current_step == total_steps:  # Log every 10 steps
            self.logger.info(progress_msg)

    def _format_duration(self, duration: float) -> str:
        """Format duration consistently."""
        if duration < 0.1:
            return f"{duration*1000:.1f}ms"
        elif duration < 60:
            return f"{duration:.2f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"


def setup_enhanced_logger(
    gpu_id: Optional[str] = None, log_level: str = "DEBUG"
) -> HyperSlothLogger:
    """Setup and return enhanced logger instance."""
    return HyperSlothLogger(gpu_id=gpu_id, log_level=log_level)


def setup_global_safe_logger(gpu_id, log_level: str) -> None:
    """Setup a global logger that's safe to use everywhere with proper gpu_id binding."""
    # Only setup if no handlers exist or if explicitly requested
    if len(logger._core.handlers) == 0:
        # Simple format without gpu_id for global usage
        simple_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # Console handler
        logger.add(
            sys.stderr,
            format=simple_format,
            level=log_level.upper(),
            colorize=True,
            enqueue=True,
        )


def get_safe_logger(gpu_id: Optional[str] = None) -> Any:
    """Get a logger instance that's safe to use with proper gpu_id binding."""
    if gpu_id is None:
        gpu_id = os.environ.get("HYPERSLOTH_LOCAL_RANK", "main")

    return logger.bind(gpu_id=gpu_id)


def format_config_display(hyper_config: Any, training_config: Any) -> Dict[str, Any]:
    """Format config objects for better display."""
    combined_config = {}

    # Extract hyper_config fields
    if hasattr(hyper_config, "model_dump"):
        hyper_dict = hyper_config.model_dump()
    else:
        hyper_dict = hyper_config.__dict__ if hasattr(hyper_config, "__dict__") else {}

    # Extract training_config fields
    if hasattr(training_config, "model_dump"):
        training_dict = training_config.model_dump()
    else:
        training_dict = (
            training_config.__dict__ if hasattr(training_config, "__dict__") else {}
        )

    # Flatten nested configs
    for key, value in hyper_dict.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                combined_config[f"{key}_{nested_key}"] = nested_value
        else:
            combined_config[key] = value

    # Add training config
    combined_config.update(training_dict)

    return combined_config
