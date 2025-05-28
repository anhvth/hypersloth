"""
Display functionality for HyperSloth logging.
"""

from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


class DisplayMixin:
    """Mixin class for display functionality."""

    # Optional attributes that may be provided by classes using this mixin
    gpu_id: Optional[str] = None
    formatter: Optional[Any] = None
    logger: Optional[Any] = None

    def __init__(self):
        self.console = Console()

    def log_config_table(
        self, config_dict: Dict[str, Any], title: str = "Configuration"
    ) -> None:
        """Log configuration in organized tables."""
        sections = self._get_config_sections()

        gpu_id = getattr(self, "gpu_id", None)
        if gpu_id == "0":
            self.console.print(f"\n[bold blue]{title}[/bold blue]")

            for section_name, keys in sections.items():
                self._display_config_section(section_name, keys, config_dict, sections)

            self.console.print()

    def _get_config_sections(self) -> Dict[str, list]:
        """Get configuration sections for organized display."""
        return {
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

    def _display_config_section(
        self,
        section_name: str,
        keys: list,
        config_dict: Dict[str, Any],
        sections: Dict[str, list],
    ) -> None:
        """Display a single config section."""
        table = Table(
            title=f"[bold cyan]{section_name}[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Parameter", style="cyan", width=30)
        table.add_column("Value", style="green", width=50)

        section_items = self._get_section_items(keys, config_dict)

        # Add remaining items to "Other" section if this is the last iteration
        if section_name == list(sections.keys())[-1]:
            other_items = self._get_other_items(config_dict, sections)
            if other_items:
                if not section_items:
                    table.title = "[bold cyan]Other Parameters[/bold cyan]"
                section_items.extend(other_items)

        if section_items:
            for param, value in section_items:
                table.add_row(param, value)
            self.console.print(table)

    def _get_section_items(self, keys: list, config_dict: Dict[str, Any]) -> list:
        """Get items for a configuration section."""
        section_items = []
        for key in keys:
            if key in config_dict:
                value = config_dict[key]
                if isinstance(value, (list, tuple)):
                    value_str = f'[{", ".join(map(str, value))}]'
                elif isinstance(value, str) and len(value) > 60:
                    value_str = f"{value[:60]}..."
                else:
                    value_str = str(value)
                section_items.append((key, value_str))
        return section_items

    def _get_other_items(
        self, config_dict: Dict[str, Any], sections: Dict[str, list]
    ) -> list:
        """Get items not covered by predefined sections."""
        all_keys = [item for sublist in sections.values() for item in sublist]
        return [(k, str(v)) for k, v in config_dict.items() if k not in all_keys]

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
        gpu_id = getattr(self, "gpu_id", None)
        if gpu_id == "0":
            panel_content = self._create_training_start_content(
                num_examples,
                num_epochs,
                batch_size,
                total_batch_size,
                gradient_accumulation_steps,
                max_steps,
                output_dir,
            )

            panel = Panel(
                panel_content,
                title="[bold blue]Training Information[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
            self.console.print(panel)

    def _create_training_start_content(
        self,
        num_examples: int,
        num_epochs: int,
        batch_size: int,
        total_batch_size: int,
        gradient_accumulation_steps: int,
        max_steps: int,
        output_dir: str,
    ) -> Text:
        """Create content for training start panel."""
        content = Text()
        content.append("🚀 Training Started", style="bold green")
        content.append("\n\n")

        content.append("📊 ", style="")
        content.append("Dataset Info:", style="bold cyan")
        content.append(f"\n   • Examples: {num_examples:,}")
        content.append(f"\n   • Epochs: {num_epochs}")
        content.append("\n\n")

        content.append("⚙️  ", style="")
        content.append("Batch Configuration:", style="bold cyan")
        content.append(f"\n   • Per Device Batch Size: {batch_size}")
        content.append(f"\n   • Total Batch Size: {total_batch_size:,}")
        content.append(f"\n   • Gradient Accumulation: {gradient_accumulation_steps}")
        content.append("\n\n")

        content.append("🎯 ", style="")
        content.append("Training Steps:", style="bold cyan")
        content.append(f"\n   • Total Steps: {max_steps:,}")
        content.append("\n\n")

        content.append("💾 ", style="")
        content.append("Output:", style="bold cyan")
        content.append(f"\n   • Directory: {output_dir}")

        return content

    def log_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Log performance metrics in a formatted table."""
        gpu_id = getattr(self, "gpu_id", None)
        if gpu_id == "0":
            table = Table(
                title="[bold green]🏁 Training Complete - Performance Metrics[/bold green]",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Metric", style="cyan", width=25)
            table.add_column("Value", style="green", width=20)

            for key, value in metrics.items():
                formatted_value = self._format_metric_value(key, value)
                table.add_row(key.replace("_", " ").title(), formatted_value)

            self.console.print(table)

    def _format_metric_value(self, key: str, value: float) -> str:
        """Format metric value based on key type."""
        if "loss" in key.lower():
            return f"{value:.4f}"
        elif "time" in key.lower() or "second" in key.lower():
            return f"{value:.2f}s"
        elif "token" in key.lower() and isinstance(value, (int, float)):
            if value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:,.0f}"
        else:
            return f"{value}"

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
        formatter = getattr(self, "formatter", None)
        logger = getattr(self, "logger", None)

        if formatter is not None and logger is not None:
            progress_msg = formatter.format_progress_step(
                step, loss, lr, grad_norm, epoch, tokens_per_sec
            )
            logger.info(progress_msg)

    def log_model_info(self, model_name: str, num_params: Optional[int] = None) -> None:
        """Log model information."""
        formatter = getattr(self, "formatter", None)
        logger = getattr(self, "logger", None)

        if formatter is not None and logger is not None:
            model_info = formatter.format_model_info(model_name, num_params)
            logger.info(model_info)

    def log_dataset_info(
        self,
        train_size: int,
        eval_size: Optional[int] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        """Log dataset information."""
        formatter = getattr(self, "formatter", None)
        logger = getattr(self, "logger", None)

        if formatter is not None and logger is not None:
            dataset_info = formatter.format_dataset_info(
                train_size, eval_size, cache_path
            )
            logger.info(dataset_info)

    def log_gpu_info(self, gpu: int, world_size: int, model_name: str = "") -> None:
        """Log GPU-specific information."""
        gpu_id = getattr(self, "gpu_id", None)
        logger = getattr(self, "logger", None)

        rank_info = f"GPU {gpu} (Rank {gpu_id}/{world_size-1})"

        if model_name:
            rank_info += f" | Model: {model_name}"

        if logger is not None:
            logger.info(f"🔧 {rank_info}")
