"""
Formatting utilities for HyperSloth logging.
"""

from typing import Optional


class LogFormatter:
    """Handle log formatting for different contexts."""

    def __init__(self, gpu_id: Optional[str] = None):
        self.gpu_id = gpu_id

    def get_log_format(self) -> str:
        """Get the main log format with GPU info."""
        return (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>GPU{extra[gpu_id]}</cyan> | "
            "<cyan>{file}:{line}</cyan> <cyan>({function})</cyan> - "
            "<level>{message}</level>"
        )

    def get_simple_format(self) -> str:
        """Get simple format without GPU info for global usage."""
        return (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{file}:{line}</cyan> <cyan>({function})</cyan> - "
            "<level>{message}</level>"
        )

    def format_progress_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        epoch: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
    ) -> str:
        """Format training step progress message."""
        progress_parts = [f"Step {step:>6}"]

        if epoch is not None:
            progress_parts.append(f"Epoch {epoch:.2f}")

        progress_parts.append(f"Loss {loss:.4f}")
        progress_parts.append(f"LR {lr:.2e}")

        if grad_norm is not None:
            progress_parts.append(f"GradNorm {grad_norm:.3f}")

        if tokens_per_sec is not None:
            progress_parts.append(f"{tokens_per_sec:.0f} tok/s")

        return "📈 " + " | ".join(progress_parts)

    def format_model_info(
        self, model_name: str, num_params: Optional[int] = None
    ) -> str:
        """Format model information message."""
        model_info = f"🤖 Model: [bold cyan]{model_name}[/bold cyan]"

        if num_params is not None:
            if num_params >= 1_000_000_000:
                param_str = f"{num_params / 1_000_000_000:.1f}B"
            elif num_params >= 1_000_000:
                param_str = f"{num_params / 1_000_000:.1f}M"
            else:
                param_str = f"{num_params:,}"
            model_info += f" | Parameters: [green]{param_str}[/green]"

        return model_info

    def format_dataset_info(
        self,
        train_size: int,
        eval_size: Optional[int] = None,
        cache_path: Optional[str] = None,
    ) -> str:
        """Format dataset information message."""
        dataset_info = f"📚 Training samples: [green]{train_size:,}[/green]"

        if eval_size is not None:
            dataset_info += f" | Eval samples: [yellow]{eval_size:,}[/yellow]"

        if cache_path:
            dataset_info += f" | Cache: [cyan]{cache_path}[/cyan]"

        return dataset_info
