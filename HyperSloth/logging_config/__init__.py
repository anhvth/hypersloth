"""
Enhanced logging configuration for HyperSloth with improved formatting and organization.
"""

import os
import sys
from typing import Optional, Dict, Any
from loguru import logger

from .logging_timer import TimingMixin
from .logging_formatter import LogFormatter
from .logging_display import DisplayMixin


class HyperSlothLogger(TimingMixin, DisplayMixin):
    """Enhanced logger for HyperSloth with better formatting and GPU-aware logging."""

    def __init__(self, gpu_id: Optional[str] = None, log_level: str = None):
        self.gpu_id = gpu_id or os.environ.get("HYPERSLOTH_LOCAL_RANK", "0")
        self.log_level = (
            log_level or os.environ.get("HYPERSLOTH_LOG_LEVEL", "INFO")
        ).upper()

        # Initialize mixins
        TimingMixin.__init__(self)
        DisplayMixin.__init__(self)

        self.formatter = LogFormatter(self.gpu_id)
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup loguru logger with enhanced formatting."""
        from loguru import logger as base_logger

        self.logger = base_logger.bind(gpu_id=self.gpu_id)

        try:
            base_logger.remove()
        except ValueError:
            pass

        log_format = self.formatter.get_log_format()

        # Console handler for GPU 0 or single GPU mode
        if self._should_add_console_handler():
            base_logger.add(
                sys.stderr,
                format=log_format,
                level=self.log_level,
                colorize=True,
                enqueue=True,
                filter=lambda record: record["extra"].get("gpu_id") is not None,
            )

        # Individual GPU log files
        self._add_file_handlers(base_logger, log_format)

    def _should_add_console_handler(self) -> bool:
        """Check if console handler should be added."""
        return (
            self.gpu_id == "0"
            or len(os.environ.get("HYPERSLOTH_GPUS", "0").split(",")) == 1
        )

    def _add_file_handlers(self, base_logger, log_format: str) -> None:
        """Add file handlers for logging."""
        log_dir = ".log"
        os.makedirs(log_dir, exist_ok=True)

        # Individual GPU log
        log_file = f"{log_dir}/gpu_{self.gpu_id}.log"
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

        # Master log for GPU 0
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

    def log_error(self, error_msg: str, exc_info: bool = False) -> None:
        """Log error with enhanced formatting."""
        self.logger.error(f"❌ {error_msg}", exc_info=exc_info)

    def log_warning(self, warning_msg: str) -> None:
        """Log warning with enhanced formatting."""
        self.logger.warning(f"⚠️  {warning_msg}")

    def log_success(self, success_msg: str) -> None:
        """Log success message with enhanced formatting."""
        self.logger.success(f"✅ {success_msg}")


def setup_hypersloth_logger(
    gpu_id: Optional[str] = None, log_level: str = None
) -> HyperSlothLogger:
    """Setup and return enhanced logger instance."""
    if log_level is None:
        log_level = os.environ.get("HYPERSLOTH_LOG_LEVEL", "INFO")
    return HyperSlothLogger(gpu_id=gpu_id, log_level=log_level)


# def setup_global_safe_logger(gpu_id) -> Any:
#     """Get a logger instance that's safe to use with proper gpu_id binding."""

#     def _setup_global_safe_logger(gpu_id, log_level: str = None) -> None:
#         """Setup a global logger that's safe to use everywhere."""
#         if log_level is None:
#             log_level = os.environ.get("HYPERSLOTH_LOG_LEVEL", "INFO")
#         # remove
#         logger.remove()
#         formatter = LogFormatter(gpu_id)
#         simple_format = formatter.get_simple_format()

#         logger.add(
#             sys.stderr,
#             format=simple_format,
#             level=log_level.upper(),
#             colorize=True,
#             enqueue=True,
#         )

#     if gpu_id is None:
#         gpu_id = os.environ.get("HYPERSLOTH_LOCAL_RANK", "main")

#     return logger.bind(gpu_id=gpu_id)


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

    combined_config.update(training_dict)
    return combined_config
