"""
Timing functionality for HyperSloth logging.
"""

import time
from typing import Dict, Optional
from rich.table import Table
from rich.console import Console


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


class TimingMixin:
    """Mixin class for timing functionality."""

    def __init__(self):
        self.step_timers: Dict[str, StepTimer] = {}
        self.step_durations: Dict[str, list] = {}
        self.total_training_start: Optional[float] = None
        self.console = Console()

    def start_timing(self, step_name: str) -> None:
        """Start timing a major step."""
        self.step_timers[step_name] = StepTimer(step_name)
        if step_name not in self.step_durations:
            self.step_durations[step_name] = []

        if hasattr(self, "logger"):
            self.logger.debug(f"⏱️  Started timing: {step_name}")

    def finish_timing(self, step_name: str, log_result: bool = True) -> float:
        """Finish timing a step and optionally log the result."""
        if step_name not in self.step_timers:
            if hasattr(self, "logger"):
                self.logger.warning(f"⚠️  Timer '{step_name}' was not started")
            return 0.0

        timer = self.step_timers[step_name]
        duration = timer.finish()
        self.step_durations[step_name].append(duration)

        if log_result:
            self.log_step_duration(step_name, duration)

        del self.step_timers[step_name]
        return duration

    def log_step_duration(self, step_name: str, duration: float) -> None:
        """Log the duration of a completed step."""
        duration_str = self._format_duration(duration)
        if hasattr(self, "logger"):
            self.logger.info(f"⏱️  {step_name}: {duration_str}")

    def start_total_training_timer(self) -> None:
        """Start the total training timer."""
        self.total_training_start = time.time()
        if hasattr(self, "logger"):
            self.logger.info("🚀 Starting total training timer")

    def log_training_summary(self) -> None:
        """Log a summary of all timing information."""
        if not self.step_durations or not hasattr(self, "gpu_id"):
            return

        if self.gpu_id == "0":
            table = Table(
                title="[bold green]⏱️  Training Step Timing Summary[/bold green]",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Step", style="cyan", width=25)
            table.add_column("Count", style="yellow", width=8)
            table.add_column("Avg Duration", style="green", width=12)
            table.add_column("Total Duration", style="blue", width=15)
            table.add_column("Percentage", style="magenta", width=10)
            table.add_column("Min/Max", style="white", width=15)

            total_time = self._calculate_total_time()

            for step_name, durations in self.step_durations.items():
                if not durations:
                    continue

                self._add_timing_row(table, step_name, durations, total_time)

            if self.total_training_start:
                total_training_time = time.time() - self.total_training_start
                table.add_row(
                    "[bold]TOTAL TRAINING[/bold]",
                    "-",
                    "-",
                    f"[bold]{self._format_duration(total_training_time)}[/bold]",
                    "[bold]100.0%[/bold]",
                    "-",
                )

            self.console.print(table)

    def _calculate_total_time(self) -> float:
        """Calculate total time from all step durations."""
        total_time = 0.0
        for durations in self.step_durations.values():
            total_time += sum(durations)

        if self.total_training_start:
            total_training_time = time.time() - self.total_training_start
            total_time = max(total_time, total_training_time)

        return total_time

    def _add_timing_row(
        self, table: Table, step_name: str, durations: list, total_time: float
    ) -> None:
        """Add a timing row to the table."""
        count = len(durations)
        avg_duration = sum(durations) / count
        total_duration = sum(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        percentage = (total_duration / total_time * 100) if total_time > 0 else 0

        table.add_row(
            step_name,
            str(count),
            self._format_duration(avg_duration),
            self._format_duration(total_duration),
            f"{percentage:.1f}%",
            f"{self._format_duration(min_duration)}/{self._format_duration(max_duration)}",
        )

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
            if hasattr(self, "logger"):
                self.logger.info(progress_msg)
