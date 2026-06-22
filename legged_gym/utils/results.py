import os
import sys
import json
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR


@dataclass
class ResultsPaths:
    root_dir: str
    terminal_log_path: str
    reward_plot_path: str


def _format_chinese_time(timestamp: datetime) -> str:
    return f"{timestamp.year}年{timestamp.month}月{timestamp.day}日{timestamp.hour}点{timestamp.minute}分"


def create_results_paths(task_name: str, algorithm_name: str, timestamp: Optional[datetime] = None) -> ResultsPaths:
    timestamp = timestamp or datetime.now()
    base_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "results",
        task_name,
        algorithm_name,
        _format_chinese_time(timestamp),
    )
    root_dir = base_dir
    suffix = 1
    while os.path.exists(root_dir):
        root_dir = f"{base_dir}_{suffix}"
        suffix += 1
    os.makedirs(root_dir, exist_ok=True)
    return ResultsPaths(
        root_dir=root_dir,
        terminal_log_path=os.path.join(root_dir, "terminal.log"),
        reward_plot_path=os.path.join(root_dir, "reward_curves.png"),
    )


def save_experiment_metadata(root_dir: str, metadata: Dict, log_dir: Optional[str] = None):
    targets = [root_dir]
    if log_dir is not None:
        targets.append(log_dir)
    for target_dir in targets:
        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, "config_snapshot.json")
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(_make_json_safe(metadata), output_file, indent=2, ensure_ascii=False)


def _make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()


@contextmanager
def mirror_terminal_output(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        stdout_tee = _TeeStream(sys.stdout, log_file)
        stderr_tee = _TeeStream(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            yield


class RewardCurveCollector:
    def __init__(self):
        self.series = defaultdict(list)

    def attach(self, runner):
        if hasattr(runner, "logger") and hasattr(runner.logger, "log"):
            self._attach_new_runner(runner.logger)
            return
        if hasattr(runner, "log"):
            self._attach_legacy_runner(runner)
            return
        raise AttributeError("Unsupported runner type: no compatible logging interface found.")

    def _attach_new_runner(self, logger):
        original_log = logger.log

        def wrapped_log(*args, **kwargs):
            iteration = kwargs.get("it")
            if iteration is None and args:
                iteration = args[0]
            self._record_from_new_logger(logger, iteration)
            return original_log(*args, **kwargs)

        logger.log = wrapped_log

    def _attach_legacy_runner(self, runner):
        original_log = runner.log

        def wrapped_log(*args, **kwargs):
            locs = kwargs.get("locs")
            if locs is None and args:
                locs = args[0]
            self._record_from_legacy_locs(locs)
            return original_log(*args, **kwargs)

        runner.log = wrapped_log

    def _record_from_new_logger(self, logger, iteration: Optional[int]):
        if iteration is None:
            return
        if len(logger.rewbuffer) > 0:
            self.series["Train/mean_reward"].append((iteration, mean(logger.rewbuffer)))

        reward_extras = self._extract_reward_extras(logger.ep_extras)
        for tag, value in reward_extras.items():
            self.series[tag].append((iteration, value))

    def _record_from_legacy_locs(self, locs):
        if not locs:
            return

        iteration = locs.get("it")
        if iteration is None:
            return

        rewbuffer = locs.get("rewbuffer")
        if rewbuffer:
            self.series["Train/mean_reward"].append((iteration, mean(rewbuffer)))

        reward_extras = self._extract_reward_extras(locs.get("ep_infos", []))
        for tag, value in reward_extras.items():
            self.series[tag].append((iteration, value))

    def _extract_reward_extras(self, ep_extras) -> Dict[str, float]:
        reward_extras = {}
        if not ep_extras:
            return reward_extras

        for key in ep_extras[0]:
            if not (key.startswith("rew_") or "reward" in key.lower()):
                continue

            values = []
            for ep_info in ep_extras:
                if key not in ep_info:
                    continue
                values.extend(self._tensor_to_list(ep_info[key]))

            if values:
                reward_extras[f"Episode/{key}"] = sum(values) / len(values)
        return reward_extras

    @staticmethod
    def _tensor_to_list(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [float(value)]

    def save_plot(self, output_path: str) -> bool:
        summary_tags = [tag for tag in self.series if tag == "Train/mean_reward" or tag.startswith("Rnd/")]
        component_tags = sorted(tag for tag in self.series if tag.startswith("Episode/rew_"))

        groups = []
        if summary_tags:
            groups.append(("Reward Summary", summary_tags))
        if component_tags:
            groups.append(("Reward Components", component_tags))
        if not groups:
            return False

        fig, axes = plt.subplots(len(groups), 1, figsize=(12, 4 * len(groups)), squeeze=False)
        for axis, (title, tags) in zip(axes[:, 0], groups):
            for tag in tags:
                points = self.series[tag]
                iterations = [point[0] for point in points]
                values = [point[1] for point in points]
                axis.plot(iterations, values, label=tag.replace("Episode/rew_", "rew_"))
            axis.set_title(title)
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Value")
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return True
