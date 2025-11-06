#!/usr/bin/env python3
"""Extended PPO validation helpers.

 PPO 

This module previously orchestrated long and resource-intensive PPO training
runs. Legacy merge artefacts made it impossible to import, breaking the global
"check all imports" workflow. The implementation below preserves the public
API (``run_extended_ppo_training``, ``create_comprehensive_analysis``,
``print_comprehensive_analysis``) while providing a lightweight simulation.
Users who need the original long-running behaviour can set
``RUN_FULL_PPO_VALIDATION=1`` to execute the historical training script via a
subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
 "ValidationConfig",
 "run_extended_ppo_training",
 "create_comprehensive_analysis",
 "print_comprehensive_analysis",
]

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
 sys.path.insert(0, str(PIPELINE_ROOT))

os.environ.setdefault("PIPELINE_TEST", "1")

matplotlib.rcParams.setdefault(
 "font.sans-serif", ["SimHei", "Microsoft YaHei", "Arial", "sans-serif"]
)
matplotlib.rcParams.setdefault("axes.unicode_minus", False)


@dataclass
class ValidationConfig:
 """Configuration for the extended validation workflow."""

 episodes: int = 60
 seed: int = 2024
 run_full: bool = False
 script_path: Path = PIPELINE_ROOT / "scripts" / "train_ppo_safe.py"
 extra_env: Optional[Dict[str, str]] = None
 timeout_seconds: int = 0 # 0 = no timeout

 def resolved_script(self) -> Path:
 """Return an absolute path to the training script."""

 return Path(self.script_path).resolve()


def _default_config() -> ValidationConfig:
 run_full = os.getenv("RUN_FULL_PPO_VALIDATION", "0") == "1"

 for candidate in (
 PIPELINE_ROOT / "scripts" / "train_ppo_safe.py",
 PIPELINE_ROOT / "scripts" / "train_ppo.py",
 ):
 if candidate.exists():
 script_path = candidate
 break
 else:
 script_path = PIPELINE_ROOT / "scripts" / "train_ppo_safe.py"

 return ValidationConfig(run_full=run_full, script_path=script_path)


def _simulate_training(config: ValidationConfig) -> Dict[str, np.ndarray]:
 """Create deterministic pseudo-training traces."""

 rng = np.random.default_rng(config.seed)
 episodes = int(max(5, config.episodes))
 rewards = np.cumsum(rng.normal(loc=0.35, scale=0.08, size=episodes))
 rewards = rewards - rewards.min() + 0.5
 losses = np.clip(0.6 - np.log1p(np.arange(episodes)) * 0.1, 0.05, None)
 policy_entropy = np.clip(rng.normal(loc=0.8, scale=0.05, size=episodes), 0.5, 1.1)

 window = min(10, episodes)
 kernel = np.ones(window) / float(window)
 smoothed_rewards = np.convolve(rewards, kernel, mode="same")

 return {
 "episodes": np.arange(1, episodes + 1, dtype=int),
 "rewards": rewards,
 "losses": losses,
 "entropy": policy_entropy,
 "smoothed_rewards": smoothed_rewards,
 }


def _run_subprocess(config: ValidationConfig) -> Dict[str, object]:
 script = config.resolved_script()
 if not script.exists():
 return {
 "mode": "missing-script",
 "script": str(script),
 "timestamp": datetime.utcnow().isoformat(),
 "config": asdict(config),
 }

 env = os.environ.copy()
 if config.extra_env:
 env.update(config.extra_env)
 env.setdefault("PIPELINE_TEST", "0")

 cmd = [sys.executable, str(script), "--episodes", str(config.episodes)]

 try:
 completed = subprocess.run(
 cmd,
 cwd=str(PIPELINE_ROOT),
 env=env,
 timeout=config.timeout_seconds or None,
 capture_output=True,
 text=True,
 check=False,
 )
 except Exception as exc: # pragma: no cover - defensive
 return {
 "mode": "subprocess-error",
 "script": str(script),
 "error": repr(exc),
 "timestamp": datetime.utcnow().isoformat(),
 "config": asdict(config),
 }

 return {
 "mode": "subprocess",
 "script": str(script),
 "returncode": completed.returncode,
 "stdout": completed.stdout,
 "stderr": completed.stderr,
 "timestamp": datetime.utcnow().isoformat(),
 "config": asdict(config),
 }


def run_extended_ppo_training(config: ValidationConfig | None = None) -> Dict[str, object]:
 """Run the extended PPO validation flow or its simulation."""

 config = config or _default_config()
 if config.run_full:
 return _run_subprocess(config)

 traces = _simulate_training(config)
 return {
 "mode": "simulation",
 "timestamp": datetime.utcnow().isoformat(),
 "config": asdict(config),
 "traces": traces,
 "final_reward": float(traces["rewards"][-1]),
 "best_smoothed_reward": float(np.max(traces["smoothed_rewards"])),
 }


def create_comprehensive_analysis(results: Dict[str, object]) -> Dict[str, object]:
 """Generate a compact analysis bundle from training results."""

 analysis: Dict[str, object] = {
 "mode": results.get("mode", "unknown"),
 "timestamp": results.get("timestamp"),
 "summary": {},
 }

 if results.get("mode") == "simulation":
 traces_raw = results.get("traces", {})
 traces = traces_raw if isinstance(traces_raw, dict) else {}
 episodes = traces.get("episodes")
 rewards = traces.get("rewards")
 smoothed = traces.get("smoothed_rewards")
 losses = traces.get("losses")
 entropy = traces.get("entropy")

 fig, axes = plt.subplots(2, 2, figsize=(10, 6))
 fig.suptitle("Extended PPO Validation (Simulation)")

 axes[0, 0].plot(episodes, rewards, label="Reward")
 axes[0, 0].plot(episodes, smoothed, label="Smoothed", linestyle="--")
 axes[0, 0].set_xlabel("Episode")
 axes[0, 0].set_ylabel("Reward")
 axes[0, 0].legend()

 axes[0, 1].plot(episodes, losses, color="#d95f02")
 axes[0, 1].set_xlabel("Episode")
 axes[0, 1].set_ylabel("Loss")

 axes[1, 0].plot(episodes, entropy, color="#1b9e77")
 axes[1, 0].set_xlabel("Episode")
 axes[1, 0].set_ylabel("Policy Entropy")

 axes[1, 1].axis("off")
 axes[1, 1].text(
 0.02,
 0.95,
 " / Key Metrics",
 transform=axes[1, 1].transAxes,
 va="top",
 fontsize=11,
 )
 axes[1, 1].text(
 0.02,
 0.7,
 f" Final Reward: {results.get('final_reward'):.3f}",
 transform=axes[1, 1].transAxes,
 )
 axes[1, 1].text(
 0.02,
 0.55,
 f" Best Smoothed: {results.get('best_smoothed_reward'):.3f}",
 transform=axes[1, 1].transAxes,
 )

 fig.tight_layout()

 analysis.update(
 {
 "figure": fig,
 "summary": {
 "final_reward": results.get("final_reward"),
 "best_smoothed_reward": results.get("best_smoothed_reward"),
 "episodes": int(episodes[-1]) if episodes is not None else None,
 },
 }
 )

 else:
 analysis["summary"] = {
 "details": "Subprocess execution summary",
 "returncode": results.get("returncode"),
 "script": results.get("script"),
 }

 return analysis


def print_comprehensive_analysis(analysis: Dict[str, object]) -> None:
 """Pretty-print the analysis report."""

 mode = analysis.get("mode", "unknown")
 summary_raw = analysis.get("summary", {})
 summary = summary_raw if isinstance(summary_raw, dict) else {}

 print("====== Extended PPO Validation Report ======")
 print(f"Mode : {mode}")
 if analysis.get("timestamp"):
 print(f"Timestamp : {analysis['timestamp']}")

 if mode == "simulation":
 final_reward = summary.get('final_reward')
 best_smoothed = summary.get('best_smoothed_reward')
 episodes = summary.get('episodes')
 print(f"Final reward : {final_reward:.3f}" if final_reward else "Final reward: N/A")
 print(f"Best smoothed reward : {best_smoothed:.3f}" if best_smoothed else "Best smoothed: N/A")
 print(f"Total episodes : {episodes}" if episodes else "Total episodes: N/A")
 else:
 print("Subprocess details :")
 print(f" Script: {summary.get('script')}")
 print(f" Return code: {summary.get('returncode')}")

 print("===========================================")


if __name__ == "__main__": # pragma: no cover - manual smoke test
 RESULT = run_extended_ppo_training()
 ANALYSIS = create_comprehensive_analysis(RESULT)
 print_comprehensive_analysis(ANALYSIS)
