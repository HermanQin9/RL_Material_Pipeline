"""
PPO策略评估脚本入口
"""
from __future__ import annotations

import argparse
from typing import Sequence

from ppo.evaluation import compare_policies, evaluate_policy


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO policy")
    parser.add_argument("--policy-path", type=str, help="Path to trained policy")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Show detailed episode information")
    parser.add_argument("--compare", nargs="+", help="Compare multiple policies")
    args = parser.parse_args(argv)

    if args.compare:
        compare_policies(args.compare, args.episodes)
    elif args.policy_path:
        evaluate_policy(args.policy_path, args.episodes, args.render)
    else:
        parser.error("Either --policy-path or --compare must be provided")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
