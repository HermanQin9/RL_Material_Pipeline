"""
Run a single Option 2 sequence through PipelineEnv and print the result.

Sequence: N0 -> N2 -> N1(mean) -> N3(outlier) -> N6(variance) -> N7(std) -> N8(rf) -> N9(terminate)
"""
import os
import sys
import logging


def main() -> None:
    # Reduce logging noise for a clean output
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('pipeline').setLevel(logging.WARNING)
    logging.getLogger('methods.data_methods').setLevel(logging.WARNING)

    # Ensure project root on sys.path
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    if project_root not in sys.path:
        sys.path.append(project_root)

    # Import after path setup
    from env.pipeline_env import PipelineEnv

    env = PipelineEnv()
    # Optional: turn off env debug prints
    if hasattr(env, 'debug'):
        env.debug = False

    # Reset environment
    _ = env.reset()

    # Define sequence (node name, method index, param value)
    seq = [
        ('N0', 0, 0.0),
        ('N2', 0, 0.0),
        ('N1', 0, 0.3),   # impute mean
        ('N3', 0, 0.5),   # cleaning outlier (IQR)
        ('N6', 0, 0.01),  # feature selection variance
        ('N7', 0, 0.0),   # scaling std
        ('N8', 0, 0.0),   # train rf (smaller model via param mapping)
        ('N9', 0, 0.0),   # terminate and compute reward
    ]

    done = False
    reward = 0.0
    metrics = {}

    for nid, midx, p in seq:
        action = {
            'node': env.pipeline_nodes.index(nid),
            'method': midx,
            'params': [float(p)],
        }
        _, reward, done, _, metrics = env.step(action)
        if done:
            break

    print(f"Done: {done}")
    print(f"Reward: {reward}")
    print(f"Metrics keys: {list(metrics.keys())}")


if __name__ == '__main__':
    main()
