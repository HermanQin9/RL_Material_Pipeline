import pytest
import numpy as np

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


def test_env_observation_has_method_count():
    env = PipelineEnv()
    obs = env.reset()
    assert 'method_count' in obs
    assert len(obs['method_count']) == len(env.pipeline_nodes)


def test_trainer_one_episode_runs():
    env = PipelineEnv()
    trainer = PPOTrainer(env, max_steps_per_episode=5)
    reward, length = trainer.train_episode()
    assert isinstance(reward, float)
    assert length == 5
