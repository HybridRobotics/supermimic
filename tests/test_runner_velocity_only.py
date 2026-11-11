"""Runner initialization must support the same slot mixes as the actor."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from supermimic.models import MultiEncoderAE
from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg, BpttPolicyLearnerCfg
from supermimic.runners import supermimic_runner as module
from tests.conftest import (
  ACTION_DIM,
  OBS_DIM,
  make_actor_specs,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_wm_representation,
)


@pytest.mark.parametrize('slot_mix', [
  {'motion': 1., 'velocity': 0.},
  {'motion': .5, 'velocity': .5},
  {'motion': 0., 'velocity': 1.},
], ids=['motion-only', 'mixed', 'velocity-only'])
@pytest.mark.parametrize('policy_cls', [BpttPolicyLearnerCfg, ActorCriticPolicyLearnerCfg])
def test_runner_initialization_supports_actor_slot_mix(monkeypatch, slot_mix, policy_cls):
  specs = [replace(spec, extract_training_goal=lambda motion, times, aux, width=spec.goal_dim:
                   torch.zeros(len(times), width)) for spec in make_actor_specs()]
  actor = MultiEncoderAE(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM, task_specs=specs,
    slot_mix=slot_mix, latent_dim=8, hidden_dims=(8, 8),
    latent_bc_weight=1., history_length=1,
  )
  monkeypatch.setattr(module, 'build_actor', lambda *args: actor)
  monkeypatch.setattr(module, 'build_mlp', lambda *args, **kwargs: torch.nn.Linear(16, 1))
  monkeypatch.setattr(module, 'BpttPolicyLearner', MagicMock())
  monkeypatch.setattr(module, 'ActorCriticPolicyLearner', MagicMock())
  monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
  env = make_mock_command_env()
  env.num_envs = 8
  cfg = SimpleNamespace(slot_mix=slot_mix, critic_hidden_dims=(8,),
    policy=policy_cls(batch_size=8), history_length=1, actor_obs='proprio',
    termination_softness=.05)
  runner = SimpleNamespace(cfg=cfg, env=env, _wm=None, device='cpu',
    _dynamics=make_mock_dynamics(), _representation=make_mock_wm_representation(8),
    replay_buffer=make_mock_buffer(), _policy_input_dims=lambda: (OBS_DIM, 16),
    _configure_reset_sampling=lambda: None)
  module.SuperMimicRunner._init_policy(runner)
  assert runner.wm_env is not None
  assert runner.wm_env.task_specs[0].name == 'motion'
  runner.motion = runner.wm_env.motion
  runner._collection_history = lambda sensor: sensor.unsqueeze(1)
  runner._representation.compute_actor_sensor.side_effect = lambda *a, **kw: kw['sim_actor_obs']
  obs = module.SuperMimicRunner._build_sim_obs_td(
    runner, torch.zeros(8, 214), torch.zeros(8, 7),
    torch.zeros(8, OBS_DIM), torch.zeros(8, dtype=torch.long), torch.zeros(8, 1),
  )
  assert 'goal_motion' in obs.keys()
  actions = actor(obs)
  assert actions.shape == (8, ACTION_DIM)
  (actions.square().mean() + actor.auxiliary_loss().mean()).backward()
  if slot_mix['motion'] == 0.:
    assert actor.slot_ranges(8) == [(0, 0), (0, 8)]
    assert all(p.grad is None for p in actor.encoders['motion'].parameters())
    assert any(p.grad is not None for p in actor.encoders['velocity'].parameters())
