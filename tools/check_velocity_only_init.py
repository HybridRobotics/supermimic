"""Run upstream initialization routing with stdlib dependency stubs.

No tensor computation is tested here. The runner's _init_policy and the
MultiTaskWorldModelEnv constructor are compiled unchanged from the checkout;
network builders, learners and the base environment are substituted. This
isolates task routing while the mjlab_research dependency is unavailable.
"""

import ast
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def method_source(path, name, class_name):
  source = path.read_text()
  class_start = source.index(f"class {class_name}(")
  start = source.index(f"  def {name}(", class_start)
  end = source.find("\n  def ", start + 1)
  return textwrap.dedent(source[start:] if end == -1 else source[start:end])


class StubBaseEnv:
  def __init__(self, **kwargs):
    self.base_arguments = kwargs


class StubActorCriticCfg:
  batch_size = 8


class StubBpttCfg:
  batch_size = 8


class StubLearner:
  def __init__(self, *args, **kwargs):
    self.arguments = args


def make_runner(slot_mix, policy_cfg):
  specs = [
    types.SimpleNamespace(name="motion", goal_dim=84),
    types.SimpleNamespace(name="velocity", goal_dim=3),
  ]
  actor = types.SimpleNamespace(task_specs=specs)
  motion_count = round(8 * slot_mix.get("motion", 0.0))
  actor.slot_ranges = lambda n: [(0, motion_count), (motion_count, n)]
  network = types.SimpleNamespace(to=lambda device: network)
  scope = dict(
    build_actor=lambda *args: actor,
    build_mlp=lambda *args, **kwargs: network,
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    ActorCriticPolicyLearnerCfg=StubActorCriticCfg,
    ActorCriticPolicyLearner=StubLearner,
    BpttPolicyLearner=StubLearner,
    StubBaseEnv=StubBaseEnv,
  )
  constructor = ast.parse(
    method_source(
      ROOT / "src/supermimic/envs/multi_task_world_model_env.py",
      "__init__",
      "MultiTaskWorldModelEnv",
    )
  ).body[0]
  cls = ast.ClassDef(
    name="MultiTaskWorldModelEnv",
    bases=[ast.Name(id="StubBaseEnv", ctx=ast.Load())],
    keywords=[],
    body=[constructor],
    decorator_list=[],
  )
  runner_init = ast.parse(
    method_source(
      ROOT / "src/supermimic/runners/supermimic_runner.py",
      "_init_policy",
      "SuperMimicRunner",
    )
  ).body[0]
  future = ast.ImportFrom(
    module="__future__", names=[ast.alias(name="annotations")], level=0
  )
  unit = ast.fix_missing_locations(
    ast.Module(body=[future, cls, runner_init], type_ignores=[])
  )
  exec(compile(unit, "<upstream initialization methods>", "exec"), scope)
  cfg = types.SimpleNamespace(
    slot_mix=slot_mix,
    policy=policy_cfg,
    critic_hidden_dims=(8,),
    history_length=1,
    termination_softness=0.05,
  )
  runner = types.SimpleNamespace(
    cfg=cfg,
    env=types.SimpleNamespace(num_envs=8),
    _wm=None,
    _dynamics=None,
    _representation=None,
    replay_buffer=None,
    device="cpu",
    _policy_input_dims=lambda: (68, 16),
    _configure_reset_sampling=lambda: None,
  )
  scope["_init_policy"](runner)
  return runner


class TestRunnerRouting(unittest.TestCase):
  def test_motion_only_initializes(self):
    for policy in (StubBpttCfg(), StubActorCriticCfg()):
      with self.subTest(policy=type(policy).__name__):
        make_runner({"motion": 1.0, "velocity": 0.0}, policy)

  def test_mixed_tasks_initialize(self):
    for policy in (StubBpttCfg(), StubActorCriticCfg()):
      with self.subTest(policy=type(policy).__name__):
        make_runner({"motion": 0.5, "velocity": 0.5}, policy)

  def test_velocity_only_initializes(self):
    # Desired behavior. This FAILS against the current upstream runner.
    for policy in (StubBpttCfg(), StubActorCriticCfg()):
      with self.subTest(policy=type(policy).__name__):
        make_runner({"motion": 0.0, "velocity": 1.0}, policy)


if __name__ == "__main__":
  print(f"Source checkout: {ROOT}", flush=True)
  print(
    "Source-method routing check with dependency stubs; not a simulator test.",
    flush=True,
  )
  unittest.main(verbosity=2)
