import dataclasses

import pytest

from humanoid_mimic_rl.supermimic import config


@pytest.mark.parametrize("make", [config.smoke_config, config.full_config])
def test_profiles_valid(make):
    make().validate()


def test_paper_defaults():
    c = config.full_config()
    assert (c.wm_window, c.horizon, c.discount, c.gae_lambda) == (8, 32, 0.99, 0.95)
    assert (c.action_noise_std, c.policy_grad_clip_elementwise, c.grad_clip_global_norm) == (0.1, 0.1, 1.0)


@pytest.mark.parametrize(
    "change, msg",
    [
        ({"buffer_steps_per_env": 8}, "buffer"),
        ({"warmup_iterations": 1}, "warm-up"),
        ({"critic_minibatches": 3, "policy_batch_size": 10, "horizon": 1}, "minibatches"),
        ({"eval_every": 0}, "eval_every"),
    ],
)
def test_invalid(change, msg):
    with pytest.raises(ValueError, match=msg):
        dataclasses.replace(config.full_config(), **change).validate()
