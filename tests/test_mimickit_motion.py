"""MimicKit motion loader: valid files load, files referencing other globals are refused (no code execution)."""

import os
import pickle

import numpy as np
import pytest

from humanoid_mimic_rl import mimickit_motion


def test_load_motion_reads_fps_and_frames(tmp_path):
    frames = np.random.default_rng(0).normal(size=(5, 34))
    path = tmp_path / "m.pkl"
    path.write_bytes(pickle.dumps({"loop_mode": 1, "fps": 16, "frames": [list(f) for f in frames]}))
    fps, out = mimickit_motion.load_motion(path)
    assert fps == 16.0
    np.testing.assert_allclose(out, frames)


def test_load_motion_accepts_numpy_frames(tmp_path):
    frames = np.zeros((3, 10))
    path = tmp_path / "m.pkl"
    path.write_bytes(pickle.dumps({"fps": 30, "frames": frames}))
    assert mimickit_motion.load_motion(path)[1].shape == (3, 10)


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_load_motion_refuses_disallowed_globals(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps({"fps": 16, "frames": _Exploit()}))
    with pytest.raises(pickle.UnpicklingError, match="disallowed global"):
        mimickit_motion.load_motion(path)


def test_load_motion_rejects_bad_shape(tmp_path):
    path = tmp_path / "m.pkl"
    path.write_bytes(pickle.dumps({"fps": 16, "frames": [[0.0, 1.0]]}))
    with pytest.raises(ValueError):
        mimickit_motion.load_motion(path)
