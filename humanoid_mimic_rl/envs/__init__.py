"""MJX environments for the Booster T1 that MuJoCo Playground does not ship.

* ``getup.T1Getup``: recover from a fall and stand (full-body collisions).
* ``backflip.T1Backflip``: standing backflip with a keyframe/phase reward.
* ``mimic.T1Mimic``: DeepMimic-style tracking of a reference clip.

Walking reuses playground's ``T1JoystickFlatTerrain`` unchanged.
"""
