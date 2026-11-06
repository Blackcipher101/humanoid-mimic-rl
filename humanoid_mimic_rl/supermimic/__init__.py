"""SuperMimic-style model-based training (Liao et al., CoRL 2026, arXiv:2610.28480).

One differentiable world model of the robot dynamics is fit to logged
simulator transitions; the tracking policy is improved by backpropagating the
return of short imagined rollouts through the frozen world model, with a
critic bootstrapping the return beyond the imagination horizon.

``state`` is numpy/JAX agnostic and unit-tested; ``replay``, ``networks`` and
``trainer`` require JAX/Flax/Optax.
"""
