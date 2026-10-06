"""Hyperparameter optimisation (HPO) package.

The modules ``param_space``, ``abc_optimizer`` and ``pareto`` are pure numpy
code without Django or simulation dependencies. The evaluation of candidates
is injected into :class:`~abmem.services.hpo.abc_optimizer.ArtificialBeeColony`
as a callable, so the optimizer can be tested on synthetic functions.
"""
