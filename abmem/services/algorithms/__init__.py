"""Algorithm layer: policy interface, shared components and algorithm implementations.

This package deliberately performs no eager imports. Algorithm classes are
resolved lazily through :mod:`abmem.services.algorithms.registry`, so importing
the package never depends on the state of the individual algorithm modules.
"""
