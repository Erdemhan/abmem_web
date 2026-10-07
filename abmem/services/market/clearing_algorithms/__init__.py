"""Market clearing algorithms (PLAN 2.10).

Every concrete subclass of :class:`.base.BaseClearingAlgorithm` with a ``key``
in a module of this package is discovered by :mod:`.registry` and becomes
selectable per scenario (``Market.clearing_algorithm``). The package performs
no eager imports.
"""
