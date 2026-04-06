"""
Server package for Canary Release Manager.
"""

__all__ = ["CanaryEnvironment"]


def __getattr__(name: str):
    if name == "CanaryEnvironment":
        from .canary_environment import CanaryEnvironment

        return CanaryEnvironment
    raise AttributeError(name)
