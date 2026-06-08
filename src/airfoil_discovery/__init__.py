"""Airfoil discovery system."""

__all__ = ["AirfoilDiscoveryPipeline"]


def __getattr__(name: str):
    if name == "AirfoilDiscoveryPipeline":
        from .pipeline import AirfoilDiscoveryPipeline

        return AirfoilDiscoveryPipeline
    raise AttributeError(name)
