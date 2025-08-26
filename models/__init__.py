"""
Minimal model registry for the reduced GeoNeXt repository.

This module exposes only the core GeoNeXt implementation and its
ConvNeXtV2 backbone.  All other experimental or auxiliary models have
been removed to keep the codebase lean and focused on the primary
GeoNeXt architecture.  The `geonext_tiny` function constructs a
GeoNeXt model with a ConvNeXtV2–Tiny encoder by configuring the
appropriate depth and dimension parameters on an argument object
before instantiating the underlying `GeoNeXt` class.

The ConvNeXtV2 implementation is also exposed so that external
libraries or custom scripts can load the backbone directly if
necessary.  Only the tiny variant of the backbone is provided; if
additional variants are required they can be added following the same
pattern.
"""

from .ConvNeXtV2 import ConvNeXtV2
from .GeoNeXt import GeoNeXt, geonext_tiny

__all__ = [
    "ConvNeXtV2",
    "GeoNeXt",
    "geonext_tiny",
]