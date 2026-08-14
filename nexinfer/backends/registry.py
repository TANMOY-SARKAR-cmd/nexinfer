"""Backend registry: discovers and loads pluggable inference backends.

Backends are Python classes registered under the ``nexinfer.backends``
entry-point group (declared in pyproject.toml) or loaded dynamically by
import path. The registry never needs changes to add a new driver -- the
author implements ``Backend`` and ships it as a package with the
entry-point, or the user points ``nexinfer config add-backend`` at a path.
"""

from __future__ import annotations

import importlib
import logging
import platform
import sys
from typing import Type

from nexinfer.backends.base import Backend

log = logging.getLogger("nexinfer.backends")

# Built-in backend entry points (matched with pyproject.toml and a manual
# fallback map so the registry works without an installed wheel too).
BUILTIN_BACKENDS: dict[str, str] = {
    "cpu_numpy": "nexinfer.backends.cpu_numpy:NumpyBackend",
    "ggml": "nexinfer.backends.ggml_backend:GGMLBackend",
    "ort": "nexinfer.backends.ort_backend:OrtBackend",
    "cuda": "nexinfer.backends.cuda_backend:CudaBackend",
    "rocm": "nexinfer.backends.rocm_backend:RocmBackend",
    "directml": "nexinfer.backends.directml_backend:DirectMLBackend",
    "tpu": "nexinfer.backends.tpu_backend:TpuBackend",
    "special_module": "nexinfer.backends.special_module:SpecialModuleBackend",
}

_registry: dict[str, Type[Backend] | None] = {}  # name -> class (None = load failed)


def _resolve_class(spec: str) -> Type[Backend]:
    """Import ``module.path:ClassName`` and return the class."""
    module_path, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"backend spec must be 'module.path:ClassName': {spec!r}")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, Backend)):
        raise TypeError(f"{spec} is not a Backend subclass")
    return cls


def register(name: str, spec_or_cls: str | Type[Backend]) -> None:
    """Manually register a backend by name."""
    if isinstance(spec_or_cls, str):
        _registry[name] = _resolve_class(spec_or_cls)
    else:
        _registry[name] = spec_or_cls
    log.info("registered backend %s -> %s", name, spec_or_cls)


def _discover_entrypoints() -> dict[str, str]:
    try:
        if sys.version_info >= (3, 12):
            from importlib.metadata import entry_points
            eps = entry_points(group="nexinfer.backends")
        else:
            from importlib.metadata import entry_points
            eps = entry_points().get("nexinfer.backends", [])  # type: ignore[union-attr]
        return {ep.name: f"{ep.module}:{ep.attr}" for ep in eps}
    except Exception as exc:  # pylint: disable=broad-except
        log.debug("entry-point discovery failed: %s", exc)
        return {}


def available_backends() -> dict[str, str]:
    specs = dict(BUILTIN_BACKENDS)
    specs.update(_discover_entrypoints())
    return specs


def load_backend(name: str, allow_missing: bool = True) -> Backend | None:
    """Load (and instantiate) a backend by registered name.

    Returns None (with a warning) when the backend's optional dependencies
    are not installed, unless ``allow_missing=False``.
    """
    specs = available_backends()
    spec = specs.get(name)
    if spec is None:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(specs)}")
    if name in _registry:
        cls = _registry[name]
    else:
        try:
            cls = _resolve_class(spec)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("could not load backend %s: %s", name, exc)
            _registry[name] = None
            if allow_missing:
                return None
            raise
        _registry[name] = cls

    if cls is None:
        return None

    instance = cls()
    if instance.platform != "any" and platform.system().lower() != instance.platform:
        log.debug("backend %s skipped (platform %s)", name, instance.platform)
        return None
    return instance


def detect_all_backends() -> list[Backend]:
    """Instantiate every loadable backend and collect its detected devices."""
    out: list[Backend] = []
    for name in available_backends():
        be = load_backend(name, allow_missing=True)
        if be is None:
            continue
        devices = be.detect_devices()
        if devices or be.name == "cpu_numpy":  # CPU always available
            out.append(be)
    return out
