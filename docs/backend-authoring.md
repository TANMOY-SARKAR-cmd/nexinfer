# Backend Authoring Guide

This guide shows how to add support for a new accelerator, driver, or
"special module" to NexusInfer without modifying the core.

## 1. The two extension points

1. **A Backend** (`nexinfer.backends.base.Backend`) — a runtime that can load
   weights, execute layers, and report devices. Choose this for a complete
   driver stack (e.g., `ort`, `ggml`, `cuda`).
2. **A ReferenceModule** (`nexinfer.backends.special_module.ReferenceModule`)
   — a lightweight plug that reports devices and hands pre-computed weights
   to the special-module backend. Choose this for a coprocessor that you
   drive from a vendor SDK and want to cooperate with the default compute
   path.

## 2. Writing a Backend

Subclass `Backend` and implement six members:

```python
from nexinfer.backends.base import Backend, DeviceInfo, ModelSpec
from nexinfer.backends.registry import register_backend

class MyAcceleratorBackend(Backend):
    name = "myaccel"

    def capabilities(self):
        return {"quantization": ["int4", "int8"], "dtype": ["fp16"]}

    def detect_devices(self) -> list[DeviceInfo]:
        # probe your driver (CLI tool, SDK call, sysfs…) and return what exists
        devices = []
        for i, dev in enumerate(my_sdk.list_devices()):
            devices.append(DeviceInfo(
                device_id=f"/myaccel:{i}", kind="accel",
                vendor="myvendor", name=dev.name,
                total_memory_bytes=dev.memory, compute_score=dev.flops / 1e12))
        return devices

    def load(self, model_path: str, spec: ModelSpec, device_ids: list[str]):
        # map weights onto device_ids; raise if any id is unavailable
        ...

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        # return logits, shape (1, vocab)
        ...

    def decode(self, req_ids: list[str], next_ids: np.ndarray) -> np.ndarray:
        # return logits, shape (len(req_ids), vocab)
        ...

    def offload_layers(self, layer_ranges):
        # called by the cluster worker to restrict this process to its layers
        ...

    def close(self):
        ...

register_backend(MyAcceleratorBackend)
```

### Device IDs

Follow the naming convention so the orchestrator's placement plan composes:
`/cpu:<i>`, `/gpu:nvidia:<i>`, `/gpu:amd:<i>`, `/gpu:intel:<i>`,
`/tpu:<i>`, `/npu:intel:<i>`, `/myaccel:<i>`. Unknown kinds are accepted —
the planner treats them as generic compute.

### Detection best practices

- Prefer the vendor's CLI tool or Python SDK first; fall back to OS APIs
  (sysfs on Linux, DXGI/NVAPI on Windows).
- Never raise from `detect_devices()` — return an empty list on failure so
  the registry simply skips this backend on machines without the hardware.
- Set a realistic `compute_score` (TFLOPS normalized); it drives the
  placement heuristics.

## 3. Writing a ReferenceModule (special module)

For a coprocessor that computes fixed weights or kernels the default path
consumes:

```python
from nexinfer.backends.special_module import ReferenceModule, register_module

class MyCoprocessor(ReferenceModule):
    module_name = "my_copro"

    def is_available(self) -> bool:
        return my_sdk.device_count() > 0

    def detect_devices(self):
        # same DeviceInfo contract as backends
        ...

    def load(self, model_path, spec, device_ids):
        # upload your weights/buffers to the device
        ...

    def transform_weights(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # optional: modify weights the default backend will use
        return weights

    def forward_hook(self, layer_name: str, x: np.ndarray) -> np.ndarray | None:
        # optional: intercept a layer's input; return None to pass through
        return None

register_module(MyCoprocessor)
```

The `special_module` backend picks up every registered module automatically;
running `nexinfer backends` shows detected module devices.

## 4. Packaging

Ship your backend as a normal pip package with an entry point:

```toml
# your_package/pyproject.toml
[project.entry-points."nexinfer.backends"]
myaccel = "your_package.backend:MyAcceleratorBackend"
```

```toml
# or for a module plug
[project.entry-points."nexinfer.modules"]
my_copro = "your_package.module:MyCoprocessor"
```

`nexinfer.backends.registry.available_backends()` discovers entry-point
backends in addition to the built-ins, so users just `pip install
your-package` and your accelerator appears in `nexinfer backends`.

## 5. Platform portability

- Detect hardware in `detect_devices()` only; never in module import time,
  so importing the package on the wrong OS stays cheap.
- Keep vendor SDK imports *inside* method bodies so the package imports
  cleanly without the SDK installed (the registry already tolerates missing
  dependencies; your entry point will be skipped gracefully).
- Windows: prefer vendor CLIs over POSIX paths; use `shutil.which()` and
  guard `subprocess` calls with timeouts.

## 6. Verifying your backend

```bash
nexinfer backends               # must list your backend + devices
nexinfer profile                # placement plan should include your devices
nexinfer run --model demo --prompt hi --backend myaccel
python -m pytest tests/         # the suite runs against the numpy backend
```

Add a small test that loads your backend with `SPEC` from `tests/test_engine.py`
and asserts `prefill()` returns logits of shape `(1, vocab_size)`.
