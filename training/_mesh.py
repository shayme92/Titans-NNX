from __future__ import annotations

from jax.sharding import Mesh
from jax.experimental import mesh_utils
import jax


def create_mesh() -> Mesh | None:
    n_devices = len(jax.devices())
    if n_devices > 1:
        return Mesh(mesh_utils.create_device_mesh((n_devices, 1)), ("batch", "model"))

    return None
