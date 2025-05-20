import slangpy as spy
import numpy as np
import os

from dataloader.mesh_loader import load_mesh_from_obj
from constants import DATA_DIR


def load_mesh(module:spy.Module, device:spy.Device):
    mesh = load_mesh_from_obj(os.path.join(DATA_DIR,"cat_head.obj"))
    
    positions = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_positions",
        data=mesh.positions,
    )
    triangles = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_triangles",
        data=mesh.triangles,
    )
    adjacencies = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_adjacencies",
        data=mesh.adjacencies,
    )
    triangle_normals = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_triangle_normals",
        data=mesh.triangle_normals,
    )
    v2t = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_v2t",
        data=mesh.v2t,
    )

    # TODO: this doesn't work
    slang_mesh = module.Mesh(positions, triangles, adjacencies, triangle_normals, v2t)
    return slang_mesh



def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # Create an SGL device with the local folder for slangpy includes
    device = spy.create_device(include_paths=[curr_dir])

    # Load the module
    module = spy.Module.load_from_file(device, os.path.join(curr_dir, "slang", "trace_geodesic.slang"))

    mesh = load_mesh(module, device)


if __name__ == "__main__":
    main()