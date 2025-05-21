import slangpy as spy
import numpy as np
import os
import torch

from dataloader.mesh_loader import load_mesh_from_obj
from constants import DATA_DIR


def load_meshes(module:spy.Module, device:spy.Device):
    mesh = load_mesh_from_obj(os.path.join(DATA_DIR,"cat_head.obj"))
    
    positions = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_positions",
        data=mesh.positions.astype(np.float32),
    )
    triangles = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_triangles",
        data=mesh.triangles.astype(np.int32),
    )
    adjacencies = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_adjacencies",
        data=mesh.adjacencies.astype(np.int32),
    )
    triangle_normals = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_triangle_normals",
        data=mesh.triangle_normals.astype(np.float32),
    )
    v2t = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="mesh_v2t",
        data=mesh.v2t.flatten().astype(np.int32),
    )

    # positions = [spy.float3(x) for x in mesh.positions]

    slang_meshes = module.Mesh(
        positions = torch.tensor(mesh.positions.astype(np.float32)), 
    )

    return slang_meshes



def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # Create an SGL device with the local folder for slangpy includes
    device = spy.create_device(include_paths=[curr_dir], enable_cuda_interop=True)

    # Load the module
    module_path = os.path.join(curr_dir, "slang", "trace_geodesic.slang")
    module = spy.TorchModule.load_from_file(device, module_path)

    meshes = load_meshes(module, device)

    meshpoint = module.MeshPoint(
        face = 0,
        uv = spy.float2(0.3,0.3),
    )

    dirs = spy.float3(0,0,0)

    # result = module.straighest_geodesic(meshes, meshpoint, dirs)


if __name__ == "__main__":
    main()