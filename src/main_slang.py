import slangpy as spy
import numpy as np
import os

from dataloader.mesh_loader import load_mesh_from_obj
from constants import DATA_DIR


def load_meshes(module:spy.Module, device:spy.Device):
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

    slang_meshes = spy.InstanceList(
        struct=module.Mesh.as_struct(),
        data={
            "positions":positions, 
            "triangles":triangles, 
            "adjacencies":adjacencies,
            "triangle_normals":triangle_normals,
            "v2t":v2t,
        }
    )

    return slang_meshes



def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # Create an SGL device with the local folder for slangpy includes
    device = spy.create_device(include_paths=[curr_dir])

    # Load the module
    module = spy.Module.load_from_file(device, os.path.join(curr_dir, "slang", "trace_geodesic.slang"))

    meshes = load_meshes(module, device)

    meshpoints = spy.InstanceList(
        struct=module.MeshPoint.as_struct(),
        data={
            "face":spy.int1(0), 
            "uv":spy.float2(0.3,0.3),
        }
    )

    dirs = spy.float3(1,1,1)

    # TODO: figure out how to call this
    result = module.straighest_geodesic(meshes, meshpoints, dirs)


if __name__ == "__main__":
    main()