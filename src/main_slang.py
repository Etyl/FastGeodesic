import slangpy as spy
import numpy as np
import os

from dataloader.mesh_loader import load_mesh_from_obj
from constants import DATA_DIR
from geometry.mesh import Mesh

def load_mesh(module, device):
    mesh:Mesh = load_mesh_from_obj(os.path.join(DATA_DIR,"cat_head.obj"))
    positions = spy.NDBuffer(device, dtype=np.double, shape=mesh.positions.shape)
    triangles = spy.NDBuffer(device, dtype=np.int32, shape=mesh.triangles.shape)
    adjacencies = spy.NDBuffer(device, dtype=np.int32, shape=mesh.adjacencies.shape)
    triangle_normals = spy.NDBuffer(device, dtype=np.double, shape=mesh.triangle_normals.shape)
    v2t = spy.NDBuffer(device, dtype=np.double, shape=mesh.triangle_normals.shape)


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # Create an SGL device with the local folder for slangpy includes
    device = spy.create_device(include_paths=[curr_dir])

    # Load the module
    module = spy.Module.load_from_file(device, os.path.join(curr_dir, "slang", "trace_geodesic.slang"))

    # Create a buffer of particles (.as_struct ensures proper Python typing)
    particles = module.Mesh()