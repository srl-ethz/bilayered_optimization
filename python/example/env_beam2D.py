# -----------------------------------------------------------------------------
# Simulation environment defining a standard 10cm x 10cm x 30cm beam.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
from cv2 import transform
import numpy as np
import torch
import pygmsh
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from py_diff_pd.core.py_diff_pd_core import QuadMesh2d, QuadDeformable
from py_diff_pd.common.quad_mesh import generate_rectangle_mesh

from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import create_folder, ndarray, print_info

from py_diff_pd.common.sim import Sim


class BeamSimulation:
    def __init__ (self, folder, gravity=False, fixed=True):
        """
        
        Returns:
            Simulation function that can be used in forward and backward calls.
        """
        self.folder = folder
        create_folder(folder, exist_ok=True)
        bin_file_name = f"{folder}/init_mesh.bin"

        # Units in meters (m)
        width, height = 100, 50
        cell_nums = [width, height]
        dx = 0.01
        origin = np.zeros(2)
        generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
        
        mesh = QuadMesh2d()
        mesh.Initialize(bin_file_name)

        verts = ndarray(mesh.py_vertices()).reshape(-1, 2)


        ### SIMULATION PARAMETERS
        self.method = 'pd_eigen'
        self.opt = {'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10}
        create_folder(f"{folder}/{self.method}", exist_ok=False)

        ### FEM parameters
        density = 1e3
        youngs_modulus = 5e3
        poissons_ratio = 0.45
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        deformable = QuadDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)

        
        ### Material parameters
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        ### Gravity
        if gravity:
            deformable.AddStateForce('gravity', ndarray([0, -9.81]))


        ### Boundary conditions: Glue vertices spatially
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        self._obj_center = (max_corner-min_corner)/2
        if fixed:
            for i in range(verts.shape[0]):
                vx, vy = verts[i]

                if abs(vx - min_corner[0]) < 1e-3:
                    deformable.SetDirichletBoundaryCondition(2 * i, vx)
                    deformable.SetDirichletBoundaryCondition(2 * i + 1, vy)


        ### Initial conditions.
        self.dofs = deformable.dofs()
        self.q0 = torch.as_tensor(ndarray(mesh.py_vertices()), dtype=torch.float64).requires_grad_(False)
        self.v0 = torch.zeros(self.dofs, dtype=torch.float64).requires_grad_(False)
        self.f_ext = torch.zeros(self.dofs, dtype=torch.float64)
        self.act_dofs = deformable.act_dofs()
        self.mesh = mesh
        self.deformable = deformable

        self.sim = Sim(deformable)

        print_info(f"Structure with {int(self.dofs/2)} vertices")



    def forward (self, q, v, act=None, f_ext=None, dt=0.01):
        """
        Computes a forward pass through the simulation.

        Arguments:
            q (torch.Tensor [3*N]): Vertex positions.
            v (torch.Tensor [3*N]): Vertex velocities.

        Returns: 
            (q, v) of next timestep
        """
        if f_ext is None:
            f_ext = self.f_ext
        if act is None:
            act = torch.zeros(self.act_dofs)

        q, v = self.sim(self.dofs, self.act_dofs, self.method, q, v, act, f_ext, dt, self.opt)

        return q, v



    def visualize (self, q, file_name):
        """
        Visualize mesh of current frame.

        Arguments:
            
        """
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        q_plot = q.reshape(-1, 2).cpu().detach().numpy()
        ax.scatter(q_plot[:, 0], q_plot[:, 1], s=1)

        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([-1, 2])
        ax.grid()
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

