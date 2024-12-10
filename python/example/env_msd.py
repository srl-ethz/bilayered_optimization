# -----------------------------------------------------------------------------
# Simulation environment defining an approximate mass spring system. The mass 
# is a 1m x 1m x 1m cube with a density of 1kg/m^3. The spring is 1m long and 
# represented by a single hexahedral element with a cross section of 0.01m x 
# 0.01m with Young's modulus of 1e6 Pa and Poisson's ratio of 0.45, with zero
# density. The spring is attached to a ceiling on one end, and at the other to 
# the mass. Gravity is applied to the mass.
# -----------------------------------------------------------------------------

import numpy as np
import torch
import pygmsh
import os
import struct

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.tet_mesh import generate_tet_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.common.hex_mesh import generate_hex_mesh

from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.display import render_hex_mesh

from py_diff_pd.common.sim import Sim


class MSDSimulation:
    def __init__ (self, folder, gravity=False):
        self.folder = folder
        create_folder(folder, exist_ok=True)
        bin_file_name = f"{folder}/init_mesh.bin"

        thickness = 1/11
        origin = ndarray([0., 0., 0.])
        voxels = np.zeros([int(1./thickness), int(1./thickness), int(2./thickness)])
        print(f"Number of voxels on each axis: {int(1./thickness)}, ideally this should be odd for symmetry.")
        # Assign ones to the voxels that are occupied by the mass and spring
        voxels[:, :, :int(1./thickness)] = 1
        mass_only = voxels.copy()
        voxels[int(np.ceil((0.5-thickness)/thickness)):int(np.floor((0.5+thickness)/thickness)), int(np.ceil((0.5-thickness)/thickness)):int(np.floor((0.5+thickness)/thickness)), int(1./thickness):] = 1
        spring_only = voxels - mass_only
        generate_hex_mesh(voxels, thickness, origin, bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))

        verts = ndarray(mesh.py_vertices()).reshape(-1, 3)


        ### SIMULATION PARAMETERS
        self.method = 'pd_eigen'
        self.opt = {'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10}
        create_folder(f"{folder}/{self.method}", exist_ok=False)

        ### FEM parameters
        density = 1
        # youngs_modulus = 1e6
        # poissons_ratio = 0.45
        deformable = HexDeformable()
        deformable.Initialize(bin_file_name, density, 'none', 0.0, 0.0)
        os.remove(bin_file_name)

        
        ### Material parameters
        # la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        # mu = youngs_modulus / (2 * (1 + poissons_ratio))
        # deformable.AddPdEnergy('corotated', [2 * mu,], [])
        # deformable.AddPdEnergy('volume', [la,], [])

        # Elasticity defined per element
        youngs_modulus_spring = 1e5
        youngs_modulus_mass = 1e8
        #youngs_modulus = np.ones(mesh.NumOfElements()) * youngs_modulus_mass
        #youngs_modulus[-1] = youngs_modulus_spring
        youngs_modulus = np.ones(mesh.NumOfElements()) * youngs_modulus_mass
        tmp = voxels + spring_only
        youngs_modulus[(tmp[tmp>0] == 2)] = youngs_modulus_spring
        #youngs_modulus[-int(spring_only.sum()):] = youngs_modulus_spring

        poissons_ratio_spring = 0.0
        poissons_ratio_mass = 0.4
        #poissons_ratio = np.ones(mesh.NumOfElements()) * poissons_ratio_mass
        #poissons_ratio[-1] = poissons_ratio_spring
        poissons_ratio = np.ones(mesh.NumOfElements()) * poissons_ratio_mass
        poissons_ratio[(tmp[tmp>0] == 2)] = poissons_ratio_spring
        #poissons_ratio[-int(spring_only.sum()):] = poissons_ratio_spring
        deformable.PyAddPdEnergyHeterogeneous('corotated', youngs_modulus.tolist())
        deformable.PyAddPdEnergyHeterogeneous('volume', poissons_ratio.tolist())

        ### Gravity
        if gravity:
            deformable.AddStateForce('gravity', ndarray([0, 0, -9.81]))


        ### Boundary conditions: Glue vertices spatially
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        self._obj_center = (max_corner-min_corner)/2

        for i in range(verts.shape[0]):
            vx, vy, vz = verts[i]

            if abs(vz - max_corner[2]) < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)


        ### Initial conditions.
        self.dofs = deformable.dofs()
        self.q0 = torch.as_tensor(ndarray(mesh.py_vertices()), dtype=torch.float64).requires_grad_(False)
        self.v0 = torch.zeros(self.dofs, dtype=torch.float64).requires_grad_(False)
        self.f_ext = torch.zeros(self.dofs, dtype=torch.float64)
        self.act_dofs = deformable.act_dofs()
        self.mesh = mesh
        self.deformable = deformable

        self.sim = Sim(deformable)

        print_info(f"Structure with {int(self.dofs/3)} vertices")



    def forward (self, q, v, act=None, f_ext=None, dt=0.01, pressure=None):
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



    def visualize (self, q, file_name, spp=4, extra_points=None):
        """
        Visualize mesh of current frame.

        Arguments:
            spp (float): Optional. Sampling for image creation. Defines how much noise is in the image, consequently how fast the image renders. Higher is better quality
        """

        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': spp,
            'max_depth': 2,
            'camera_pos': (1, -6., 2),  # Position of camera
            'camera_lookat': (0, 0, 1.2)     # Position that camera looks at
        }

        renderer = PbrtRenderer(options)
        transforms = [
            ('s', 1), 
            ('t', [-self._obj_center[0], -self._obj_center[1], 0.5])
        ]

        tmp_bin_file_name = '.tmp.bin'
        self.deformable.PySaveToMeshFile(ndarray(q), tmp_bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(tmp_bin_file_name)
        os.remove(tmp_bin_file_name)

        renderer.add_hex_mesh(
            mesh, 
            transforms=transforms,
            render_voxel_edge=True,
            color='056137'
        )

        if extra_points is not None:
            for point in extra_points:
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': ndarray(point),
                    'radius': 0.0075
                    },
                    color='ff3025', #red
                    transforms=transforms
                )

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',texture_img='chkbd_24_0.7', transforms=[('s', 5)])

        renderer.render()

