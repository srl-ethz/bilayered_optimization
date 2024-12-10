# -----------------------------------------------------------------------------
# Simulation environment defining a standard 10cm x 10cm x 3cm beam.
# -----------------------------------------------------------------------------

from cv2 import transform
import numpy as np
import torch
import pygmsh
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.tet_mesh import generate_tet_mesh

from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.renderer import PbrtRenderer

from py_diff_pd.common.sim import Sim


class BeamSimulation:
    def __init__ (self, folder, gravity=False, fixed=True):
        self.folder = folder
        create_folder(folder, exist_ok=True)
        bin_file_name = f"{folder}/init_mesh.bin"

        # Units in meters (m)
        with pygmsh.geo.Geometry() as geom:
            # First draw rectangle on xy-plane
            poly = geom.add_polygon(
                [
                    [0.00, 0.00],
                    [0.10, 0.00],
                    [0.10, 0.03],
                    [0.00, 0.03],
                ],
                mesh_size=0.01,
            )
            geom.extrude(poly, [0.0, 0.0, 0.03], num_layers=5)
            mesh = geom.generate_mesh()


            mesh_points = np.array(mesh.points)
            # [0] are lines, [1] are the triangles, [2] are tetrahedra, and [3] are vertices
            mesh_tets = np.array(mesh.cells[2].data)
            self.faces = mesh_tets

            generate_tet_mesh(mesh_points, mesh_tets, bin_file_name)

        mesh = TetMesh3d()
        mesh.Initialize(bin_file_name)

        verts = ndarray(mesh.py_vertices()).reshape(-1, 3)


        ### SIMULATION PARAMETERS
        self.method = 'pd_eigen'
        self.opt = {'max_pd_iter': 10000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10}
        create_folder(f"{folder}/{self.method}", exist_ok=False)

        ### FEM parameters
        density = 1070
        youngs_modulus = 263824
        poissons_ratio = 0.45
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        deformable = TetDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(bin_file_name)

        
        ### Material parameters
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        ### Gravity
        if gravity:
            deformable.AddStateForce('gravity', ndarray([0, 0, -9.81]))


        ### Boundary conditions: Glue vertices spatially
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        self._obj_center = (max_corner-min_corner)/2
        if fixed:
            for i in range(verts.shape[0]):
                vx, vy, vz = verts[i]

                if abs(vx - min_corner[0]) < 1e-3:
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
        tmp_bin_file_name = '.tmp.bin'
        generate_tet_mesh(q.view(-1, 3), self.faces, tmp_bin_file_name)

        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': spp,
            'max_depth': 2,
            'camera_pos': (0.25, -0.5, 0.3),  # Position of camera
            'camera_lookat': (0, 0, .2)     # Position that camera looks at
        }

        renderer = PbrtRenderer(options)
        transforms = [
            ('s', 1), 
            ('t', [-self._obj_center[0], -self._obj_center[1], 0.2])
        ]

        mesh = TetMesh3d()
        mesh.Initialize(tmp_bin_file_name)
        os.remove(tmp_bin_file_name)
        
        renderer.add_tri_mesh(
            mesh, 
            transforms=transforms,
            render_tet_edge=True,
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

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render()

