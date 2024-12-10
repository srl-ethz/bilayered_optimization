# -----------------------------------------------------------------------------
# Simulation environment defining a pneumatically actuated soft arm, loaded 
# from a VTK file.
# -----------------------------------------------------------------------------

import numpy as np
import torch
import os
import meshio
import trimesh

from collections import OrderedDict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.tet_mesh import generate_tet_mesh

from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.renderer import PbrtRenderer

from py_diff_pd.common.sim import Sim


class SoftArmSimulation:
    def __init__ (self, folder, gravity=False):
        self.folder = folder
        create_folder(folder, exist_ok=True)
        bin_file_name = f"{folder}/init_mesh.bin"

        ### Load Mesh
        vtkFile = f"_meshes/softarm.vtk"
        mesh = meshio.read(vtkFile)
        generate_tet_mesh(mesh.points * 1e-3, mesh.cells[-1].data, bin_file_name)

        mesh = TetMesh3d()
        mesh.Initialize(bin_file_name)
        verts = ndarray(mesh.py_vertices()).reshape(-1, 3)
        elements = np.array([np.array(mesh.py_element(i), dtype=np.int64) for i in range(mesh.NumOfElements())], dtype=np.int64)


        ### SIMULATION PARAMETERS
        self.method = 'pd_eigen'
        self.opt = {'max_pd_iter': 10000, 'max_ls_iter': 10, 'abs_tol': 1e-7, 'rel_tol': 1e-8, 'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10}
        create_folder(f"{folder}/{self.method}", exist_ok=False)

        ### FEM parameters
        density = 1070
        youngs_modulus = 151685
        poissons_ratio = 0.49
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
        self._obj_center = verts.mean(axis=0)

        for i in range(verts.shape[0]):
            vx, vy, vz = verts[i]

            if abs(vz - max_corner[2]) < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)

        ### Define inner faces of the 6 chambers
        self._boundary = self._get_boundary_ordered(verts, elements)
        all_meshes = trimesh.Trimesh(verts, self._boundary, process=False)
        components = trimesh.graph.connected_component_labels(edges=all_meshes.face_adjacency, node_count=len(all_meshes.faces))
        components_indices = []
        # Split all boundary indices into their respective components
        for label in np.unique(components):
            components_indices.append(np.argwhere(components == label).ravel())
        # Remove the longest component, which is the outer surface
        longest_list = max(components_indices, key=len)
        components_indices.remove(longest_list)
        inner_faces = []
        for component in components_indices:
            inner_faces.append(self._boundary[component])
        self._inner_faces = inner_faces

        ### Initial conditions.
        self.dofs = deformable.dofs()
        self.q0 = torch.as_tensor(ndarray(mesh.py_vertices()), dtype=torch.float64).requires_grad_(False)
        self.v0 = torch.zeros(self.dofs, dtype=torch.float64).requires_grad_(False)
        self.f_ext = torch.zeros(self.dofs, dtype=torch.float64)
        self.act_dofs = deformable.act_dofs()
        self.deformable = deformable

        self.sim = Sim(deformable)

        print_info(f"Structure with {int(self.dofs/3)} vertices")


    def _get_boundary_ordered(self, vertices, elements):
        """
        The boundary mesh whose normal points outward will be returned in the order of right hand rule. This is necessary to apply pressure to chambers.
        """
        boundary = OrderedDict()

        def fix_tet_faces(verts):
            verts = ndarray(verts)
            v0, v1, v2, v3 = verts
            f = []
            if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
                f = [(0, 1, 2), (2, 1, 3), (1, 0, 3), (0, 2, 3)]
            else:
                f = [(1, 0, 2), (1, 2, 3), (0, 1, 3), (2, 0, 3)]

            return ndarray(f).astype(np.int)

        for e in elements:
            element_vert = []

            for vi in e:
                element_vert.append(vertices[vi])

            element_vert = ndarray(element_vert)
            face_indices = fix_tet_faces(element_vert)

            for indices in face_indices:
                face = e[indices]
                sorted_face = tuple(sorted(face))

                if sorted_face in boundary:
                    del boundary[sorted_face]
                else:
                    boundary[sorted_face] = face

        faces = np.vstack(boundary.values())
        return faces


    def _apply_inner_pressure (self, q, pressure):
        """
        Applies pressure to the 6 inner chambers of the soft arm.

        Arguments:
            pressure (list [6]): Pressures to apply defined for each chamber.

        Returns:
            f_pressure (ndarray [3*N]) : External forces on all nodes, but only the pressure forces inside the chambers are nonzero.
        """
        f_pressure = torch.zeros(self.dofs, dtype=torch.float64)
        f_count = np.zeros_like(f_pressure, dtype=int)  # Forces are averaged over all faces adjacent to a vertex

        ### Loop over all pressure chambers
        for chamber_idx, chamber_faces in enumerate(self._inner_faces):
            chamber_faces = self._inner_faces[chamber_idx]
            chamber_p = pressure[chamber_idx]

            for face in chamber_faces:
                # Find surface normal (same for tet and hex)
                v0, v1, v2 = q.reshape(-1, 3)[face[0]], q.reshape(-1, 3)[face[1]], q.reshape(-1, 3)[face[2]]
                cross_prod = np.cross((v1 - v0), (v2 - v0))

                # Triangle area
                area_factor = 0.5
                f_p = -chamber_p * area_factor * cross_prod

                for vertex_idx in face:
                    # Apply forces in x, y and z directions (3 dimensional)
                    for d in range(3):
                        # Increase occurence count of vertex
                        f_count[3 * vertex_idx + d] += 1
                        # Set pressure force, the computation refers to SOFA SurfacePressureConstraint.
                        f_pressure[3 * vertex_idx + d] += f_p[d] / 3

        return f_pressure


    def forward (self, q, v, act=None, f_ext=None, pressure=None, dt=0.01):
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

        if pressure is not None:
            f_ext += self._apply_inner_pressure(q, pressure)

        q, v = self.sim(self.dofs, self.act_dofs, self.method, q, v, act, f_ext, dt, self.opt)

        return q, v


    def visualize (self, q, file_name, spp=4, extra_points=None):
        """
        Visualize mesh of current frame.

        Arguments:
            spp (float): Optional. Sampling for image creation. Defines how much noise is in the image, consequently how fast the image renders. Higher is better quality
        """
        tmp_bin_file_name = '.tmp.bin'
        self.deformable.PySaveToMeshFile(ndarray(q), tmp_bin_file_name)

        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': spp,
            'max_depth': 2,
            'camera_pos': (0.5, -1., 0.5),  # Position of camera
            'camera_lookat': (0, 0, .2)     # Position that camera looks at
        }

        renderer = PbrtRenderer(options)
        transforms = [
            ('s', 1.5), 
            ('t', [0., 0, 0.4])
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
                    'radius': 0.25
                    },
                    color='ff3025', #red
                    transforms=transforms
                )

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render()

