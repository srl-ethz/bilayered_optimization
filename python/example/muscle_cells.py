# -----------------------------------------------------------------------------
# Just a muscle cell moving.
# -----------------------------------------------------------------------------

import numpy as np
import torch
import os
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from py_diff_pd.common.display import export_mp4

from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable
from py_diff_pd.common.hex_mesh import generate_hexmesh_from_voxel

from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.renderer import PbrtRenderer

from py_diff_pd.common.sim import Sim

from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.display import display_hex_mesh



class DiagMuscleCell:
    def __init__ (self, folder):
        """
        Multiple connected cells with 5x5 muscle voxels in diagonal actuation mode.
        """
        self.folder = folder
        create_folder(folder, exist_ok=True)
        create_folder(folder + '/frames', exist_ok=True)

        bin_file_name = str(folder + '/' + 'muscle.bin')

        # TODO: Add a bezel around the muscles, otherwise it stretches too far out of the structure.
        self.dx = 0.01
        structure_width = 5
        num_segments = 4    # How many squares with side structure_width we should create
        binary_mask = np.ones([structure_width * num_segments, structure_width, 1])
        generate_hexmesh_from_voxel(binary_mask, bin_file_name, dx=self.dx)
        mesh = HexMesh3d()
        mesh.Initialize(bin_file_name)
        display_hex_mesh(mesh, file_name=str(folder + '/' + 'musclemesh.png'), show=False)

        q0 = ndarray(mesh.py_vertices())

        ### SIMULATION PARAMETERS
        #opt = { 'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': 16, 'use_bfgs': 1, 'bfgs_history_size': 10 }
        self.opt = {'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-8, 'rel_tol': 1e-10, 'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10}

        ### FEM parameters
        density = 1e2
        youngs_modulus = 4e4
        poissons_ratio = 0.3
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))

        deformable = HexDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        ### Muscle Pattern
        forward_diag_idx = []
        backward_diag_idx = []
        forward_com = []
        backward_com = []

        muscle_pattern = "diagonal"
        if muscle_pattern == "diagonal":
            for i in range(mesh.NumOfElements()):
                if i % (2*structure_width**2) >= structure_width**2:
                    if (i%(structure_width**2))%(structure_width-1) == 0 and i % (2*structure_width**2) <= (2*structure_width**2 - 4) and (i%(structure_width**2)) != 0:
                        backward_diag_idx.append(i)
                        backward_com.append(np.mean([mesh.py_vertex(v_idx) for v_idx in mesh.py_element(i)], axis=0))
                elif (i%(structure_width**2))%(structure_width+1) == 0:
                    forward_diag_idx.append(i)
                    forward_com.append(np.mean([mesh.py_vertex(v_idx) for v_idx in mesh.py_element(i)], axis=0))

        elif muscle_pattern == "snake":
            for i in range(mesh.NumOfElements()):
                if (
                    (i % (2*structure_width**2) >= 0 and i % (2*structure_width**2) < structure_width) 
                    or 
                    (i % (2*structure_width**2) >= structure_width**2 and i % (2*structure_width**2) < structure_width**2 + structure_width)
                ):
                    continue
                elif np.sign((i % (2*structure_width**2))-structure_width**2) * ((i % structure_width) - structure_width//2) > 0:
                    backward_diag_idx.append(i)
                    backward_com.append(np.mean([mesh.py_vertex(v_idx) for v_idx in mesh.py_element(i)], axis=0))
                elif np.sign((i % (2*structure_width**2))-structure_width**2) * ((i % structure_width) - structure_width//2) < 0:
                    forward_diag_idx.append(i)
                    forward_com.append(np.mean([mesh.py_vertex(v_idx) for v_idx in mesh.py_element(i)], axis=0))

        self.forward_com = forward_com
        self.backward_com = backward_com
        self.forward_act_direction = np.array([1.0, 1.0, 0.0])
        self.backward_act_direction = np.array([-1.0, 1.0, 0.0])
        deformable.AddActuation(5e4, self.forward_act_direction, forward_diag_idx)
        deformable.AddActuation(5e4, self.backward_act_direction, backward_diag_idx)


        ### Initial conditions.
        dofs = deformable.dofs()
        self.q0 = torch.as_tensor(q0, dtype=torch.float64).requires_grad_(False)
        self.v0 = torch.zeros(dofs, dtype=torch.float64).requires_grad_(False)
        self.dofs = dofs
        self.act_dofs = deformable.act_dofs()

        self.deformable = deformable
        self.sim = Sim(deformable)

        print_info(f"Structure with {self.dofs//3} vertices")

        self.unique_str = f"{density}_{youngs_modulus}_{poissons_ratio}"

        self._spp = 4
        self._camera_pos = (0.5, -1, 1.25)
        self._camera_lookat = (0.5, 0, 0.15)
        self._color = '056137'
        self._resolution = (1600, 900)



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
            f_ext = torch.zeros(self.dofs, dtype=torch.float64)
        if act is None:
            act = torch.zeros(self.act_dofs, dtype=torch.float64)
        
        q, v = self.sim(self.dofs, self.act_dofs, 'pd_eigen', q, v, act, f_ext, dt, self.opt)

        return q, v


    def visualize (self, q, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self._spp,
            'max_depth': 2,
            'camera_pos': self._camera_pos,
            'camera_lookat': self._camera_lookat,
            'resolution': self._resolution
        }
        renderer = PbrtRenderer(options)
        transforms=[
            ('s', 5.), 
            ('t', [0., 0., 0.])
        ]

        # Create mesh from vertices
        tmp_bin_file_name = '.tmp.bin'
        self.deformable.PySaveToMeshFile(ndarray(q), tmp_bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(tmp_bin_file_name)
        os.remove(tmp_bin_file_name)

        renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=self._color, transforms=transforms)

        for com in self.forward_com:
            shifted_com = com + [0,0,0.5*self.dx]   # Have arrows appear on upper surface
            vi = shifted_com
            vj = shifted_com + 3e-1*self.dx*self.forward_act_direction
            renderer.add_shape_mesh(
                {'name': 'curve', 
                'point': ndarray([vi, (2 * vi + vj) / 3, (vi + 2 * vj) / 3, vj]),
                'width': 0.1*self.dx}, 
                transforms=transforms, 
                color='ff0000'
            )
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(vj),
                'radius': 0.1*self.dx
                },
                color='ff3025', #red
                transforms=transforms
            )

        for com in self.backward_com:
            shifted_com = com + [0,0,0.5*self.dx]   # Have arrows appear on upper surface
            vi = shifted_com
            vj = shifted_com + 3e-1*self.dx*self.backward_act_direction
            renderer.add_shape_mesh(
                {'name': 'curve', 
                'point': ndarray([vi, (2 * vi + vj) / 3, (vi + 2 * vj) / 3, vj]),
                'width': 0.1*self.dx}, 
                transforms=transforms, 
                color='ff0000'
            )
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(vj),
                'radius': 0.1*self.dx
                },
                color='ff3025', #red
                transforms=transforms
            )

        renderer.add_tri_mesh(str(root_path) + '/asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7', transforms=[('s', 3)])

        renderer.render()



def main (num_frames, dt, visualization):
    folder = 'diagonalmuscle'
    sim = DiagMuscleCell(folder)

    q, v = sim.q0, sim.v0

    com = []
    for i in range(num_frames):
        start_time = time.time()

        if visualization:
            sim.visualize(q, f"{folder}/frames/frame_{i:04d}.png")
        end_vis = time.time()

        frequency = 2
        amplitude = 0.75
        act = amplitude * torch.ones(sim.act_dofs, dtype=torch.float64) * np.sin(i*dt * 2*torch.pi * frequency)
        act[sim.act_dofs//2:] *= -1
        act += 1

        q, v = sim.forward(q, v, act, dt=dt)

        # Time including visualization
        print(f"Frame [{i+1}/{num_frames}]: {1000*(time.time()-end_vis):.2f}ms (+ {1000*(end_vis-start_time):.2f}ms for visualization)")

    if visualization:
        export_mp4(f"{folder}/frames", f"{folder}/cell_{dt}_{amplitude}_{frequency}__{sim.unique_str}.mp4", int(1/dt))



if __name__ == "__main__":
    num_frames = 100
    dt = 0.01

    # Visualization slows down simulation a lot, can disable it if you only want to plot movement of Center of Mass (COM).
    main(num_frames, dt, visualization=True)