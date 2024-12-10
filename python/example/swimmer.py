# -----------------------------------------------------------------------------
# Just a swimmer moving.
# -----------------------------------------------------------------------------

import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from scipy.special import comb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from py_diff_pd.common.display import export_mp4

from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable
from py_diff_pd.common.hex_mesh import generate_hexmesh_from_voxel, filter_hex

from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.renderer import PbrtRenderer

from py_diff_pd.common.sim import Sim

from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.display import display_hex_mesh


# plt.rcParams.update({'font.size': 7})     # Font size should be max 7pt and min 5pt
# plt.rcParams.update({'pdf.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
# plt.rcParams.update({'ps.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})
mm = 1/25.4



SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# From https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python
def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def plot_kineticEnergy (kineticEnergies, num_frames, dt):
    fig, ax = plt.subplots(figsize=(88*mm, 60*mm))
    ax.plot(np.linspace(0, num_frames*dt, num_frames), kineticEnergies)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Kinetic Energy (J)")
    ax.set_xlim(0, num_frames*dt)
    ax.grid()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title(f"Integrated Kinetic Energy: {sum(kineticEnergies)*dt:.4e} J s")
    fig.savefig("kinetic_energy.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_totalDeformation (totalDeformations, num_frames, dt):
    fig, ax = plt.subplots(figsize=(88*mm, 60*mm))
    ax.plot(np.linspace(0, num_frames*dt, num_frames), totalDeformations)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Deformation (m)")
    ax.set_xlim(0, num_frames*dt)
    ax.grid()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title(f"Summed Total Deformation: {sum(totalDeformations):.4e} m")
    fig.savefig("total_deformation.png", dpi=300, bbox_inches='tight')
    plt.close()



class Swimmer:
    def __init__ (self, folder, shapeParams):
        """
        Voxels with muscle in the bottom middle
        """
        self.folder = folder
        create_folder(folder, exist_ok=True)
        create_folder(folder + '/frames', exist_ok=True)

        bin_file_name = str(folder + '/' + 'swimmer.bin')

        width, height, depth = 15e-3, 1e-3, 3.5e-3
        nx = 30 if 'nx' not in shapeParams else shapeParams['nx']
        ny = 10 if 'ny' not in shapeParams else shapeParams['ny']
        nz = 2 if 'nz' not in shapeParams else shapeParams['nz']
        dx, dy, dz = width/nx, height/ny, depth/nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        binary_mask = np.ones([nx, ny, nz])
        generate_hexmesh_from_voxel(binary_mask, bin_file_name, dx=self.dx, dy=self.dy, dz=self.dz)
        mesh = HexMesh3d()
        mesh.Initialize(bin_file_name)
        # display_hex_mesh(mesh, file_name=str(folder + '/' + 'swimmermesh.png'), show=False)

        q0 = ndarray(mesh.py_vertices())

        ### SIMULATION PARAMETERS
        #opt = { 'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': 16, 'use_bfgs': 1, 'bfgs_history_size': 10 }
        self.opt = {'max_pd_iter': 2000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10}

        ### FEM parameters
        density = 1.1e3
        scaffoldMaterial = {
            'youngsModulus': 25e3 if 'scaffoldYM' not in shapeParams else shapeParams['scaffoldYM'],
            'poissonsRatio': 0.4,
        }
        muscleMaterial = {
            'youngsModulus': 25e3,
            'poissonsRatio': 0.4,
        }
        for m in [scaffoldMaterial, muscleMaterial]:
            m['la'] = m['youngsModulus'] * m['poissonsRatio'] / ((1 + m['poissonsRatio']) * (1 - 2 * m['poissonsRatio']))
            m['mu'] = m['youngsModulus'] / (2 * (1 + m['poissonsRatio']))
        corotatedEnergyList = []
        volumeEnergyList = []

        ### Muscle Pattern
        if 'widthRatio' in shapeParams and 'heightRatio' in shapeParams:
            assert 0.0 <= shapeParams['widthRatio'] <= 1.0 and 0.0 <= shapeParams['heightRatio'] <= 1.0
            muscleIdx = []
            for i in range(mesh.NumOfElements()):
                eleCom = np.mean([mesh.py_vertex(v_idx) for v_idx in mesh.py_element(i)], axis=0)
                
                if ((eleCom[0] >= (0.5-shapeParams['widthRatio']/2)*width and eleCom[0] <= (0.5+shapeParams['widthRatio']/2)*width) 
                    and (eleCom[1] >= 0.0*height and eleCom[1] <= shapeParams['heightRatio']*height)):
                    muscleIdx.append(i)
                    corotatedEnergyList.append(2*muscleMaterial['mu'])
                    volumeEnergyList.append(muscleMaterial['la'])
                else:
                    corotatedEnergyList.append(2*scaffoldMaterial['mu'])
                    volumeEnergyList.append(scaffoldMaterial['la'])

        elif 'muscleIdx2d' in shapeParams:
            muscleIdx = np.array(shapeParams['muscleIdx2d']).flatten()*int(depth/dz)
            for i in range(1, int(depth/dz)):
                muscleIdx = np.concatenate((muscleIdx, np.array(shapeParams['muscleIdx2d']).flatten()*int(depth/dz) + i))
            muscleIdx = list(map(int, list(muscleIdx)))

            corotatedEnergyList = [2*muscleMaterial['mu'] if i in muscleIdx else 2*scaffoldMaterial['mu'] for i in range(mesh.NumOfElements())]
            volumeEnergyList = [muscleMaterial['la'] if i in muscleIdx else scaffoldMaterial['la'] for i in range(mesh.NumOfElements())]

        else:
            raise ValueError("No muscle pattern specified")

        ### Initialize deformable object
        deformable = HexDeformable()
        deformable.Initialize(bin_file_name, density, 'none', 0.0, 0.0)
        # Elasticity
        deformable.PyAddPdEnergyHeterogeneous('corotated', corotatedEnergyList)
        deformable.PyAddPdEnergyHeterogeneous('volume', volumeEnergyList)

        deformable.AddActuation(5e3, np.array([1.0, 0.0, 0.0]), muscleIdx)
        self.muscleIdx = muscleIdx


        ### Initial conditions.
        dofs = deformable.dofs()
        self.q0 = torch.as_tensor(q0, dtype=torch.float64).requires_grad_(False)
        self.v0 = torch.zeros(dofs, dtype=torch.float64).requires_grad_(False)
        self.dofs = dofs
        self.act_dofs = deformable.act_dofs()

        self.deformable = deformable
        self.sim = Sim(deformable)

        # print_info(f"Structure with {self.dofs//3} vertices")

        self._spp = 32
        self._camera_pos = (0.0, -1.0, 3.0)
        self._camera_lookat = (0.0, 0.05, 0.0)
        self._meshColor = '496d8a'
        self._muscleColor = '6e4039'
        # self._color = '056137'
        self._resolution = (800, 800)



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
            ('s', 100.), 
            ('t', [-0.75, 0., 0.])
        ]

        # Create mesh from vertices
        tmp_bin_file_name = '.tmp.bin'
        self.deformable.PySaveToMeshFile(ndarray(q), tmp_bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(tmp_bin_file_name)
        os.remove(tmp_bin_file_name)

        # Filter and visualize the mesh
        meshScaffold = filter_hex(mesh, [i for i in range(mesh.NumOfElements()) if i not in self.muscleIdx])
        renderer.add_hex_mesh(meshScaffold, render_voxel_edge=True, color=self._meshColor, transforms=transforms)
        meshMuscle = filter_hex(mesh, self.muscleIdx)
        renderer.add_hex_mesh(meshMuscle, render_voxel_edge=True, color=self._muscleColor, transforms=transforms)

        renderer.add_tri_mesh(str(root_path) + '/asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7', transforms=[('s', 3)])

        renderer.render()



def run_sim (num_frames, dt, visualization=False, shapeParams=None):
    folder = 'swimmer'
    sim = Swimmer(folder, shapeParams)

    q, v = sim.q0, sim.v0

    rampTime = 0.25
    maxVal = 1.0
    actuation = smoothstep(np.linspace(0, dt*num_frames, num_frames), x_min=0.0, x_max=rampTime, N=1)
    actuation *= maxVal
    actuation = torch.as_tensor(1-actuation, dtype=torch.float64)

    end_vis = time.time()
    velocities = []
    for frame in range(num_frames):
        start_time = time.time()
        if visualization and frame % 1 == 0:
            sim.visualize(q, f"{folder}/frames/frame_{frame:04d}.png")
            # Time including visualization
            print(f"Frame [{frame+1}/{num_frames}]: {1000*(time.time()-end_vis):.2f}ms (+ {1000*(end_vis-start_time):.2f}ms for visualization)")
        end_vis = time.time()

        # Forward pass
        act = actuation[frame] * torch.ones(sim.act_dofs, dtype=torch.float64)
        q, v = sim.forward(q, v, act, dt=dt)
        velocities.append(v)

    # Compute kinetic energy
    # kineticEnergies = np.zeros(len(velocities))
    # for i in range(sim.deformable.NumOfElements()):
    #     vertexIdx = np.array(sim.deformable.mesh().py_element(i))
    #     eleMass = sim.deformable.density() * sim.deformable.mesh().element_volume(i)
    #     # Mass doesn't change during simulation since volume does not change.
    #     eleVel = np.stack(velocities, axis=0).reshape(len(velocities),-1,3)[:,vertexIdx].mean(axis=1)
    #     kineticEnergies += 0.5 * eleMass * (eleVel**2).sum(axis=-1)

    # print(f"Integrated kinetic energy: {sum(kineticEnergies)*dt:.4e}")

    # Compute maximum deformation
    totalDeformation = max(np.linalg.norm((q - sim.q0).reshape(-1, 3), axis=-1))
    print(f"Max deformation: {1e3*totalDeformation:.4e}mm")

    sim.visualize(q, f"{folder}/frames/frame.png")
    # plot_kineticEnergy(kineticEnergies, num_frames, dt)
    # plot_totalDeformation(totalDeformation, num_frames, dt)

    if visualization:
        export_mp4(f"{folder}/frames", f"{folder}/swimmer.mp4", int(0.5/dt))

    # return sum(kineticEnergies) * dt
    return 1e3*totalDeformation


def main (num_frames, dt, visualization=False, sweep=False):
    shapeParams = {
        'widthRatio': 1.0,
        'heightRatio': 0.5
    }

    solution = np.loadtxt(f"swimmer/runs/swimmer_{6.1e-10:.1e}.txt")
    shapeParams = {
        'muscleIdx2d': np.where(solution)[0]
        # 'muscleIdx2d': np.random.choice(np.arange(30*10), 150, replace=False)
    }

    if sweep:
        M, N = 15, 10
        w, h = np.meshgrid(np.linspace(1/M, 1.0, M), np.linspace(1/N, 1.0, N))
        params = np.stack([h.flatten(), w.flatten()], axis=1)
        metrics = np.empty(M*N)
        metrics[:] = np.nan

        results = []
        for i, (hR, wR) in enumerate(params):
            shapeParams['widthRatio'] = wR
            shapeParams['heightRatio'] = hR

            print(f"Width Ratio: {wR:.2f} - Height Ratio: {hR:.2f}")
            try:
                m = run_sim(num_frames, dt, visualization, shapeParams)
            except:
                print(f"Simulation failed for width ratio {wR:.2f} and height ratio {hR:.2f}")
            metrics[i] = m
            
            # plot heatmap of sweep
            metric = np.array(metrics).reshape(N, M)
            fig, ax = plt.subplots(figsize=(88*mm, 70*mm))
            c = ax.imshow(metric, cmap='jet', origin='lower', aspect='auto')
            ax.set_xticks(np.arange(2, M, 3))
            ax.set_yticks(np.arange(0, N, 2))
            ax.set_xticklabels(map(FormatStrFormatter('%.1f').format_data, np.linspace(1/M, 1.0, M)[2::3]))
            ax.set_yticklabels(map(FormatStrFormatter('%.1f').format_data, np.linspace(1/N, 1.0, N)[::2]))
            ax.set_xlabel("Width Ratio (-)")
            ax.set_ylabel("Height Ratio (-)")
            # ax.set_title("Integrated Kinetic Energy (J s)")
            # ax.set_title("Max Deformation (mm)")
            fig.colorbar(c)
            fig.savefig("heatmap.png", dpi=300, bbox_inches='tight')
            fig.savefig("heatmap.pdf", bbox_inches='tight')
            plt.close()

            # Store results
            results.append([wR, hR, m])
            np.savetxt("sweep.txt", results)

    else:
        startTime = time.time()
        run_sim(num_frames, dt, visualization, shapeParams)
        print(f"Simulation took {time.time()-startTime:.2f}s")



if __name__ == "__main__":
    num_frames = 40
    dt = 0.01

    # Visualization slows down simulation a lot, can disable it if you only want to plot movement of Center of Mass (COM).
    main(num_frames, dt, visualization=True, sweep=False)

