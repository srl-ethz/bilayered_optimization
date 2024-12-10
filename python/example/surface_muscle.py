# -----------------------------------------------------------------------------
# A muscle model that only has muscles on the surface.
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


plt.rcParams.update({'font.size': 7})     # Font size should be max 7pt and min 5pt
plt.rcParams.update({'pdf.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({'ps.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})
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


def plot_kineticEnergy (kineticEnergies, num_frames, dt, fileName):
    fig, ax = plt.subplots(figsize=(88*mm, 60*mm))
    ax.plot(np.linspace(0, num_frames*dt, num_frames), kineticEnergies)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Kinetic Energy (J)")
    ax.set_xlim(0, num_frames*dt)
    ax.grid()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title(f"Integrated KE: {sum(kineticEnergies)*dt:.4e} J s")
    fig.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.close()


class SurfaceMuscle:
    def __init__ (self, folder, shapeParams):
        """
        Voxels with muscle on surface layer.
        """
        start_time = time.time()
        self.folder = folder
        create_folder(folder, exist_ok=True)
        create_folder(folder + '/frames', exist_ok=True)

        bin_file_name = str(folder + '/' + 'surfacemuscle.bin')

        ### Geometry
        width, height, depth = 15e-3, 6e-3, 3e-3
        nx, ny, nz = 15, 60, 30
        dx, dy, dz = width/nx, height/ny, depth/nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        muscleThickness = 0.1e-3
        # Anchor holes on both sides
        anchorHoleWidth = 2e-3 # in m
        anchorBoundary = 1e-3 # in m, distance from the boundary
        # Bridge holes, always symmetric, outer layers have to be bridges
        nBridges = shapeParams['nBridges'] if 'nBridges' in shapeParams else 1
        if 'bridgeRatio' in shapeParams:
            bridgeRatio = shapeParams['bridgeRatio']
            bridgeHeight = height*bridgeRatio/nBridges if 'bridgeRatio' in shapeParams else height
        else:
            bridgeRatio = 1.0
            bridgeHeight = shapeParams['bridgeHeight'] if 'bridgeHeight' in shapeParams else 1e-3
        bridgeHoleHeight = (height - nBridges*bridgeHeight)/(nBridges-1) if nBridges > 1 else 0

        # Create voxel grid
        binary_mask = np.ones([nx, ny, nz])
        binary_mask[round(anchorBoundary/dx):round((anchorBoundary+anchorHoleWidth)/dx), round(anchorBoundary/dy):-round(anchorBoundary/dy)] = 0
        binary_mask[-round((anchorBoundary+anchorHoleWidth)/dx):-round(anchorBoundary/dx), round(anchorBoundary/dy):-round(anchorBoundary/dy)] = 0
        for i in range(1, nBridges):
            binary_mask[round((2*anchorBoundary+anchorHoleWidth)/dx):-round((2*anchorBoundary+anchorHoleWidth)/dx), round((i*bridgeHeight+(i-1)*bridgeHoleHeight)/dy):round((i*bridgeHeight+i*bridgeHoleHeight)/dy)] = 0

        generate_hexmesh_from_voxel(binary_mask, bin_file_name, dx=self.dx, dy=self.dy, dz=self.dz)
        mesh = HexMesh3d()
        mesh.Initialize(bin_file_name)
        # display_hex_mesh(mesh, file_name=str(folder + '/' + 'mesh.png'), show=False)

        print(f"Mesh has {mesh.NumOfVertices()} vertices and {mesh.NumOfElements()} elements, finished in {time.time()-start_time:.2f}s")

        q0 = ndarray(mesh.py_vertices())

        ### SIMULATION PARAMETERS
        self.opt = {'max_pd_iter': 2000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10}

        ### FEM parameters
        density = 1.1e3
        youngs_modulus = 0.39e6
        poissons_ratio = 0.4
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        tendonYM = 0.39e6
        tendonPR = poissons_ratio

        deformable = HexDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        corotatedList = np.array([2*mu] * mesh.NumOfElements())
        volumeList = np.array([la] * mesh.NumOfElements())

        ### Muscle Pattern
        start_time = time.time()
        muscleIdx = []
        tendonIdx = []
        for i in range(mesh.NumOfElements()):
            eleCom = np.mean([mesh.py_vertex(v_idx) for v_idx in mesh.py_element(i)], axis=0)

            # Check if middle muscle part of the MTU
            if (eleCom[0] >= 2*anchorBoundary+anchorHoleWidth and eleCom[0] <= width-2*anchorBoundary-anchorHoleWidth):
                if ((eleCom[1] >= 0.0 and eleCom[1] <= muscleThickness) 
                    or (eleCom[1] >= height-muscleThickness and eleCom[1] <= height)
                    or (eleCom[2] >= 0.0 and eleCom[2] <= muscleThickness)
                    or (eleCom[2] >= depth-muscleThickness and eleCom[2] <= depth)
                ):
                    # Check if it's on the surface
                    muscleIdx.append(i)
                    pass
                else:
                    # Check if it's part of the bridge surface
                    for j in range(1, nBridges):
                        # Bridge hole height is not guaranteed a multiple of dx.
                        if (eleCom[1] >= round((j*bridgeHeight+(j-1)*bridgeHoleHeight)/dy)*dy-muscleThickness and eleCom[1] <= round((j*bridgeHeight+j*bridgeHoleHeight)/dy)*dy+muscleThickness):
                            muscleIdx.append(i)
                            break
            # Part of the tendons around anchors
            else:
                tendonIdx.append(i)

        deformable.AddActuation(0.3e6, np.array([1.0, 0.0, 0.0]), muscleIdx)
        self.muscleIdx = muscleIdx
        print(f"Muscle pattern with {len(muscleIdx)} elements created in {time.time()-start_time:.2f}s")

        ### Homogeneous elasticity
        # deformable.AddPdEnergy('corotated', [2 * mu,], [])
        # deformable.AddPdEnergy('volume', [la,], [])

        ### Heterogeneous elasticity
        tendonLa = tendonYM * tendonPR / ((1 + tendonPR) * (1 - 2 * tendonPR))
        tendonMu = tendonYM / (2 * (1 + tendonPR))
        corotatedList[tendonIdx] = 2*tendonMu
        volumeList[tendonIdx] = tendonLa
        deformable.PyAddPdEnergyHeterogeneous('corotated', corotatedList)
        deformable.PyAddPdEnergyHeterogeneous('volume', volumeList)



        ### Initial conditions.
        dofs = deformable.dofs()
        self.q0 = torch.as_tensor(q0, dtype=torch.float64).requires_grad_(False)
        self.v0 = torch.zeros(dofs, dtype=torch.float64).requires_grad_(False)
        self.dofs = dofs
        self.act_dofs = deformable.act_dofs()

        self.deformable = deformable
        self.sim = Sim(deformable)

        self._spp = 32
        self._camera_pos = (0.0, -0.65, 2.5)
        self._camera_lookat = (0.0, 0.3, 0.0)
        self._meshColor = '056137'
        self._muscleColor = 'ff3025'
        self._resolution = (1600, 900)

        self.visualize(q0, f"{folder}/frames/frame_{0:04d}.png")
        print_info(f"Structure with {self.dofs//3} vertices")


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
    folder = 'surfacemuscle'
    sim = SurfaceMuscle(folder, shapeParams)

    q, v = sim.q0, sim.v0

    actuation = torch.as_tensor(1 - 2*smoothstep(np.linspace(0, dt*num_frames, num_frames), x_min=0.0, x_max=0.25, N=1), dtype=torch.float64)

    end_vis = time.time()
    velocities = []
    for frame in range(num_frames):
        start_time = time.time()
        if visualization and frame % 1 == 0:
            sim.visualize(q, f"{folder}/frames/frame_{frame:04d}.png")
            # Time including visualization
            print(f"Frame [{frame+1}/{num_frames}]: {1000*(start_time-end_vis):.2f}ms (+ {1000*(time.time()-start_time):.2f}ms for visualization)")
        end_vis = time.time()

        # Forward pass
        act = actuation[frame] * torch.ones(sim.act_dofs, dtype=torch.float64)
        q, v = sim.forward(q, v, act, dt=dt)
        velocities.append(v)

    # Compute kinetic energy
    kineticEnergies = np.zeros(len(velocities))
    for i in range(sim.deformable.NumOfElements()):
        vertexIdx = np.array(sim.deformable.mesh().py_element(i))
        eleMass = sim.deformable.density() * sim.deformable.mesh().element_volume(i)
        # Mass doesn't change during simulation since volume does not change.
        eleVel = np.stack(velocities, axis=0).reshape(len(velocities),-1,3)[:,vertexIdx].mean(axis=1)
        kineticEnergies += 0.5 * eleMass * (eleVel**2).sum(axis=-1)

    print(f"Integrated kinetic energy: {sum(kineticEnergies)*dt:.4e}")

    sim.visualize(q, f"{folder}/frames/frame.png")
    plot_kineticEnergy(kineticEnergies, num_frames, dt, f"{folder}/kinetic_energy.png")

    if visualization:
        export_mp4(f"{folder}/frames", f"{folder}/surfacemuscle.mp4", int(1/dt))

    return sum(kineticEnergies) * dt


def main (num_frames, dt, visualization=False, sweep=False):
    dx = 0.1e-3
    shapeParams = {
        # 'nBridges': 3,
        # 'bridgeHeight': 2*dx,
        # 'bridgeRatio': 0.3
    }

    if sweep:
        M, N = 12, 4
        widthscale = np.linspace(2/15, 13/15, M)
        # widthscale = np.arange(2, 29, 29//M) * dx
        M = len(widthscale)
        heightscale = np.linspace(1, 4, N)
        w, h = np.meshgrid(widthscale, heightscale)
        params = np.stack([h.flatten(), w.flatten()], axis=1)
        energies = np.empty(M*N)
        energies[:] = np.nan

        results = []
        for i, (hR, wR) in enumerate(params):
            shapeParams['bridgeRatio'] = wR
            shapeParams['nBridges'] = int(hR)
            # shapeParams['bridgeHeight'] = wR

            if i > 0 and hR == 1:
                energies[i] = energies[i-1]
                continue

            # Check feasibility of bridge height, holes at least 1dx.
            # if int(hR) * wR + (int(hR)-1) * dx > 6e-3:
            #     continue

            print(f"Number of Bridges: {hR:.2f} - Bridge Height: {wR:.2e}")
            kE = np.nan
            try:
                kE = run_sim(num_frames, dt, visualization, shapeParams)
            except Exception as e:
                print(e)
                print(f"Simulation failed for bridge ratio {wR:.2f} and number of bridges {hR:.2f}")
            energies[i] = kE
            
            # plot heatmap of kinetic energies
            plotEnergies = np.array(energies).reshape(N, M)
            fig, ax = plt.subplots(figsize=(88*mm, 70*mm))
            c = ax.imshow(plotEnergies, cmap='jet', origin='lower', aspect='auto')
            ax.set_xticks(np.arange(0, M, 2))
            ax.set_yticks(np.arange(0, N, 1))
            # ax.set_xticklabels(map(FormatStrFormatter('%.2f').format_data, 1e3*widthscale[::2]))
            ax.set_xticklabels(map(FormatStrFormatter('%.2f').format_data, widthscale[::2]))
            ax.set_yticklabels(map(FormatStrFormatter('%.2f').format_data, heightscale[::1]))
            ax.set_xlabel("Bridge Height (mm)")
            ax.set_ylabel("Number of Bridges (-)")
            ax.set_title("Integrated Kinetic Energy (J s)")
            fig.colorbar(c)
            fig.savefig("heatmap.png", dpi=300, bbox_inches='tight')
            fig.savefig("heatmap.pdf", bbox_inches='tight')
            plt.close()

            # Store results
            results.append([wR, hR, kE])
            np.savetxt("sweep.txt", results)

    else:
        startTime = time.time()
        run_sim(num_frames, dt, visualization, shapeParams)
        print(f"Simulation took {time.time()-startTime:.2f}s")



if __name__ == "__main__":
    num_frames = 40
    dt = 0.01

    # Visualization slows down simulation a lot, can disable it if you only want to plot movement of Center of Mass (COM).
    main(num_frames, dt, visualization=False, sweep=True)

