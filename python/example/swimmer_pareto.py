# -----------------------------------------------------------------------------
# Sweeping for two metrics.
# -----------------------------------------------------------------------------

import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from py_diff_pd.common.display import export_mp4

from swimmer import Swimmer, smoothstep


plt.rcParams.update({'font.size': 10})     # Font size should be max 7pt and min 5pt
plt.rcParams.update({'pdf.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({'ps.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": False})
plt.rcParams.update({"font.family": 'Arial'})
mm = 1/25.4


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_sim (num_frames, dt, visualization=False, shapeParams=None):
    folder = 'swimmer'
    sim = Swimmer(folder, shapeParams)

    q, v = sim.q0, sim.v0

    rampTime = 0.5*dt*num_frames/2
    maxVal = 1.0
    actuation = smoothstep(np.linspace(0, dt*num_frames/2, num_frames//2), x_min=0.0, x_max=rampTime, N=1)
    actuation *= maxVal
    actuation = torch.as_tensor(np.concatenate([1-actuation, actuation]), dtype=torch.float64)

    end_vis = time.time()
    velocities = []
    for frame in range(num_frames):
        start_time = time.time()
        if visualization and frame % 10 == 0:
            sim.visualize(q, f"{folder}/frames/frame_{frame:04d}.png")
            # Time including visualization
            print(f"Frame [{frame+1}/{num_frames}]: {1000*(time.time()-end_vis):.2f}ms (+ {1000*(end_vis-start_time):.2f}ms for visualization)")
        end_vis = time.time()

        # Forward pass
        act = actuation[frame] * torch.ones(sim.act_dofs, dtype=torch.float64)
        q, v = sim.forward(q, v, act, dt=dt)
        velocities.append(v)

    # Compute total deformation
    totalDeformation = max(np.linalg.norm((q - sim.q0).reshape(-1, 3), axis=-1))
    # Compute ratio of contraction time vs expansion time
    print(f"Max deformation: {1e3*totalDeformation:.4e}mm")

    sim.visualize(q, f"{folder}/frames/frame.png")

    fig, ax1 = plt.subplots(figsize=(88*mm, 70*mm))
    ax2 = ax1.twinx()
    timeAxis = np.linspace(0, dt*num_frames, num_frames)
    ax1.plot(timeAxis, np.mean(np.linalg.norm(np.stack(velocities, axis=0).reshape(len(velocities), -1, 3), axis=-1), axis=1), color='tab:blue')
    ax2.plot(timeAxis, actuation, color='tab:red')    
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (m/s)", color='tab:blue')
    ax2.set_ylabel("Actuation (-)", color='tab:red')
    ax1.grid()
    plt.savefig("velocities_actuation.png", dpi=300, bbox_inches='tight')
    plt.close()

    if visualization:
        export_mp4(f"{folder}/frames", f"{folder}/swimmer.mp4", int(1/dt))

    return 1e3*totalDeformation


def main (num_frames, dt, visualization=False, sweep=False):
    shapeParams = {
        'widthRatio': 1.0,
        'heightRatio': 0.5
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
    num_frames = 100
    dt = 0.01

    # Visualization slows down simulation a lot, can disable it if you only want to plot movement of Center of Mass (COM).
    main(num_frames, dt, visualization=False, sweep=False)

