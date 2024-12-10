# -----------------------------------------------------------------------------
# Example of a 2D beam with Dirichlet boundary conditions falling under gravity
# -----------------------------------------------------------------------------
from env_beam2D import BeamSimulation

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from py_diff_pd.common.display import export_mp4


def main (num_frames, dt, visualization):
    folder = 'f_ext_testing'
    beam = BeamSimulation(folder, gravity=False, fixed=False)

    q, v = beam.q0, beam.v0
    force_per_volume = 1000 / 0.5    # N/m^2, # Area of whole structure is 0.5m^2
    f_ext = np.zeros(beam.dofs).reshape(-1,2)
    elements_per_vertex = np.zeros(beam.dofs//2).reshape(-1,1)
    for i in range(beam.mesh.NumOfElements()):
        for v_idx in beam.mesh.py_element(i):
            f_ext[v_idx,1] += beam.mesh.element_volume(i) * force_per_volume
            elements_per_vertex[v_idx] += 1
    f_ext = torch.tensor(np.where(elements_per_vertex, f_ext/elements_per_vertex, f_ext)).flatten()

    com = []
    for i in range(num_frames):
        start_time = time.time()

        if visualization:
            beam.visualize(q, f"{folder}/{beam.method}/frame_{i:04d}.png")
        end_vis = time.time()

        q, v = beam.forward(q, v, f_ext=f_ext)
        com.append(q.view(-1,2).detach().cpu().numpy().mean(0))

        # Time including visualization
        #print(f"Frame [{i+1}/{num_frames}]: {1000*(time.time()-end_vis):.2f}ms (+ {1000*(end_vis-start_time):.2f}ms for visualization)")

    if visualization:
        export_mp4(f"{folder}/{beam.method}", f"{folder}/{beam.method}.mp4", 30)

    com = np.array(com)
    ### Plotting movement of center of mass in z-direction
    fig, ax = plt.subplots(figsize=(4,3))
    x = np.linspace(0, num_frames*dt, num_frames)
    ax.plot(x, com[:,-1])
    ax.set_xlim([0, num_frames*dt])
    ax.set_title("Center of Mass Vertical Movement")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z (m)")
    ax.grid()
    fig.savefig(f"{folder}/vertical_movement.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{folder}/vertical_movement.pdf", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(4,3))
    x = np.linspace(0, num_frames*dt, num_frames)
    vel = (com[2:,-1] - com[:-2,-1])/(2*dt)
    ax.plot(x[1:-1], vel)
    ax.set_xlim([0, num_frames*dt])
    ax.set_title("Center of Mass Vertical Velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("v (m/s)")
    ax.grid()
    fig.savefig(f"{folder}/vertical_vel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Acceleration: {((vel[2:] - vel[:-2])/(2*dt)).mean():.4e}")


if __name__ == "__main__":
    num_frames = 40
    dt = 0.01

    # Visualization slows down simulation a lot, can disable it if you only want to plot movement of Center of Mass (COM).
    main(num_frames, dt, visualization=False)