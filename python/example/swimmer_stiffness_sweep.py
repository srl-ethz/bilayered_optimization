# -----------------------------------------------------------------------------
# Just a swimmer moving. And checking muscle vs scaffold stiffness.
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

from swimmer import run_sim



plt.rcParams.update({'font.size': 10})     # Font size should be max 7pt and min 5pt
plt.rcParams.update({'pdf.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({'ps.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": False})
plt.rcParams.update({"font.family": 'Arial'})
mm = 1/25.4



SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)



def main (num_frames, dt, visualization=False, sweep=False):
    shapeParams = {
        'widthRatio': 1.0,
        'heightRatio': 0.25,
        'scaffoldYM': 200e3,
        'nx': 15,
        'ny': 20,
        'nz': 2,
    }

    if sweep:
        M, N = 19, 100
        hRs = np.linspace(1/(M+1), 1.0-1/(M+1), M)
        yms = np.linspace(2.5e3, 250e3, N)
        hR, ym = np.meshgrid(hRs, yms)
        params = np.stack([ym.flatten(), hR.flatten()], axis=1)
        metrics = np.empty(M*N)
        metrics[:] = np.nan

        results = []
        for i, (ym, hR) in enumerate(params):
            shapeParams['scaffoldYM'] = ym
            shapeParams['heightRatio'] = hR

            print(f"Height Ratio: {hR:.2f} - Scaffold Young's Modulus: {ym:.2e}Pa")
            try:
                m = run_sim(num_frames, dt, visualization, shapeParams)
            except Exception as e:
                print(f"Simulation failed for height ratio {hR:.2f} and scaffold Young's Modulus {ym:.2e} with error: {e}")
            metrics[i] = m
            
            # Plot heatmap of sweep
            metric = np.array(metrics).reshape(N, M)
            fig, ax = plt.subplots(figsize=(88*mm, 70*mm))
            c = ax.imshow(metric, cmap='jet', origin='lower', aspect='auto')
            ax.set_xticks(np.arange(2, M, 4))
            ax.set_yticks(np.arange(9, N, 20))
            ax.set_xticklabels(map(FormatStrFormatter('%.2f').format_data, hRs[2::4]))
            ax.set_yticklabels(map(FormatStrFormatter('%.0f').format_data, 1e-3*yms[9::20]))
            ax.set_xlabel("Height Ratio (-)")
            ax.set_ylabel("Scaffold Young's Modulus (kPa)")
            ax.xaxis.labelpad = 5
            ax.yaxis.labelpad = 5
            # ax.set_title("Integrated Kinetic Energy (J s)")
            fig.colorbar(c)
            fig.savefig("heatmap_stiffness.png", dpi=300, bbox_inches='tight')
            fig.savefig("heatmap_stiffness.pdf", bbox_inches='tight')
            plt.close()

            # Relative y-axis
            fig, ax = plt.subplots(figsize=(88*mm, 70*mm))
            c = ax.imshow(metric, cmap='jet', origin='lower', aspect='auto')
            ax.set_xticks(np.arange(2, M, 4))
            ax.set_yticks(np.arange(9, N, 20))
            ax.set_xticklabels(map(FormatStrFormatter('%.2f').format_data, hRs[2::4]))
            ax.set_yticklabels(map(FormatStrFormatter('%.1f').format_data, yms[9::20] / 25e3))
            ax.set_xlabel("Height Ratio (-)")
            ax.set_ylabel("Relative Scaffold Young's Modulus (-)")
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10
            # ax.set_title("Integrated Kinetic Energy (J s)")
            fig.colorbar(c)
            fig.savefig("heatmap_relstiffness.png", dpi=300, bbox_inches='tight')
            fig.savefig("heatmap_relstiffness.pdf", bbox_inches='tight')
            plt.close()

            # Store results
            results.append([hR, ym, m])
            np.savetxt("sweep_stiffness.txt", results)

    else:
        startTime = time.time()
        run_sim(num_frames, dt, visualization, shapeParams)
        print(f"Simulation took {time.time()-startTime:.2f}s")



if __name__ == "__main__":
    num_frames = 40
    dt = 0.01

    # Visualization slows down simulation a lot, can disable it if you only want to plot movement of Center of Mass (COM).
    main(num_frames, dt, visualization=False, sweep=True)

