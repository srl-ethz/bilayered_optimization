# -----------------------------------------------------------------------------
# Example of a 3D beam with Dirichlet boundary conditions. We optimize the
# external force to reach a target position using the backward pass of DiffPD.
# -----------------------------------------------------------------------------

from env_fish import FishSimulation

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from py_diff_pd.common.display import export_mp4



def main (num_epochs, num_frames, dt, visualization):
    folder = 'fish_fext_optim'
    fish = FishSimulation(folder, gravity=False)
    
    ### LBFGS takes multiple steps during every optimization step, and in our case in particular converges faster/better. Strong-Wolfe condition makes the line-search not diverge as easily when the LR is not chosen appropriately.
    params = torch.randn([num_frames], dtype=torch.float64, requires_grad=True)
    ### Since our gradients will usually be small, we need to lower the gradient tolerance from the default 1e-5 to 1e-7.
    optimizer = torch.optim.LBFGS([params], lr=1e-2, max_iter=5, line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 1.4)

    ### Define target location
    target = fish.q0.view(-1,3).mean(axis=0) + torch.Tensor([8., 0, 0.])
    
    loss_history = []
    for epoch in range(num_epochs):
        start_time = time.time()

        def closure():
            optimizer.zero_grad()
            # Box-constraint optimization
            # with torch.no_grad():
            #     for param in params:
            #         param.clamp_(-50, 50)

            q, v = fish.q0, fish.v0
            loss = 0
            for i in range(num_frames):
                f_ext = torch.zeros(fish.dofs, dtype=torch.float64)
                f_ext[::3] += params[i]
                q, v = fish.forward(q, v, f_ext=f_ext, dt=dt)
                loss += ((q.view(-1, 3).mean(axis=0) - target)**2).sum()
            
            loss /= num_frames

            ### Additionally add condition on f_ext change to not be too sudden
            threshold = 100
            df_ext_loss = torch.maximum((params[1:] - params[:-1])**2 - threshold**2, torch.zeros_like(params[1:])).sum()
            loss += 1e-1 * df_ext_loss

            # Backward gradients so we know which direction to update parameters
            loss.backward()

            return loss

        # Actually update parameters
        loss = optimizer.step(closure)
        scheduler.step()
        loss_history.append(np.log(loss.item()))
    
        with np.printoptions(precision=3):
            print(f"Epoch [{epoch+1}/{num_epochs}]: {(time.time()-start_time):.2f}s - Loss {loss.item():.4e} - f_ext: {params[0].detach().cpu().numpy():.6f} - grad: {params.grad[0].detach().cpu().numpy():.2e}  - Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        
        ### Plotting f_ext
        x = np.linspace(0, num_frames*dt, num_frames)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(x, params.detach().cpu().numpy())
        ax.set_xlim([0, num_frames*dt])
        ax.set_title("External Force over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("f_ext")
        ax.grid()
        fig.savefig(f"{folder}/f_ext.png", dpi=300, bbox_inches='tight')
        plt.close()


        ### Early stopping
        if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-8:
            break


    ### Plotting Loss History
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(loss_history)
    ax.set_xlim([0, len(loss_history)])
    ax.set_title("Log Loss History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    fig.savefig(f"{folder}/loss_history.png", dpi=300, bbox_inches='tight')
    plt.close()

    ### Plotting Optimal f_ext
    x = np.linspace(0, num_frames*dt, num_frames)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(x, params.detach().cpu().numpy())
    ax.set_xlim([0, num_frames*dt])
    ax.set_title("Optimal External Force over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f_ext")
    ax.grid()
    fig.savefig(f"{folder}/f_ext_optimal.png", dpi=300, bbox_inches='tight')
    plt.close()


    with torch.no_grad():
        q, v = fish.q0, fish.v0

        com = []
        for i in range(num_frames):
            start_time = time.time()

            f_ext = torch.zeros(fish.dofs, dtype=torch.float64)
            f_ext[::3] += params[i]

            if visualization:
                fish.visualize(q, f"{folder}/{fish.method}/frame_{i:04d}.png", extra_points=[target])
            end_vis = time.time()
            q, v = fish.forward(q, v, f_ext=f_ext, dt=dt)

            com.append(q.view(-1,3).detach().cpu().numpy().mean(axis=0)[0])

            # Time including visualization
            print(f"Frame [{i+1}/{num_frames}]: {1000*(time.time()-end_vis):.2f}ms (+ {1000*(end_vis-start_time):.2f}ms for visualization)")

        if visualization:
            export_mp4(f"{folder}/{fish.method}", f"{folder}/{fish.method}.mp4", 30)


    ### Plotting movement of center of mass in x-direction
    fig, ax = plt.subplots(figsize=(4,3))
    x = np.linspace(0, num_frames*dt, num_frames)
    ax.plot(x, target.detach().cpu().numpy()[0]*np.ones(num_frames), linestyle='dashed', label="Target")
    ax.plot(x, com, label="Simulation COM")
    ax.set_xlim([0, num_frames*dt])
    ax.set_title("Center of Mass Horizontal Movement")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x")
    ax.grid()
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2, fancybox=True, shadow=True)
    fig.savefig(f"{folder}/horizontal_movement.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{folder}/horizontal_movement.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    num_epochs = 50
    num_frames = 100
    dt = 0.01

    # Visualization slows down simulation a lot, can disable it if you only want to plot movement of Center of Mass (COM).
    main(num_epochs, num_frames, dt, visualization=True)