# -----------------------------------------------------------------------------
# We plot the loss and gradients of a simple example of a 3D beam with 
# Dirichlet boundary conditions. We optimize the external force to reach a 
# target position using the backward pass of DiffPD. The landscape is plotted 
# on a linear scale for external forces.
# -----------------------------------------------------------------------------

from env_beam import BeamSimulation

import time
import torch
import numpy as np
import matplotlib.pyplot as plt




def main (N, num_frames, dt):
    folder = 'loss_landscape1D'
    beam = BeamSimulation(folder, gravity=False)


    ### 1D optimization example
    values = np.linspace(-10, 10, N).astype(np.float64)

    loss_history = []
    grad_history = []
    for val in values:
        params = torch.tensor([val], requires_grad=True)

        ### Define target location
        target = beam.q0.view(-1,3).mean(axis=0) + torch.Tensor([0, 0, -0.05])
        
        start_time = time.time()

        q, v = beam.q0, beam.v0
        f_ext = torch.zeros(beam.dofs, dtype=torch.float64)
        f_ext[2::3] += params
        loss = 0
        for i in range(num_frames):
            q, v = beam.forward(q, v, f_ext=f_ext, dt=dt)
            loss += ((q.view(-1, 3).mean(axis=0) - target)**2).sum()
        
        loss /= num_frames 

        # Backward gradients so we know which direction to update parameters
        loss.backward()

        loss_history.append(loss.item())
        grad_history.append(params.grad[0].detach().cpu().numpy())
    
        with np.printoptions(precision=3):
            print(f"Time: {(time.time()-start_time):.2f}s - Loss {loss.item():.6e} - f_ext: {params[0].detach().cpu().numpy():.4f} - grad: {params.grad[0].detach().cpu().numpy():.4e}")


        ### Plotting Loss Landscape
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(values[:len(loss_history)], loss_history)
        ax.set_xlim([values.min(), values.max()])
        ax.set_title("Loss Landscape")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Loss")
        ax.grid()
        fig.savefig(f"{folder}/loss_landscape.png", dpi=300, bbox_inches='tight')
        plt.close()

        if len(grad_history) > 2:
            ### Plotting Gradient Landscape
            # Compute numerical gradient with central finite difference
            num_grad = (np.array(loss_history)[2:] - np.array(loss_history)[:-2]) / (values[2:len(loss_history)] - values[:len(loss_history)-2])
            # Forward difference for first value
            num_grad = np.insert(num_grad, 0, (np.array(loss_history)[1] - np.array(loss_history)[0]) / (values[1] - values[0]))
            # Backward difference for last value
            num_grad = np.insert(num_grad, -1, (np.array(loss_history)[-1] - np.array(loss_history)[-2]) / (values[len(loss_history)-1] - values[len(loss_history)-2]))

            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(values[:len(loss_history)], grad_history, label="Analytical Gradient")
            ax.plot(values[:len(loss_history)], num_grad, label="Numerical Gradient")
            ax.set_xlim([values.min(), values.max()])
            ax.set_title("Gradient Landscape")
            ax.set_xlabel("Parameter")
            ax.set_ylabel("Gradient")
            ax.grid()
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2, fancybox=True, shadow=True)
            fig.savefig(f"{folder}/grad_landscape.png", dpi=300, bbox_inches='tight')
            plt.close()


        ### Colored Loss Landscape
        neg_idx = np.where(np.array(grad_history)<0)[0].astype(int)
        pos_idx = np.where(np.array(grad_history)>0)[0].astype(int)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.scatter(values[neg_idx], np.array(loss_history)[neg_idx], color='red', s=10, label="Negative Gradient")
        ax.scatter(values[pos_idx], np.array(loss_history)[pos_idx], color='blue', s=10, label="Positive Gradient")
        ax.plot(values[:len(loss_history)], loss_history, color='black', linewidth=1, markersize=1)

        ax.set_xlim([values.min(), values.max()])
        ax.set_title("Directional Loss Landscape")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Loss")
        ax.grid()
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2, fancybox=True, shadow=True)
        fig.savefig(f"{folder}/lossgrad_landscape.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{folder}/lossgrad_landscape.pdf", bbox_inches='tight')
        plt.close()

        np.savetxt(f'{folder}/params.csv', values, delimiter=',')
        np.savetxt(f'{folder}/loss.csv', np.array(loss_history), delimiter=',')
        np.savetxt(f'{folder}/grad.csv', np.array(grad_history), delimiter=',')



if __name__ == "__main__":
    N = 100
    num_frames = 50
    dt = 0.01

    main(N, num_frames, dt)