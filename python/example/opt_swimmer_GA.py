# -----------------------------------------------------------------------------
# Genetic algorithm to optimize the shape of a swimmer.
# -----------------------------------------------------------------------------

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import pygad
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from py_diff_pd.common.display import export_mp4
from py_diff_pd.common.common import create_folder

from swimmer import Swimmer, smoothstep, plot_kineticEnergy


plt.rcParams.update({'font.size': 7})     # Font size should be max 7pt and min 5pt
plt.rcParams.update({'pdf.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({'ps.fonttype': 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})
mm = 1/25.4



SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_FRAMES = 40
DT = 0.01


def run_sim (sim, num_frames, dt, visualization=False, shapeParams=None):
    q, v = sim.q0, sim.v0

    rampTime = 0.25
    maxVal = 2.0
    actuation = smoothstep(np.linspace(0, dt*num_frames, num_frames), x_min=0.0, x_max=rampTime, N=1)
    actuation *= maxVal
    actuation = torch.as_tensor(1-actuation, dtype=torch.float64)

    velocities = []
    for frame in range(num_frames):
        act = actuation[frame] * torch.ones(sim.act_dofs, dtype=torch.float64)

        q, v = sim.forward(q, v, act, dt=dt)
        velocities.append(v)

        # Time including visualization
        # print(f"Frame [{frame+1}/{num_frames}]: {1000*(time.time()-end_vis):.2f}ms (+ {1000*(end_vis-start_time):.2f}ms for visualization) -\t Kinetic Energy: {kineticEnergy:.4e}")

    # Compute kinetic energy
    kineticEnergies = np.zeros(len(velocities))
    for i in range(sim.deformable.NumOfElements()):
        vertexIdx = np.array(sim.deformable.mesh().py_element(i))
        eleMass = sim.deformable.density() * sim.deformable.mesh().element_volume(i)
        # Mass doesn't change during simulation since volume does not change.
        eleVel = np.stack(velocities, axis=0).reshape(len(velocities),-1,3)[:,vertexIdx].mean(axis=1)
        kineticEnergies += 0.5 * eleMass * (eleVel**2).sum(axis=-1)
    
    # if visualization:
    #     export_mp4(f"{folder}/frames", f"{folder}/swimmer.mp4", int(1/dt))

    return kineticEnergies, q, v


def fitness_func (gaInstance, solution, solution_idx):
    shapeParams = {
        'muscleIdx2d': np.where(solution)[0]
    }

    folder = 'swimmer'
    create_folder(f"{folder}/runs", exist_ok=True)
    sim = Swimmer(folder, shapeParams)

    kE, q, v = run_sim(sim, NUM_FRAMES, DT, visualization=False, shapeParams=shapeParams)

    # plot_kineticEnergy(kE, NUM_FRAMES, DT)

    # Store result
    filename = f"{folder}/runs/swimmer_{sum(kE)*DT:.1e}"
    if not os.path.isfile(f"{filename}.png"):
        sim.visualize(q, f"{filename}.png")
        np.savetxt(f"{filename}.txt", solution)

    return sum(kE) * DT


maxFitnessHist = [0]
meanFitnessHist = [0]
lastTime = time.time()
def on_generation (gaInstance):
    global maxFitnessHist, meanFitnessHist, lastTime
    sol, solFitness, _ = gaInstance.best_solution(gaInstance.last_generation_fitness)
    maxFitnessHist.append(solFitness)
    meanFitnessHist.append(np.mean(gaInstance.last_generation_fitness))

    print(f"Generation {gaInstance.generations_completed} in {time.time() - lastTime:.2f}s: \tBest Fitness: {solFitness:.4e} \t- Change: {solFitness - maxFitnessHist[-2]:.2e} \t- Average Fitness: {meanFitnessHist[-1]:.4e}")
    # print(f"Best solution: {sol}")
    lastTime = time.time()

    fig, ax = plt.subplots(figsize=(88*mm,60*mm))
    ax.plot(maxFitnessHist, label="Best Fitness")
    ax.plot(meanFitnessHist, label="Average Fitness")
    ax.set_xlabel("Generation (-)")
    ax.set_ylabel("Integrated Kinetic Energy (J s)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.legend(loc="lower right")
    ax.grid()
    fig.savefig("fitness.png", dpi=300, bbox_inches='tight')
    plt.close()


def main (visualization=False, sweep=False):
    shapeParams = {
        'muscleIdx2d': np.random.choice(np.arange(30*10), 80, replace=False),
        # 'widthRatio': 1.0,
        # 'heightRatio': 1.0,
    }

    if sweep:
        gaInstance = pygad.GA(
            num_genes=30*10,
            num_generations=5000,
            num_parents_mating=4, # Number of solutions to be selected as parents in the mating pool.
            sol_per_pop=20, # Number of solutions in the population.
            keep_elitism=3, # Number of best solutions to be copied to the next generation.
            parent_selection_type = "sss", # Type of parent selection.
            keep_parents=1, # Number of parents to keep in the next generation.
            crossover_probability=0.8, # Probability of crossover.
            mutation_percent_genes=20, # Percentage of genes to mutate.
            mutation_probability=0.9, # Probability of mutation within the genes to mutate.
            gene_type=np.int8,
            gene_space=[0,1],
            random_seed=SEED,
            save_solutions=True,
            fitness_func=fitness_func,
            on_generation=on_generation,
        )
        gaInstance.run()

    else:
        folder = 'swimmer'
        create_folder(f"{folder}/runs", exist_ok=True)
        sim = Swimmer(folder, shapeParams)
        kE, q, v = run_sim(sim, NUM_FRAMES, DT, visualization, shapeParams)

        sim.visualize(q, f"{folder}/runs/swimmer_{sum(kE):.4f}.png")


if __name__ == "__main__":
    main(visualization=False, sweep=True)