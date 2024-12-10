# Bilayer Simulations

We simulate the bilayered structures using DiffPD (https://github.com/mit-gfx/diff_pd). In this document we describe the examples to run for generating the results in the paper.


## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)


## Installation
```
git clone --recursive git@github.com:srl-ethz/bilayered_optimization.git
cd bilayered_optimization
conda env create -f environment.yml
conda activate diffPD
./install.sh
```
We run optimization with PyTorch, hence, depending on your system, a different PyTorch version might be required for different CUDA installations. Something like `pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html` should do the job.


If you would like to enable multi-threading, set the thread_ct in the options object in the python script. The examples below all use a default of 8 threads for parallel processes. Using 1 will force the program to run sequentially.


## Examples
Navigate to the `python/example` path and run `python [example_name].py` where the `example_name` could be the following:

- `swimmer.py`: The most basic simulation that will generate a swimmer simulation environment, where the mesh resolution and materials are defined. The option exists to sweep over different geometries in this file.
- `opt_swimmer_GA.py`: Runs a genetic algorithm from PyGAD to optimize the swimmer defined in swimmer.py environment.
- `swimmer_stiffness_sweep.py`: Runs a sweep of the stiffness of the scaffold to see how the ratio between the muscle and scaffold stiffness impacts the final deformation.
