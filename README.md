# Differentiable Projective Dynamics

On this tutorial branch you will find several simple examples demonstrating the capabilities of DiffPD. It is an easy way to get started and understand all different parts of DiffPD.

Note: this is a fork of https://github.com/mit-gfx/diff_pd


## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)


## Installation
```
git clone --recursive git@github.com:srl-ethz/diff_pd.git
cd diff_pd
conda env create -f environment.yml
conda activate diff_pd
./install.sh
```
We run optimization with PyTorch, hence, depending on your system, a different PyTorch version might be required for different CUDA installations. Something like `pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html` should do the job.


If you would like to enable multi-threading, set the thread_ct in the options object in the python script. The examples below all use a default of 8 threads for parallel processes. Using 1 will force the program to run sequentially.


## Examples
Navigate to the `python/example` path and run `python [example_name].py` where the `example_name` could be the following:

- `simple_forward.py`: Run the example without any arguments and it will generate a falling beam under gravity. The visualization takes a while, but will in the end create a video. A plot is also created showing the movement of the center of mass in vertical direction. We use the "Beam" environment, which creates a 10cm x 10cm x 30cm beam made of 0.1MPa elastic material, with one face fixed on a wall.
- `simple_backward.py`: Optimize the external force applied on the 3D beam to reach a target location leveraging the differentiable nature of DiffPD.
- `loss_landscape1D.py`: Runs a sweep of 1D parameters for the 3D beam, computing the loss and gradients for each parameter and plotting them. Shows how the optimization "landscape" looks like, and compares the DiffPD gradients to numerical gradients.
- `fish_fext_optim.py`: Loads a triangular mesh from an external file and optimizes the external force (time-dependent in x-direction) to get the center of mass position to reach a target location. 
