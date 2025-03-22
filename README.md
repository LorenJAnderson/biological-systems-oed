# biological-systems-oed
Code for paper *Optimal Parameter Estimation of Biological Systems through 
Deep Reinforcement Learning* by Fadil Santosa and Loren Anderson. Presented 
at ICMLA 2024.

DOI: 10.1109/ICMLA61862.2024.00199

URL: https://ieeexplore.ieee.org/abstract/document/10903242

### Code Structure

The three reinforcement learning environments are located in the 
`environments` folder. The algorithm files are located in the `algorithms` 
folder. This code has been tested on Ubuntu 22.04 OS. Some file paths in 
the Python scripts may need to be changed if running on a different OS. A 
folder called `data` may need to be manually created at the same level of 
the `environments` and `algorithms` folders; otherwise, it may be created 
automatically when the algorithms save data.   

### Running Algorithms

Greedy design is run through the `algorithms/greedy_design.py` file, and 
reinforcement learning is run through the `algorithms/reinforcement 
learning.py` file. The `algorithms/batch_design.py` file can be used to 
determine scores for batch design, random design, and endpoint design. The `algorithms` files need to be modified to run the algorithms 
on different environments and different reward functions. The possible 
environment names are `diffusion1d`, `source2d`, and `lotka_volterra`. The 
possible reward functions are `kl`, `max_proximity`, and `max_forward`. 