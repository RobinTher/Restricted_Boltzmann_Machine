# Restricted_Boltzmann_Machine
Run "Reproduce_figures.ipynb" to reproduce the figures of the paper https://arxiv.org/abs/2410.16150.
Markdown cells indicate which groups of cells must be ran in order to reproduced each figure.

Figures were originally made using python 3.8.5 with numpy 1.24.3, matplotlib 3.7.5, scipy 1.10.1, cmasher 1.6.3, jax 0.4.6 and torch 2.2.0+cu118.

Follow the instructions here https://jax.readthedocs.io/en/latest/installation.html to install jax and here https://pytorch.org/get-started/locally/ to install torch.

The Monte-Carlo simulations take a few hours to run. The saddle-point equations take approximately the same amount of time, except for the phase diagrams as a function
of the number of hidden units $P$, the correlation strength $c$, the temperature $T$ and the data load $\alpha$, which can take up to a day.
