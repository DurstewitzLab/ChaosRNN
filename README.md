 *On the difficulty of learning chaotic dynamics with RNNs*
------
Code for the training of RNNs with sparsely forced BPTT. 

This folder contains the python code, data files and plots from 

["On the difficulty of learning chaotic dynamics with RNNs", Jonas Mikhaeil, Z. Monfared and D. Durstewitz](https://neurips.cc/virtual/2022/poster/53371).

This package is distributed under the terms of the GNU GPLv3 & Creative Commons Attribution License. Please credit the source and cite the reference above when using the code in any from of publication.

--------------
main.py: Starts individual runs \
ubermain.py: Allows to start multiple runs simultanously. Used to sweep parameters, such as n_interleave (called learning interval $tau$ in the paper) \
main_eval.py: Evaluates the reconstruction quality of (multiple) trained models and creates files such as klx.csv, which are used to create the plots in the paper

CreateFigures.ipyn : code to create the figures from the csv files created in the model evaluation. 

Datasets: contains all datasets.

Figures: contains all figures. 
