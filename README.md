# Optimization_Perspective_NNR
This is the repository for the codes to reproduce results presented in the paper "Understanding Neural Network Reparameterization in Geophysical Inversion from an Optimization Perspective"

## Summary
Neural network reparameterization methods, also called test-time learning (TTL) methods or implicit neural representation methods, have recently emerged as a promising approach for geophysical inversion. These methods provide implicit regularization effects, which can be in addition to explicit regularization. Previous studies emphasize the benefits of designing more effective neural network architectures/frameworks. However, how they learn during inversions, and what shapes their learning dynamics, has received little attention. We translate understanding from regularization in supervised learning to TTL, and find that the benefits come not only from inductive bias of machine learning models, but also from implicit bias of the optimizer. We find that the learning rate can be tuned to control smoothness of boundaries and compactness of target in inverted geophysical models. We compare sharpness values (curvature of the loss landscape at a limit) and show the catapult effect (loss increases at beginning of learning, before decreasing and converging). We find that flatter minima are associated with sharper boundaries of the recovered target, which always come from an inversion with a larger learning rate. We argue that the effect of a large learning rate is associated with two-phase learning, which is a process that allows the optimization to escape a sharp region, which is observed and explained in supervised learning methods. We also find that positional encoding introduces an inductive bias that affects the convergence rate and the features recovered. We test our hypotheses in three-dimensional muon tomography surveys, two-dimensional direct current resistivity surveys, and two-dimensional seismic tomography surveys. 

## Contents
'Seismic' directory:
- no_PE.ipynb: Reproduce the NFs-Inv result for Case 1 (cross-hole seismic tomography inversion without postional encoding).
- Gaussian_PE.ipynb: Reproduce the NFs-Inv result for Case 2 (cross-hole seismic tomography inversion with Gaussian positional encoding). 

All notesbook can be run on Colab (!pip install is included in the first cell). For more information on running Colab notebooks, see https://colab.google.

'DCR' (Direct Current Resistivity) directory:
- Generate_Synthetic_Voltages.ipynb: Runs the forward simulation for the DCR cases shown in the paper. 5% Gaussion noises are added to the synthetic data.
- 'Case_*_truncated_z_225' folders: Contain the voltage measurements and topology information for each DCR trial.
- 'Case_* .py' files: Reproduce the NFs-Inv result for each DCR trial. The final results (both the final predicted conductivity model and the weights) will be stored in the proper format with name defined in the corresponding ' NF_*' folders.

'Muon' directory:

'Creighton Mine' folder:

'Block Cave' folder:

'Hessian_analysis' directory:

'loss_visualization' directory:


## License
These notebooks are licensed under the [MIT License](/LICENSE).
