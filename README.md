# Optimization_Perspective_NNR
This is the repository for the codes to reproduce results presented in the paper "Understanding Neural Network Reparameterization in Geophysical Inversion from an Optimization Perspective"

## Summary
Neural network reparameterization methods, also called test-time learning (TTL) methods or implicit neural representation methods, have recently emerged as a promising approach for geophysical inversion. These methods provide implicit regularization effects, which can be in addition to explicit regularization. Previous studies emphasize the benefits of designing more effective neural network architectures/frameworks. However, how they learn during inversions, and what shapes their learning dynamics, has received little attention. We translate understanding from regularization in supervised learning to TTL, and find that the benefits come not only from inductive bias of machine learning models, but also from implicit bias of the optimizer. We find that the learning rate can be tuned to control smoothness of boundaries and compactness of target in inverted geophysical models. We compare sharpness values (curvature of the loss landscape at a limit) and show the catapult effect (loss increases at beginning of learning, before decreasing and converging). We find that flatter minima are associated with sharper boundaries of the recovered target, which always come from an inversion with a larger learning rate. We argue that the effect of a large learning rate is associated with two-phase learning, which is a process that allows the optimization to escape a sharp region, which is observed and explained in supervised learning methods. We also find that positional encoding introduces an inductive bias that affects the convergence rate and the features recovered. We test our hypotheses in three-dimensional muon tomography surveys, two-dimensional direct current resistivity surveys, and two-dimensional seismic tomography surveys. 

## Contents
'Seismic' directory:
- Case_2_test_no_PE_final_orthogonal.ipynb: Reproduce the NFs-Inv result for cross-hole seismic tomography inversion without postional encoding. Results are stored in the 'no_PE_results' folder
- Case_2_test_Gaussian_PE_orthogonal.ipynb: Reproduce the NFs-Inv result for cross-hole seismic tomography inversion with Gaussian positional encoding. Results are stored in the Gaussian_PE_results.
- utrue.pkl: The true subsurface velocity model.

All notesbook can be run on Colab (!pip install is included in the first cell). For more information on running Colab notebooks, see https://colab.google.

'DCR' (Direct Current Resistivity) directory:
- Generate_Synthetic_Voltages.ipynb: Runs the forward simulation for the DCR cases shown in the paper. 5% Gaussion noises are added to the synthetic data.
- 'Case_1.1_truncated_z_225' folders: Contain the voltage measurements and topology information.
- 'DCR_* .py' files: Reproduce the NFs-Inv results. The final results (both the final predicted conductivity model and the weights) will be stored in the proper format with name defined in the corresponding 'DCR_*' folders.
- 'hessian* .py' files: Calculate the final Hessian.

Usage of 'DCR' inversion:

Dependencies are specified in [requirements.txt](/requirements.txt)

```
pip install -r requirements.txt
```
You can then clone this repository. From a command line, run

```
git clone https://github.com/anna1963/Optimization_Perspective_NNR.git
```

Then `cd` into the `DCR` directory:

```
cd DCR
```

To setup your software environment, we recommend you use the provided conda environment

```
conda env create -f environment.yml
conda activate NNR-environment
```

'Muon' directory:

The forward simulation code for the 3D muon tomography example belongs to Ideon and is not publicly available, but is described in Schouten et al. (2022). The inversion and analysis codes used in this study are stored in the following directories:

'Block Cave' folder:
- complicated_deepCove_block_cave_MLP_10_layers_no_PE_6e4_epoch_82e2_30_days.py: Reproduce the NFs-Inv result for Example 1: Block Cave without positional encoding. Results are stored in 'Muon_hete_block_cave_hat_gpu_no_PE_10_layers_6e_4_epoch_82e2_seed_0_30_days' folder.
- complicated_deepCove_block_cave_MLP_10_layers_Gaussian_PE_2_1e3_epoch_13e2_30_days.py: Reproduce the NFs-Inv result for Example 1: Block Cave with Gaussian PE (sigma = 2) for Example 1: Block Cave without positional encoding. Results are stored in 'Muon_hete_block_cave_hat_gpu_Gaussion_PE_2_10_layers_1e_3_seed_0_epoch_13e2_30_days' folder.
- homogenous_Block_Cave: The codes related to the simplified density model to investigate the large learning rate effects.

'Creighton Mine' folder:
The inversion codes and visualization codes for the Example 2:Creighton Mine with different positional encoding.


## License
These notebooks are licensed under the [MIT License](/LICENSE).
