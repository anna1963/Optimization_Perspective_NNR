import numpy as np
import simpeg
from SimPEG.utils import sdiag

from collections import OrderedDict
from discretize import TensorMesh
from discretize.utils import volume_average
import jax
jax.config.update("jax_enable_x64", True)
import pandas as pd
from pathlib import Path
import copy
from time import time

from muon_forward_modelling import MuonSensor
import muon_forward_modelling.simulations as muon_simulation

# %matplotlib widget
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import os
# from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pickle
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

z = np.linspace(-1,1,30)
y = np.linspace(-1,1,100)
x = np.linspace(-1,1,100)
input_list = []
index = 0
for i in z:
  for k in y:
    for t in x:
      input_list.append([i,k,t])
      index += 1
input_ = torch.tensor(input_list, dtype=torch.float32).to(device)
def input_mapping(x, B):
  if B is None:
    return x
  else:
    x_proj = (np.pi*x) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)
np.random.seed(0)
mapping_size = 128
scale = 2.#2.
B_gauss = np.random.normal(size = (mapping_size, 3))*scale
_x = input_mapping(input_.detach().cpu(), B_gauss)
input_pos=torch.tensor(_x, dtype=torch.float32).to(device)

class NFTOMO(nn.Module):
    def __init__(self):
         super(NFTOMO, self).__init__()
         self.b1 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b2 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b3 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b6 = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b7 = nn.Sequential(nn.Linear(128,64), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b8 = nn.Sequential(nn.Linear(64,1),nn.Tanh())

    def forward(self, x):
        """
        x_y_z = (vec(x), y(value), z(kernal))

        vec(x) is from the scaled input (-1,1), value is from y_list, kernal is from z_list

        """

        x1 = self.b1(x)

        x2 = self.b2(x1)

        x3 = self.b3(x2)

        x6 = self.b6(x3)

        _x7 = self.b7(x6)

        x7 = self.b8(_x7)

        return x7*2+2 #(0,4) bound

model = NFTOMO().to(device)

# build muon simulator
#root_data_dir = Path("./inputs")

## Create muon sensors
#df_sensor_locs = pd.read_csv(root_data_dir.joinpath("Ideon_detectors.csv"), sep=",", header=0)
sensors = OrderedDict()

n_bins_x = 60 #ideally 50+
n_bins_y = 60 #ideally 50+
fov_max_tan_theta = 1.2 # max tan(theta) = 1.7, corresponds to a max muon zenith angle of ~60 degrees
radiograph_x_axis = np.linspace(-fov_max_tan_theta, fov_max_tan_theta, n_bins_x+1)
radiograph_y_axis = np.linspace(-fov_max_tan_theta, fov_max_tan_theta, n_bins_y+1)

# Populate MuonSensor objects with location and radiograph grid
for i, row in df_sensor_locs.iterrows():
    loc = np.array([row['X'], row['Y'], row['Z']])
    sensors[f"d_{i}"] = MuonSensor(loc, radiograph_x_axis, radiograph_y_axis)

#mesh_file = "./inputs/Creighton_muon_20m.msh"  #using coarse model here, 10m model also provided
#model_file = "./inputs/Creighton_muon_20m_GT_2.den"

mesh = TensorMesh.read_UBC(mesh_file)

true_model =  mesh.read_model_UBC(model_file)

is_active = true_model > 0.0
active_map = simpeg.maps.InjectActiveCells(mesh, is_active, valInactive=0.0)

day = 86400.0 # number of seconds in a day
exposure_time = 180 * day # 180, Exposure time in seconds
np.random.seed(0)
start_time = time()
muon_counts_simulator = muon_simulation.MuonCountSimulation(
    mesh, sensors, exposure_time, n_rays_per_radiograph_pixel=10, model_map=active_map
)
end_time = time()
print('muon simulator is ready!' + ' Time needed is '+str(end_time-start_time))

#load data and define W
d_muon_counts_target = muon_counts_simulator.get_data(true_model[is_active])

air_density = 0.0
background_density = 2.95
reference_density_model = np.full(mesh.n_cells, air_density)
reference_density_model[is_active] = background_density
d_muon_counts_reference = muon_counts_simulator.get_data(reference_density_model[is_active])

simpeg_data_container = simpeg.Data(
    survey = muon_counts_simulator.survey,
    dobs = d_muon_counts_target,
    standard_deviation = np.sqrt(d_muon_counts_reference), 
)

W = sdiag(1 / (simpeg_data_container.standard_deviation))

#get the optimial weights
directory_name = 'Gaussian_PE_sigma_2_20000_epochs'
model.load_state_dict(torch.load(directory_name+'/weights.pt'))

ini_weight = [p.data for p in model.parameters()]

# Calculate the consine similarity of the two directions
def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)

#define loss
def criterion(network, muon_counts_simulator, simpeg_data_container, W, input):
    """
    network: the model with the shifted weight loaded
    muon_counts_simulator, simpeg_data_container, W: defined based on the survey geometry and field data
    input: input_z (for CNN)
    """
    output = network(input)
    m = output.detach().cpu().flatten().numpy()[is_active]
    dif_=W*muon_counts_simulator.residual(m, simpeg_data_container.dobs)
    diff = 0.5*np.vdot(dif_,dif_)
    return diff

#define the steps on the two directions
d_1_coordinates= np.linspace(-2., 2., 101)
d_2_coordinates= np.linspace(-2., 2., 101)

x_1, x_2 = np.meshgrid(d_1_coordinates, d_2_coordinates)


#define the weights (change weights along directions with a step size)
def set_weights(ind_1, ind_2, net, x_1, x_2, d_1, d_2, ini_weight):
    """
    Change weights of the ML model along directions with a step size.
    """
    # d_x_1 = x_1[ind_1][ind_2]* d_1
    # d_x_2 = x_2[ind_1][ind_2]* d_2
    changes = [d1*x_1[ind_1][ind_2] + d2*x_2[ind_1][ind_2] for (d1, d2) in zip(d_1, d_2)]
    for (p, w, d) in zip(net.parameters(), ini_weight, changes):
        p.data = w + torch.Tensor(d).type(type(w)).to(device)

#loss map generation starts!
direction_seed_list = [42, 123]
for i in range(len(direction_seed_list)):
    seed_number = int(direction_seed_list[i])
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    d_1=[torch.randn(w.size()).to(device).flatten() for w in ini_weight]
    d_2_=[torch.randn(w.size()).to(device).flatten() for w in ini_weight]

    d_2 = []
    for _d_1, _d_2 in zip(d_1, d_2_):
        d_2.append(_d_2 - (torch.dot(_d_1,_d_2)/torch.dot(_d_1,_d_1))*_d_1)

    # Nomalizing the directions using 'filter' node
    for d, w in zip(d_1, ini_weight):
        d.mul_(w.norm()/(d.norm() + 1e-10))
    for d, w in zip(d_2, ini_weight):
        d.mul_(w.norm()/(d.norm() + 1e-10))

    # Calculate the consine similarity of the two directions
    vec1 = nplist_to_tensor(d_1)
    vec2 = nplist_to_tensor(d_2)
    cos_similarity = torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm()).item()
    print(cos_similarity)

    for i in range(len(d_1)):
        d_1[i] = d_1[i].reshape(ini_weight[i].size())
        d_2[i] = d_2[i].reshape(ini_weight[i].size())

    loss_mesh = np.ones(x_1.shape)
    start = time()
    for ind_1 in range(101):
        for ind_2 in range(101):
            set_weights(ind_1, ind_2, model, x_1, x_2, d_1, d_2, ini_weight)
            model.eval()
            with torch.no_grad():
                loss_ = criterion(model, muon_counts_simulator, simpeg_data_container, W, input_pos)
                loss_mesh[ind_1][ind_2] = loss_
    end = time()
    print(end-start)

    import pickle
    pkl_name = directory_name+'/vis_NN_'+str(seed_number)+'_large_range_2_orthogonal.pkl'
    f = open(pkl_name, 'wb')
    pickle.dump(d_1, f)
    pickle.dump(d_2,f)
    pickle.dump(loss_mesh,f)
    pickle.dump(cos_similarity,f)
    f.close()

    fig = plt.figure()
    CS = plt.contour(x_1,x_2, np.log(loss_mesh), cmap='summer', levels=np.arange(min(np.log(loss_mesh.flatten())), max(np.log(loss_mesh.flatten())), .5))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(directory_name+'/NN_'+str(seed_number)+ '_large_range_vis_2_orthogonal.pdf', dpi=300,
            bbox_inches='tight', format='pdf')
