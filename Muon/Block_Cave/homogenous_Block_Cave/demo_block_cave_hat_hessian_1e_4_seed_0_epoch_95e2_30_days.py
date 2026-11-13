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


import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pickle
device = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)

class NFTOMO(nn.Module):
    def __init__(self):
         super(NFTOMO, self).__init__()
         self.b1 = nn.Sequential(nn.Linear(3,128), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b2 = nn.Sequential(nn.Linear(128,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b3 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b4 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b5 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b6 = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b7 = nn.Sequential(nn.Linear(128,64), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b8 = nn.Sequential(nn.Linear(64,1))
         self.b9 = nn.Tanh()

    def forward(self, x):
        """
        x_y_z = (vec(x), y(value), z(kernal))

        vec(x) is from the scaled input (-1,1), value is from y_list, kernal is from z_list

        """

        x1 = self.b1(x)

        x2 = self.b2(x1)

        x3 = self.b3(x2)

        x4 = self.b4(x3)

        x5 = self.b5(x4)

        x6 = self.b6(x5)

        _x7 = self.b7(x6)

        x7 = self.b8(_x7)

        x8 = self.b9(2*x7)

        return x8*3+2 #(-1,5) bound

torch.manual_seed(0)
np.random.seed(0)
model = NFTOMO().to(device)
ini_weight = [p.data*1 for p in model.parameters()]

z = np.linspace(-1,1,17)
y = np.linspace(-1,1,54)
x = np.linspace(-1,1,54)
input_list = []
index = 0
for i in z:
  for k in y:
    for t in x:
      input_list.append([i,k,t])
      index += 1
input_ = torch.tensor(input_list, dtype=torch.float32).to(device)

root_data_dir = Path("../inputs")

## Create muon sensors
#df_sensor_locs = pd.read_csv(root_data_dir.joinpath("muon_detector_locations.txt"), sep="\t", header=None, names=['x', 'y', 'z'])
sensors = OrderedDict()
n_bins_x = 15
n_bins_y = 15
fov_max_tan_theta = 1.2 # max tan(theta) = 1.2, corresponds to a max muon zenith angle of ~50 degrees
radiograph_x_axis = np.linspace(-fov_max_tan_theta, fov_max_tan_theta, n_bins_x+1)
radiograph_y_axis = np.linspace(-fov_max_tan_theta, fov_max_tan_theta, n_bins_y+1)

# Populate MuonSensor objects with location and radiograph grid
for i, row in df_sensor_locs.iterrows():
    loc = np.array([row['x'], row['y'], row['z']])
    sensors[f"d_{i}"] = MuonSensor(loc, radiograph_x_axis, radiograph_y_axis)


x_min_sensors = min([sensor.loc[0] for sensor in sensors.values()])
x_max_sensors = max([sensor.loc[0] for sensor in sensors.values()])
y_min_sensors = min([sensor.loc[1] for sensor in sensors.values()])
y_max_sensors = max([sensor.loc[1] for sensor in sensors.values()])
z_min_sensors = min([sensor.loc[2] for sensor in sensors.values()])
z_max_domain = 850.0 # This deepcove example uses flat topography at 850m elevation
z_min_domain = z_min_sensors
x_min_domain = x_min_sensors - (z_max_domain - z_min_domain) * fov_max_tan_theta
x_max_domain = x_max_sensors + (z_max_domain - z_min_domain) * fov_max_tan_theta
y_min_domain = y_min_sensors - (z_max_domain - z_min_domain) * fov_max_tan_theta
y_max_domain = y_max_sensors + (z_max_domain - z_min_domain) * fov_max_tan_theta

h = 40.0 # cell size in meters
# SimPEG needs an array of cell sizes in each dimension
nx = np.ceil((x_max_domain - x_min_domain) / h).astype(int)
ny = np.ceil((y_max_domain - y_min_domain) / h).astype(int)
nz = np.ceil((z_max_domain - z_min_domain) / h).astype(int)
hx = np.full(nx, h)
hy = np.full(ny, h)
hz = np.full(nz, h)
mesh = TensorMesh([hx, hy, hz], x0=np.array([x_min_domain, y_min_domain, z_min_domain]))

#mesh_from_file = TensorMesh.read_UBC(root_data_dir.joinpath("mesh.msh"))
#basic_cave_1 = mesh_from_file.read_model_UBC(root_data_dir.joinpath("basic1_cave.den"))

basic_cave_1_fwd_mesh = volume_average(mesh_from_file, mesh, basic_cave_1)
basic_cave_1_fwd_mesh = np.clip(basic_cave_1_fwd_mesh, 0.0, np.inf)

is_active = basic_cave_1_fwd_mesh > 0.0
active_map = simpeg.maps.InjectActiveCells(mesh, is_active, valInactive=0.0)

import pickle
#f = open('../hat_cave_1'+'.pkl', 'rb')
#hat_cave_1_fwd = pickle.load(f)
#f.close()

# Create the muon simulation object. Assume 30 day exposure time
day = 86400.0 # number of seconds in a day
exposure_time = 30 * day # Exposure time in seconds
start_time = time()
np.random.seed(0)
muon_counts_simulator = muon_simulation.MuonCountSimulation(
    mesh, sensors, exposure_time, n_rays_per_radiograph_pixel=10, model_map=active_map
)
end_time = time()
print('muon simulator is ready!' + ' Time needed is '+str(end_time-start_time))
d_muon_counts_target_hat = muon_counts_simulator.get_data(hat_cave_1_fwd[is_active])

air_density = 0.0
background_density = 2.7
reference_density_model = np.full(mesh.n_cells, air_density)
reference_density_model[is_active] = background_density
d_muon_counts_reference = muon_counts_simulator.get_data(reference_density_model[is_active])

simpeg_data_container = simpeg.Data(
    survey = muon_counts_simulator.survey,
    dobs = d_muon_counts_target_hat,
    standard_deviation = np.sqrt(d_muon_counts_reference), 
)

W = sdiag(1 / (simpeg_data_container.standard_deviation))

torch.manual_seed(0)
np.random.seed(0)

model_2e_4 = NFTOMO().to(device)
directory_name = 'Muon_block_cave_hat_gpu_1e_4_seed_0_epoch_95e2_30_days'
model_2e_4.load_state_dict(torch.load(directory_name+'/weights.pt'))
weight_2e_4 = [p.data for p in model_2e_4.parameters()]

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
def cos_sim(d_1,d_2):
    vec1 = nplist_to_tensor(d_1)
    vec2 = nplist_to_tensor(d_2)
    cos_similarity = torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm()).item()
    return cos_similarity

print("cos similarity is ", cos_sim(ini_weight,weight_2e_4))

input_ = input_[is_active]
print(input_.size()[0])

jacob_2e_4 = []
input_.requires_grad = False
for param in model_2e_4.parameters():
    param.requires_grad = True
for i in range(input_.size()[0]):
    in_ = input_[i]*1
    output_ = model_2e_4(in_)
    der_list = torch.zeros((1)).to(device)
    for param in model_2e_4.parameters():
        der_ = torch.autograd.grad(output_[0], param, retain_graph = True)
        der_list = torch.concatenate([der_list,der_[0].flatten()])
    jacob_2e_4.append(der_list[1:])

jacob = torch.stack(jacob_2e_4)

output = model_2e_4(input_)
m = output.detach().cpu().flatten().numpy()
J_tot = np.ones((d_muon_counts_reference.shape[0], jacob.shape[1]))
print('J_tot shape is ', J_tot.shape)
for i in range(J_tot.shape[1]):
    J = muon_counts_simulator.Jvec(m, jacob[:,i])
    J_tot[:,i] = J*1

import pickle
f = open(directory_name+'/J_tot.pkl', 'wb')
pickle.dump(J_tot, f)
f.close()

"""A^TA nad AA^T have the same non-zero eigenvalues"""
W_J_tot = W@J_tot
hessian = W_J_tot@W_J_tot.T
print('hessian shape ', hessian.shape)

import scipy
tol = 0.
retvals = scipy.sparse.linalg.eigsh(hessian, k=2, which='LA', tol=tol)
eigvals, eigvecs = retvals

print(eigvals)

f = open(directory_name+'/eig_hessian.pkl', 'wb')
pickle.dump(eigvals, f)
pickle.dump(eigvecs, f)
f.close()

