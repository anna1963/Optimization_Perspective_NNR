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
# from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

class NFTOMO(nn.Module):
    def __init__(self):
         super(NFTOMO, self).__init__()
         self.b1 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b2 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b3 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b4 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b4_1 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b4_2 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
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

        x4 = self.b4_1(x4)

        x4 = self.b4_2(x4)

        x5 = self.b5(x4)

        x6 = self.b6(x5)

        _x7 = self.b7(x6)

        x7 = self.b8(_x7)

        x8 = self.b9(2*x7)

        return x8*2+2 #(0,4) bound

torch.manual_seed(0)
np.random.seed(0)
model = NFTOMO().to(device)

z = np.linspace(-1,1,33)
y = np.linspace(-1,1,108)
x = np.linspace(-1,1,108)
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

x= input_mapping(input_.detach().cpu(), B_gauss)
input_pos=torch.tensor(x, dtype=torch.float32).to(device)

#root_data_dir = Path("../inputs")

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

h = 20.0 # cell size in meters 
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
#f = open('../hete_block_cave_hat_1'+'.pkl', 'rb')
#test_model = pickle.load(f)
#f.close()

is_active = basic_cave_1_fwd_mesh > 0.0
active_map = simpeg.maps.InjectActiveCells(mesh, is_active, valInactive=0.0)

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
d_muon_counts_target_hat = muon_counts_simulator.get_data(test_model[is_active])

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

def theta_m(output):
    theta_m = torch.abs(torch.sub(output.flatten(), torch.ones(output.flatten().size(dim=0)), alpha = 2.7))
    return torch.sum(theta_m)

def total_loss(output, J, beta):
    """
    Note beta here differs from the beta in the convential EM inversion where beta typically starts with a value greater than 1 (e.x. 10), and then cooling down with a fixed rate.
    beta: [0,1]
    total_loss = (1-beta)*theta_phi + beta*theta_m
    """
    theta_phi =(1/(output.size(dim=0)))*torch.matmul(J,output.flatten())
    if beta != 0.:
        theta_m_ = (1/(output.size(dim=0)))*theta_m(output)
        return (1-beta)*theta_phi + beta*theta_m_
    else:
        return theta_phi

epochs = 1300
lr_ = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr_)

diff_list = []
loss_list = []
m_list = []
directory_name = 'Muon_hete_block_cave_hat_gpu_Gaussion_PE_2_10_layers_1e_3_seed_0_epoch_13e2_30_days'
import os
os.mkdir(directory_name)
loop_start = time()
time_sim = 0
input_pos= input_pos[is_active]
for epoch in range(epochs):
    beta = 0.0 
    output = model(input_pos)
    m = output.detach().cpu().flatten().numpy()
    start = time()
    dif_=W*muon_counts_simulator.residual(m, simpeg_data_container.dobs)
    Jtvec = muon_counts_simulator.Jtvec(m, W.T*(dif_))
    end = time()
    time_sim += end-start
    diff = 0.5*np.vdot(dif_,dif_)
    loss = total_loss(output.flatten(), torch.from_numpy(Jtvec).type(torch.float32).to(device), beta)
    loss_list.append(loss)
    diff_list.append(diff)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100 == 0:
        print(np.max(m), np.min(m))
        print("The ",epoch," epoch diff is ", diff)
        print("The ",epoch," epoch total loss is ", loss)
        m_list.append(m)
loop_end = time()
print("Total time needed "+str(loop_end-loop_start))
print('Total muon simulation cost '+str(time_sim))
pkl_name = directory_name+'/test_final'+'.pkl'
f = open(pkl_name, 'wb')
pickle.dump(m, f)
pickle.dump(diff_list,f)
pickle.dump(loss_list,f)
pickle.dump(m_list,f)
f.close()

torch.save(model.state_dict(), directory_name+'/weights.pt')
torch.save(model, directory_name+'/model.pt')

#Return the final result
model_= NFTOMO().to(device)
model_.load_state_dict(torch.load(directory_name+'/weights.pt'))
output = model_(input_pos)
m = output.detach().cpu().numpy().flatten()
pkl_name = directory_name+'/final'+'.pkl'
f = open(pkl_name, 'wb')
pickle.dump(m, f)
f.close()
