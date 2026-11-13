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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

import pickle

torch.manual_seed(0)
np.random.seed(0)

class NFTOMO(nn.Module):
    def __init__(self):
         super(NFTOMO, self).__init__()
         self.b1 = nn.Sequential(nn.Linear(24,128), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b2 = nn.Sequential(nn.Linear(128,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b3 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b4 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b5 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
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

        x4 = self.b4(x3)

        x5 = self.b5(x4)

        x6 = self.b6(x5)

        _x7 = self.b7(x6)

        x7 = self.b8(_x7)

        return x7*2+2 #(0,4) bound
    
torch.manual_seed(0)
np.random.seed(0)
model = NFTOMO().to(device)

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

def input_mapping(x, B, L = 1):
  if B is None:
    return x
  else:
    cof_ = L/2
    x_proj = (cof_*torch.pi*x) @ B.T
  return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

num_L = 4
_x = input_mapping(input_.detach().cpu(), torch.eye(3, dtype=torch.float32), L = 1)
for i in range(num_L- 1):
  x_ = input_mapping(input_.detach().cpu(), torch.eye(3, dtype=torch.float32), L = i+2)
  _x = torch.concatenate([_x, x_], axis=-1)

input_pos = (_x*1).to(device)

epochs = 30000
lr_=1e-3
optimizer = optim.Adam(model.parameters(), lr=lr_)

# All the input data should be in subdirectories of this root data directory
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

start_time = time()
np.random.seed(0)
muon_counts_simulator = muon_simulation.MuonCountSimulation(
    mesh, sensors, exposure_time, n_rays_per_radiograph_pixel=10, model_map=active_map
)
end_time = time()
print('muon simulator is ready!' + ' Time needed is '+str(end_time-start_time))
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

def theta_m(output):
    theta_m = torch.abs(torch.sub(output.flatten(), torch.ones(output.flatten().size(dim=0)).to(device), alpha = 2.95))
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

def decay(x):
  return np.exp(-x/4e3)

def lr_decay(lr, x):
    decay_step = 30000
    return lr*((1+np.sin((np.pi*x/decay_step)+np.pi/2))/2)+1e-4

diff_list = []
loss_list = []
beta_list = []
reg_list = []
m_list = []
directory_name = 'Linear_PE_L_4'
import os
os.mkdir(directory_name)
loop_start = time()
input_pos = input_pos[is_active]
for epoch in range(epochs):
    beta = torch.tensor(decay(epoch)).to(device)
    output = model(input_pos)
    m = output.detach().cpu().flatten().numpy()
    m_full = np.zeros(mesh.n_cells)
    m_full[is_active] = m*1
    print(np.max(m), np.min(m))
    dif_=W*muon_counts_simulator.residual(m, simpeg_data_container.dobs)
    Jtvec = muon_counts_simulator.Jtvec(m, W.T*(dif_))
    diff = 0.5*np.vdot(dif_,dif_)
    loss = total_loss(output.flatten(), torch.from_numpy(Jtvec).type(torch.float32).to(device), beta)
    loss_list.append(loss.cpu())
    diff_list.append(diff)
    new_lr = lr_decay(lr_, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("The ",epoch," epoch diff is ", diff)
    print("The ",epoch," epoch total loss is ", loss)
    if epoch%2000 == 0:  
        m_list.append(output.cpu())
    if beta != 0.:
        reg = theta_m(output)
        beta_list.append(beta.cpu())
        reg_list.append(reg.cpu().detach().numpy())
        print("The ",epoch," epoch beta is ", beta)
        print("The ",epoch," epoch reg is ", reg)
loop_end = time()
print("Total time needed "+str(loop_end-loop_start))
pkl_name = directory_name+'/test_final'+'.pkl'
f = open(pkl_name, 'wb')
pickle.dump(m, f)
pickle.dump(diff_list,f)
pickle.dump(loss_list,f)
pickle.dump(beta_list,f)
pickle.dump(reg_list,f)
pickle.dump(m_list,f)
pickle.dump(loop_end-loop_start,f)
f.close()

torch.save(model.state_dict(), directory_name+'/weights.pt')
torch.save(model, directory_name+'/model.pt')
