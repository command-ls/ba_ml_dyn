import os
import numpy as np
from copy import deepcopy as cp
from dscribe.descriptors import SOAP
from const_defin import *
from tools_bpnn_5 import *


n_states = 2
start_state = 2
time_ps = 0.5
step_nuc = 1000
step_elec = 200
work_dir = os.popen('pwd').readlines()[0].split('\n')[0] + '/'

# The complex128 in numpy represents float64 for both real and imaginary parts
den = np.zeros([n_states, n_states], dtype=np.complex128)
den[start_state-1, start_state-1] = np.array([1.0 + 0.0j])
cur_state = cp(start_state)
time_step = time_ps * ps2au / step_nuc
dyn_xyz_path = work_dir + 'dyn.xyz'
dyn_state_path = work_dir + 'dyn.state'
dyn_ene_path = work_dir + 'dyn.ene'
geom_init_path = work_dir + 'geom'
# vel_init_path = work_dir + 'veloc'
momenta_init_path = work_dir + 'momenta'

# soap_atom_fea_in = 63. load bpnn model
fea_in = 63
hidden_num_list = [96, 88, 112, 120, 116]
model_dir = "/home/lish/data_ls/ml_dyn/model_best/"
model_s0_10_path = model_dir + "bpnn_soap_hid96_s10_enediff_force_long_s0.pth"
model_s0_100_path = model_dir + "bpnn_soap_hid88_s100_enediff_force_long_s0.pth"
model_s0_1000_path = model_dir + "bpnn_soap_hid112_s1000_enediff_force_long_s0.pth"
model_s0_10000_path = model_dir + "bpnn_soap_hid120_s10000_enediff_force_long_s0.pth"
model_s0_100000_path = model_dir + "bpnn_soap_hid116_s100000_enediff_force_long_s0.pth"
model_s0_path_list = [model_s0_10_path, model_s0_100_path, model_s0_1000_path, model_s0_10000_path, model_s0_100000_path]
model_s1_10_path = model_dir + "bpnn_soap_hid96_s10_enediff_force_long_s1.pth"
model_s1_100_path = model_dir + "bpnn_soap_hid88_s100_enediff_force_long_s1.pth"
model_s1_1000_path = model_dir + "bpnn_soap_hid112_s1000_enediff_force_long_s1.pth"
model_s1_10000_path = model_dir + "bpnn_soap_hid120_s10000_enediff_force_long_s1.pth"
model_s1_100000_path = model_dir + "bpnn_soap_hid116_s100000_enediff_force_long_s1.pth"
model_s1_path_list = [model_s1_10_path, model_s1_100_path, model_s1_1000_path, model_s1_10000_path, model_s1_100000_path]
model_s0_list = load_model_bpnn_5(fea_in, hidden_num_list, model_s0_path_list)
model_s1_list = load_model_bpnn_5(fea_in, hidden_num_list, model_s1_path_list)

# mean_std load
mean_std_dir = "/home/lish/data_ls/ml_dyn/mean_std/"
msef_s0_10_path = mean_std_dir + "scale_list_ene_s10_enediff_force_s0.pt"
msef_s0_100_path = mean_std_dir + "scale_list_ene_s100_enediff_force_s0.pt"
msef_s0_1000_path = mean_std_dir + "scale_list_ene_s1000_enediff_force_s0.pt"
msef_s0_10000_path = mean_std_dir + "scale_list_ene_s10000_enediff_force_s0.pt"
msef_s0_100000_path = mean_std_dir + "scale_list_ene_s100000_enediff_force_s0.pt"
mean_std_ene_s0_path_list = [msef_s0_10_path, msef_s0_100_path, msef_s0_1000_path, msef_s0_10000_path, msef_s0_100000_path]
msef_s1_10_path = mean_std_dir + "scale_list_ene_s10_enediff_force_s1.pt"
msef_s1_100_path = mean_std_dir + "scale_list_ene_s100_enediff_force_s1.pt"
msef_s1_1000_path = mean_std_dir + "scale_list_ene_s1000_enediff_force_s1.pt"
msef_s1_10000_path = mean_std_dir + "scale_list_ene_s10000_enediff_force_s1.pt"
msef_s1_100000_path = mean_std_dir + "scale_list_ene_s100000_enediff_force_s1.pt"
mean_std_ene_s1_path_list = [msef_s1_10_path, msef_s1_100_path, msef_s1_1000_path, msef_s1_10000_path, msef_s1_100000_path]
msdes_10_path = mean_std_dir + "scale_list_des_atom_list_s10_enediff_force.pt"
msdes_100_path = mean_std_dir + "scale_list_des_atom_list_s100_enediff_force.pt"
msdes_1000_path = mean_std_dir + "scale_list_des_atom_list_s1000_enediff_force.pt"
msdes_10000_path = mean_std_dir + "scale_list_des_atom_list_s10000_enediff_force.pt"
msdes_100000_path = mean_std_dir + "scale_list_des_atom_list_s100000_enediff_force.pt"
mean_std_des_path_list = [msdes_10_path, msdes_100_path, msdes_1000_path, msdes_10000_path, msdes_100000_path]

# Configuration Descriptor, soap: des(28, 63) der(28, 28, 3, 63)
soap = SOAP(species=["H", "C", "N"], periodic=False, r_cut=6.0, sigma=0.5, n_max=2, l_max=2)
symbol_list = ['N', 'N', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 
               'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H' ]
var_limit = 100

