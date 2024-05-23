import torch
from ase.atoms import Atoms
import numpy as np
from const_defin import *
from model_bpnn import BPNN


def load_model_bpnn_5(fea_in, hidden_num_list, model_dict_path_list):
    model_list = []
    for i in range(len(hidden_num_list)):
        cur_hidden_num = hidden_num_list[i]
        cur_model = BPNN(fea_in, cur_hidden_num)
        cur_model_dict_path = model_dict_path_list[i]
        cur_model.load_state_dict(torch.load(cur_model_dict_path, map_location='cpu'))
        model_list.append(cur_model)
    return model_list


def gen_des_der_by_soap(pos, symbol_list, soap):
    pos = pos * au2ang
    traj = [Atoms(symbols=symbol_list, positions=pos)]
    num_atom = pos.shape[0]
    der, des = soap.derivatives(traj, centers=[[*range(num_atom)]], method="analytical")
    return des, der


def stand_scale(tensor, mean, std):
    tensor_scale = (tensor - mean) / std
    return tensor_scale


def inverse_scale(tensor, mean, std):
    tensor_inverse = (tensor * std) + mean
    return tensor_inverse


def des_der_process(mean_std_path, des, der):
    des_atom_mean_std_list = torch.load(mean_std_path)
    atom_num = len(des_atom_mean_std_list)
    des_list, der_list = [], []
    for i in range(atom_num):
        cur_des_atom = torch.DoubleTensor(des[i])
        cur_mean, cur_std = des_atom_mean_std_list[i]
        cur_des_atom = stand_scale(cur_des_atom, cur_mean, cur_std)
        cur_des_atom.requires_grad = True
        des_list.append(cur_des_atom)

        cur_der_atom = torch.DoubleTensor(der[i])
        cur_der_atom = cur_der_atom / cur_std
        der_list.append(cur_der_atom)
    return des_list, der_list


def ene_force_process(mean_std_path, ene, force):
    mean, std = torch.load(mean_std_path)
    mean, std = mean.numpy(), std.numpy()
    ene_inverse = inverse_scale(ene, mean, std)
    force_inverse = force * std
#     ene_inverse = ene_inverse / au2kcal
#     force_inverse = force_inverse * au2ang / au2kcal
    return ene_inverse, force_inverse


def get_enediff_force(geom, symbol_list, soap, model_s0_list, model_s1_list, 
                      mean_std_des_path_list, mean_std_ene_s0_path_list, mean_std_ene_s1_path_list):
    n_atoms = geom.shape[0]
    n_models = len(model_s0_list)
    des, der = gen_des_der_by_soap(geom, symbol_list, soap)
    ene_s0, ene_s1 = np.zeros(n_models), np.zeros(n_models)
    force_s0, force_s1 = np.zeros([n_models, n_atoms, 3]), np.zeros([n_models, n_atoms, 3])
    for i in range(n_models):
        cur_mean_std_des_path = mean_std_des_path_list[i]
        des_list, der_list = des_der_process(cur_mean_std_des_path, des, der)
        
        cur_model_s0 = model_s0_list[i]
        cur_model_s1 = model_s1_list[i]
        [ene_s0_pred, force_s0_pred] = cur_model_s0(des_list, der_list)
        [ene_s0_pred, force_s0_pred] = [ene_s0_pred.detach().numpy(), force_s0_pred.detach().numpy()]
        [ene_s1_pred, force_s1_pred] = cur_model_s1(des_list, der_list)
        [ene_s1_pred, force_s1_pred] = [ene_s1_pred.detach().numpy(), force_s1_pred.detach().numpy()]
        
        cur_mean_std_ene_s0_path = mean_std_ene_s0_path_list[i]
        cur_mean_std_ene_s1_path = mean_std_ene_s1_path_list[i]
        ene_s0_pred, force_s0_pred = ene_force_process(cur_mean_std_ene_s0_path, ene_s0_pred, force_s0_pred)
        ene_s1_pred, force_s1_pred = ene_force_process(cur_mean_std_ene_s1_path, ene_s1_pred, force_s1_pred)
        
        ene_s0[i] = ene_s0_pred
        ene_s1[i] = ene_s1_pred
        force_s0[i] = force_s0_pred
        force_s1[i] = force_s1_pred
    return ene_s0, ene_s1, force_s0, force_s1


def judge_var(ene_s0, ene_s1, var_limit=100):
    # print(ene_s0, ene_s1)
    ene_s0_var = np.var(ene_s0)
    ene_s1_var = np.var(ene_s1)
    ene_var_mean = (ene_s0_var + ene_s1_var) / 2
    if ene_var_mean < var_limit:
        return True, ene_var_mean
    else:
        return False, ene_var_mean


def gen_add_enediff_force_5(ene, force, geom, symbol_list, soap, model_s0_list, model_s1_list, 
                               mean_std_des_path_list, mean_std_ene_s0_path_list, mean_std_ene_s1_path_list, var_limit=100):
    ene_s0, ene_s1, force_s0, force_s1 = get_enediff_force(geom, symbol_list, soap, model_s0_list, model_s1_list, 
        mean_std_des_path_list, mean_std_ene_s0_path_list, mean_std_ene_s1_path_list)
    n_atoms = force_s0[0].shape[0]
    if_var, ene_var_mean = judge_var(ene_s0, ene_s1, var_limit)
    print("@energy variance@ %10.1f" % ene_var_mean)
    ene_s0_pred = np.zeros(1)
    ene_s1_pred = np.zeros(1)
    force_s0_pred = np.zeros([n_atoms, 3])
    force_s1_pred = np.zeros([n_atoms, 3])
    
    if if_var:
        ene_s0_pred = np.mean(ene_s0) / au2kcal
        ene_s1_pred = np.mean(ene_s1) / au2kcal
        force_s0_pred = np.mean(force_s0, axis=0) * au2ang / au2kcal
        force_s1_pred = np.mean(force_s1, axis=0) * au2ang / au2kcal
    else:
        with open('geom_error.xyz', 'a') as f_xyz:
            print(" ", n_atoms, file=f_xyz)
            print(" ene_var_mean: ", ene_var_mean, file=f_xyz)
            for i in range(n_atoms):
                print("%2s %15.8f %15.8f %15.8f" % (symbol_list[i], geom[i][0], geom[i][1], geom[i][2]), file=f_xyz)
    
    ene_s0_pred += cas_sub_om2_s0_refer
    ene_s1_pred += cas_sub_om2_s1_refer
    ene_pred = np.array([ene_s0_pred, ene_s1_pred])
    force_pred = np.array([force_s0_pred, force_s1_pred])
    ene += ene_pred
    force += force_pred
    return ene, force

