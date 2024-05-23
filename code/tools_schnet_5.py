import os
import numpy as np
import torch
from ase.db import connect
from ase.io.extxyz import read_xyz
from model_schnet import *
from const_defin import *


def schnet_best_model_parameters(fea_num):
    n_features = fea_num
    cutoff = 5.0
    n_gauss = 40
    n_atom_basis = n_features
    n_filters = n_features
    n_interactions = 3
    coupled_interactions = False
    num_embeddings = 100
    gauss_params_train = False
    out_hidden_neuron = int(n_features / 2)
    model_params = [n_atom_basis, n_filters, n_interactions, cutoff, n_gauss, coupled_interactions,
                    num_embeddings, gauss_params_train, out_hidden_neuron]
    return model_params


def load_model_schnet(model_params, mean_std_path, model_dict_path=None):
    [n_atom_basis, n_filters, n_interactions, cutoff, n_gauss, coupled_interactions,
     num_embeddings, gauss_params_train, out_hidden_neuron] = model_params
    mean, std = torch.load(mean_std_path)
    mean, std = mean["energy"], std["energy"]
    model = AtomisticModel(representation=SchNet(n_atom_basis, n_filters, n_interactions, cutoff,
                                                 n_gauss, coupled_interactions, num_embeddings, gauss_params_train),
                           output_modules=Atomwise(n_in=n_atom_basis, n_neurons=out_hidden_neuron,
                                                   property="energy",
                                                   mean=mean, stddev=std,
                                                   derivative="forces", negative_dr=True))
    model.load_state_dict(torch.load(model_dict_path, map_location='cpu'))
    return model


def load_schnet_5(fea_para_list, model_s0_path_list, model_s1_path_list, mean_std_ene_s0_path_list, mean_std_ene_s1_path_list):
    n_models = len(fea_para_list)
    model_s0_list = []
    model_s1_list = []
    for i in range(n_models):
        cur_fea = fea_para_list[i]
        cur_model_params = schnet_best_model_parameters(cur_fea)
        cur_model_s0_path = model_s0_path_list[i]
        cur_model_s1_path = model_s1_path_list[i]
        cur_mean_std_s0_path = mean_std_ene_s0_path_list[i]
        cur_mean_std_s1_path = mean_std_ene_s1_path_list[i]
        cur_model_s0 = load_model_schnet(cur_model_params, cur_mean_std_s0_path, cur_model_s0_path)
        model_s0_list.append(cur_model_s0)
        cur_model_s1 = load_model_schnet(cur_model_params, cur_mean_std_s1_path, cur_model_s1_path)
        model_s1_list.append(cur_model_s1)
    return model_s0_list, model_s1_list


def ext2db(path_ext, path_db):
    with connect(path_db, use_lock_file=False) as conn:
        with open(path_ext) as f:
            for at in read_xyz(f, index=slice(None)):
                value_ene = np.array(at.info['energy'])
                at.info['energy'] = value_ene
                data = {}
                if at.has("forces"):
                    data["forces"] = at.get_forces()
                data.update(at.info)
                conn.write(at, data=data)


def gen_db_ene_force(symbol_list, pos, path_ext, path_db):
    num_atom = pos.shape[0]
    ene = 0.0
    force = np.zeros([num_atom, 3])
    with open(path_ext, 'w') as file:
        file.write(str(num_atom) + "\n")
        file.write("Properties=species:S:1:pos:R:3:forces:R:3 energy=" + str(ene) + "\n")
        for i in range(num_atom):
            file.write("%2s %15.9f %15.9f %15.9f %15.9f %15.9f %15.9f\n" % (
                symbol_list[i], pos[i][0], pos[i][1], pos[i][2], force[i][0], force[i][1], force[i][2]))
    ext2db(path_ext, path_db)


def gen_and_load_db(work_dir, symbol_list, pos):
    pos = pos * au2ang
    os.system("rm -rf " + work_dir + "run_xyz2db")
    os.system("mkdir " + work_dir + "run_xyz2db")
    path_ext = work_dir + "run_xyz2db/temp.extxyz"
    path_db = work_dir + "run_xyz2db/temp.db"
    gen_db_ene_force(symbol_list, pos, path_ext, path_db)
    available_properties = ["energy"]
    data_atoms = AtomsData(dbpath=path_db, available_properties=available_properties)
    data_loader = AtomsLoader(data_atoms, 1)
    data_db = data_loader.__iter__().__next__()
    return data_db


def get_enediff_force(data_db, symbol_list, model_s0_list, model_s1_list):
    n_atoms = len(symbol_list)
    n_models = len(model_s0_list)
    ene_s0, ene_s1 = np.zeros(n_models), np.zeros(n_models)
    force_s0, force_s1 = np.zeros([n_models, n_atoms, 3]), np.zeros([n_models, n_atoms, 3])
    for i in range(n_models):
        cur_model_s0 = model_s0_list[i]
        cur_model_s1 = model_s1_list[i]
        result_s0 = cur_model_s0(data_db)
        ene_s0_pred = result_s0["energy"].detach().numpy()
        force_s0_pred = result_s0["forces"].detach().numpy()
        result_s1 = cur_model_s1(data_db)
        ene_s1_pred = result_s1["energy"].detach().numpy()
        force_s1_pred = result_s1["forces"].detach().numpy()

        ene_s0[i] = ene_s0_pred
        ene_s1[i] = ene_s1_pred
        force_s0[i] = force_s0_pred
        force_s1[i] = force_s1_pred
    return ene_s0, ene_s1, force_s0, force_s1


def judge_var(ene_s0, ene_s1, var_limit=100):
#     print(ene_s0, ene_s1)
    ene_s0_var = np.var(ene_s0)
    ene_s1_var = np.var(ene_s1)
    ene_var_mean = (ene_s0_var + ene_s1_var) / 2
    if ene_var_mean < var_limit:
        return True, ene_var_mean
    else:
        return False, ene_var_mean


def gen_add_enediff_force_5(ene, force, data_db, symbol_list, model_s0_list, model_s1_list, 
                            mean_std_ene_s0_path_list, mean_std_ene_s1_path_list, var_limit=100):
    ene_s0, ene_s1, force_s0, force_s1 = get_enediff_force(data_db, symbol_list, model_s0_list, model_s1_list)
    n_atoms = force_s0[0].shape[0]
    if_var, ene_var_mean = judge_var(ene_s0, ene_s1, var_limit)
    print("@energy variance@ %10.1f" % ene_var_mean)
    ene_s0_pred = np.zeros(1)[0]
    ene_s1_pred = np.zeros(1)[0]
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

