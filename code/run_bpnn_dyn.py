#!/home/lish/programs/anaconda3/bin/python
import numpy as np
# 导入函数及常量
import sys
code_path = '/home/lish/data_ls/ml_dyn/code/'
if code_path not in sys.path:
    sys.path.append(code_path)
# print(sys.path)
from dyn_fssh_all import *
from params_bpnn import *
# 设置运行的起始时间
import time
time_now = int(time.time())
t1 = time.perf_counter()
# 设置随机数及打印偏好
np.random.seed(10)
np.set_printoptions(linewidth=150)


# Read initial geometry and velocity
element, atom_num, n_atoms, mass, geom, vel = read_init_geom_momenta(symbol_list, geom_init_path, momenta_init_path)
# Generate energy, force and nonadiabatic coupling for the initial structure
ene, force, nac = calc_ene_by_mndo(work_dir, n_atoms, atom_num, geom, n_states)
# Generate energy, force by ml. And additional ml energy and force to om2
ene, force = gen_add_enediff_force_5(ene, force, geom, element, soap, model_s0_list, model_s1_list, 
                mean_std_des_path_list, mean_std_ene_s0_path_list, mean_std_ene_s1_path_list, var_limit)
# Output initial state information
output_info(element, n_atoms, n_states, mass, geom, vel, den, ene, cur_state)
# Record initial structure, state, and energy information
os.system("rm -f " + dyn_xyz_path + " " + dyn_state_path + " " + dyn_ene_path + " ")
write_xyz_state_ene(dyn_xyz_path, dyn_state_path, dyn_ene_path, n_atoms, element, geom, cur_state, ene)


geom_n, vel_n, den_n, ene_n, force_n, nac_n = geom, vel, den, ene, force, nac
# Start dynamic cycle
for loop in range(step_nuc):
# for loop in range(5):
    loop += 1
    print("._______________________________________________________________.")
    print("|                                                               |")
    print("%28s %6d" % ("this is MFSH MD. run step:", loop))
    # Propagate geometry to (t + dt) using Velocity-Verlet Algorithm
    geom_n = prop_nuc(geom, vel, time_step, force, mass, cur_state)
    if np.abs(geom_n).max() > 20:
        print("Please check! geom(max) is already greater than 20 a.u.!!!")
        break
    # Calculate energy, force and nac at (t + dt)
    ene_n, force_n, nac_n = calc_ene_by_mndo(work_dir, n_atoms, atom_num, geom_n, n_states)
    # simplest phase correction
    nac_n = simple_phase_correction(n_states, nac, nac_n)
    # Generate energy, force by ml. And additional ml energy and force to om2
    ene_n, force_n = gen_add_enediff_force_5(ene_n, force_n, geom_n, element, soap, model_s0_list, model_s1_list, 
                        mean_std_des_path_list, mean_std_ene_s0_path_list, mean_std_ene_s1_path_list, var_limit)
    # Propagate velocity using Velocity-Verlet Algorithm
    vel_n = prop_vel(vel, time_step, force, force_n, mass, cur_state)
    # Propagate electronic states(calc hopping prob) using RK4
    prob, den_n = prop_elec(time_step, step_elec, mass, ene, ene_n, vel, vel_n, nac, nac_n, den, n_states, cur_state)
    # Hopping justification
    hop, tar_state = hop_judge(prob, n_states, cur_state, ene_n, vel_n, mass)
    # Velocity rescaling
    if hop:
        vel_n, cur_state = revel_nac(mass, nac_n, vel_n, ene_n, cur_state, tar_state, hop, force_n)
    # Output status for this run
    print("After this run:")
    output_info(element, n_atoms, n_states, mass, geom_n, vel_n, den_n, ene_n, cur_state)
#     print(np.hstack((force_n[0], force_n[1])))
    # Write outfile
    write_xyz_state_ene(dyn_xyz_path, dyn_state_path, dyn_ene_path, n_atoms, element, geom_n, cur_state, ene_n)
    # Exchange important parameters for next step
    geom, vel, den, ene, force, nac = geom_n, vel_n, den_n, ene_n, force_n, nac_n
    print("|_______________________________________________________________|")
t2 = time.perf_counter()
print('All done! Function runtime: %10.2f min' % ((t2 - t1) / 60.0))

