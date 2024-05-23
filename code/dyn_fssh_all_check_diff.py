import os
from copy import deepcopy as cp
import numpy as np
from const_defin import amu2au, au2ang, au2kcal, ev2au, au2k
mndo_path = "/home/lish/bin/mndo99"


def read_init_geom_vel(geom_init_path, vel_init_path):
    geom_origin = np.loadtxt(geom_init_path, dtype=np.str_)
    element = geom_origin[:, 0]
    atom_num = geom_origin[:, 1].astype(np.float64).astype(np.int64)
    n_atoms = atom_num.shape[0]
    x = geom_origin[:, 2].reshape([-1, 1]).astype(np.float64)
    y = geom_origin[:, 3].reshape([-1, 1]).astype(np.float64)
    z = geom_origin[:, 4].reshape([-1, 1]).astype(np.float64)
    mass = geom_origin[:, -1].astype(np.float64) * amu2au
    geom = np.concatenate([x, y, z], axis=1)
    vel = np.loadtxt(vel_init_path)
    return element, atom_num, n_atoms, mass, geom, vel


def read_init_geom_momenta(symbol_list, geom_init_path, momenta_init_path):
    n_atoms = len(symbol_list)
    element = symbol_list
    atom_num = np.zeros(n_atoms).astype(np.int64)
    for i in range(n_atoms):
        cur_atom = symbol_list[i]
        if cur_atom == 'N':
            atom_num[i] = 7
        elif cur_atom == 'C':
            atom_num[i] = 6
        elif cur_atom == 'H':
            atom_num[i] = 1
        else:
            print("please check!")
    mass = np.zeros(n_atoms)
    for i in range(n_atoms):
        cur_atom = symbol_list[i]
        if cur_atom == 'N':
            mass[i] = 14.00307401
        elif cur_atom == 'C':
            mass[i] = 12.00000000
        elif cur_atom == 'H':
            mass[i] = 1.00782504
        else:
            print("please check!")
    mass *= amu2au
    
    geom = np.loadtxt(geom_init_path)
    momenta = np.loadtxt(momenta_init_path)
    vel = momenta / mass.reshape(-1, 1)
    return element, atom_num, n_atoms, mass, geom, vel


def gen_mndo_in_file(work_dir, n_atoms, atom_num, geom):
    mndo_head = ["IOP=-6 JOP=-2 IGEOM=1 IFORM=1 NSAV15=9 ICUTS=-1 ICUTG=-1 +",
                 "ISCF=9 IPLSCF=9 IPRINT=1 MPRINT=1 +",
                 "DSTEP=0.00001 +",
                 "KHARGE=0 IMULT=1 KITSCF=9999 KCI=5 +",
                 "ICI1=4 ICI2=0 IOUTCI=1 MOVO=0 NCIREF=3 MCIREF=0 +",
                 "LEVEXC=2 IROOT=VALUE IUVCD=2 IMOMAP=0 NCIGRD=2 ICROSS=2",
                 "",
                 "CI OM2/MRCI"]
    mndo_tail = [" 0    0.0000000000 0    0.0000000000 0    0.0000000000 0",
                 "",
                 "1 2"]

    os.system("rm -rf " + work_dir + "run_mndo")
    os.system("mkdir " + work_dir + "run_mndo")

    geom = geom * au2ang
    with open(work_dir + 'run_mndo/mndo.in', 'w') as file:
        for i in range(len(mndo_head)):
            print(mndo_head[i], file=file)

        for i in range(n_atoms):
            print("%2d %15.10f %1d %15.10f %1d %15.10f %1d" % (
                atom_num[i], geom[i][0], 1, geom[i][1], 1, geom[i][2], 1), file=file)

        for i in range(len(mndo_tail)):
            print(mndo_tail[i], file=file)


def load_ene_force_nac(work_dir, n_atoms, n_states):
    ene = np.zeros([n_states])
    grad = np.zeros([n_states, n_atoms, 3])
    vc = np.zeros([n_states, n_states, n_atoms, 3])
    with open(work_dir + "run_mndo/mndo.out") as f:
        lines = f.readlines()
        while True:
            if lines == []:
                break
            line = lines.pop(0)
            #         print(line, end='')
            if ",  Mult." in line:
                cur_state = int(line.split()[1].split(',')[0])
                cur_ene = float(line.split()[8])
                ene[cur_state - 1] = cur_ene

    with open(work_dir + "run_mndo/fort.15") as f:
        lines = f.readlines()
        while True:
            if lines == []:
                break
            line = lines.pop(0)
            #         print(line, end='')
            if "CARTESIAN GRADIENT FOR STATE" in line:
                cur_state = int(line.split()[-1])
                for i in range(n_atoms):
                    xyz = lines.pop(0).split()
                    temp1, temp2, x, y, z = xyz
                    grad[cur_state - 1][i] = [float(x), float(y), float(z)]
            if "CARTESIAN INTERSTATE COUPLING GRADIENT FOR STATES" in line:
                for i in range(n_atoms):
                    xyz = lines.pop(0).split()
                    temp1, temp2, x, y, z = xyz
                    vc[0][1][i] = [float(x), float(y),
                                   float(z)]  # Vibration coupling between the first and second states
                    vc[1][0][i] = cp(vc[0][1][i])  # Vibration coupling between the second and first states
    ene = ene * ev2au
    grad = grad * au2ang / au2kcal
    force = -grad
    vc = vc * au2ang / au2kcal
    nac = vc
    # Nonadiabatic coupling is calculated from vibration coupling and energy difference
    ene_diff = abs(ene[0] - ene[1])
    if not ene_diff > 1E-10:
        print("Please check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Current ene_diff is: ", ene_diff, "; cur_ene is:", ene)
        nac[0][1] = nac[0][1] / (ene[0] - ene[1] + 1E-7)
        nac[1][0] = nac[1][0] / (ene[1] - ene[0] + 1E-7)
        nac[0][0], nac[1][1] = 0.0, 0.0
        return ene, force, nac
    nac[0][1] = nac[0][1] / (ene[0] - ene[1])
    nac[1][0] = nac[1][0] / (ene[1] - ene[0])
    nac[0][0], nac[1][1] = 0.0, 0.0
    return ene, force, nac


def calc_ene_by_mndo(work_dir, n_atoms, atom_num, geom, n_states):
    gen_mndo_in_file(work_dir, n_atoms, atom_num, geom)
    dir_mndo_run = work_dir + "run_mndo/"
    cmd_line = "cd " + dir_mndo_run + "; " + mndo_path + " < mndo.in > mndo.out"
    os.system(cmd_line)
    ene, force, nac = load_ene_force_nac(work_dir, n_atoms, n_states)
    return ene, force, nac


def calc_kine(vel, mass):
    mass = mass.reshape(-1, 1)
    kine = 0.5 * np.sum(np.multiply(mass, np.square(vel)))
    return kine


def calc_ene_total(kine, ene, cur_state):
    index_state = cur_state - 1
    ene_total = kine + ene[index_state]
    return ene_total


def calc_tem(kine, n_atoms):
    tem = 2 * kine / (3 * n_atoms - 6)
    return tem


def calc_mom(vel, mass):
    mass = mass.reshape(-1, 1)
    mom = np.multiply(mass, vel)
    return mom


def dot_nacv_vel(nacv, vel):
    dot_value = np.sum(np.multiply(nacv, vel))
    return dot_value


def output_info(element, n_atoms, n_states, mass, geom, vel, den, ene, cur_state):
    kine = calc_kine(vel, mass)
    tem = calc_tem(kine, n_atoms)
    ene_total = calc_ene_total(kine, ene, cur_state)

    index_state = cur_state - 1
    print("@molecular coordinate@")
    for i in range(n_atoms):
        print("%2s %15.8f %15.8f %15.8f" % (element[i], geom[i][0], geom[i][1], geom[i][2]))

    print("@molecular velocity@")
    for i in range(n_atoms):
        print("%2s %20.10f %20.10f %20.10f" % (element[i], vel[i][0], vel[i][1], vel[i][2]))

    print("%21s %18.8f %1s" % ("@current temperature@", tem * au2k, "K"))

    print("@current density matrix@")
    for i in range(n_states):
        for j in range(n_states):
            print("%6s%2d%1s%2d%1s %15.8f %15.8f" % (">den(", i + 1, ",", j + 1, ")", den[i, j].real, den[i, j].imag))

    print("%15s %2d" % ("@current state@", cur_state))

    print("@molecular energy@")
    print("%28s %18.10f" % ("kinetic      energy (a.u.) ", kine))
    print("%28s %18.10f" % ("potential    energy (a.u.) ", ene[index_state]))
    print("%28s %18.10f" % ("total        energy (a.u.) ", ene_total))
    print("@adiabatic energy@")
    for i in range(n_states):
        print("%28s %18.10f" % ("potential    energy (a.u.) ", ene[i]))


def write_xyz_state_ene(dyn_xyz_path, dyn_state_path, dyn_ene_path, n_atoms, element, geom, cur_state, ene):
    geom = geom * au2ang
    with open(dyn_xyz_path, 'a') as f_xyz:
        print(" ", n_atoms, file=f_xyz)
        print(" cur_state: ", cur_state, file=f_xyz)
        for i in range(n_atoms):
            print("%2s %15.8f %15.8f %15.8f" % (element[i], geom[i][0], geom[i][1], geom[i][2]), file=f_xyz)

    with open(dyn_state_path, 'a') as f_state:
        print(" ", cur_state, file=f_state)

    with open(dyn_ene_path, 'a') as f_ene:
        print("%18.10f %18.10f" % (ene[0], ene[1]), file=f_ene)


def prop_nuc(geom, vel, time_step, force, mass, cur_state):
    force_cur = force[cur_state - 1]
    mass = mass.reshape([-1, 1])
    acc = np.divide(force_cur, mass)
    geom_n = geom + vel * time_step + 0.5 * acc * time_step * time_step
    return geom_n


def prop_vel(vel, time_step, force, force_n, mass, cur_state):
    force_cur = force[cur_state - 1]
    force_n_cur = force_n[cur_state - 1]
    mass = mass.reshape([-1, 1])
    acc = np.divide(force_cur, mass)
    acc_n = np.divide(force_n_cur, mass)
    vel_n = vel + 0.5 * (acc + acc_n) * time_step
    return vel_n


def get_tar_state(prob, n_states):
    random_num = np.random.random()
    tar_state = 0
    prob_cum = 0
    for i in range(n_states):
        i += 1
        tar_state = i
        prob_cum += prob[i - 1]
        if random_num < prob_cum:
            break
    # output probability
    print("%16s %20.5f" % (" @random number: ", random_num))
    print(" @probabilities:")
    for i in range(n_states):
        i += 1
        if i == tar_state:
            print("%2d %20.5f %2s" % (i, prob[i - 1], "<<<"))
        else:
            print("%2d %20.5f" % (i, prob[i - 1]))
    return tar_state


def hop_judge(prob, n_states, cur_state, ene_n, vel_n, mass):
    tar_state = get_tar_state(prob, n_states)
    ene_sub_kcal = (ene_n[cur_state - 1] - ene_n[tar_state - 1]) * au2kcal
    ene_diff_kcal = abs(ene_sub_kcal)
    kine = calc_kine(vel_n, mass)
    hop = False
    # Check the dice
    if tar_state != cur_state:
        hop = True
        # output energy gap
        print("%23s %10.5f" % ("@energy gap (kcal/mol)", ene_sub_kcal))
    else:
        print("not enough probability --- hopping rejected.")
        hop = False
    # first, the energy gap between current & target must .le. 10 kcal/mol
    if hop:
        if ene_diff_kcal < 10.0:
            hop = True
        else:
            print("too large energy gap --- hopping rejected.")
            hop = False
    # second, there must be enough kinetic energy
    if hop:
        cur_ene = ene_n[cur_state - 1] + kine
        tar_ene = ene_n[tar_state - 1]
        if cur_ene > tar_ene:
            hop = True
        else:
            print("not enough total energy --- hopping rejected.")
            hop = False
    return hop, tar_state


def calc_abcd(mass, nac_tar_cur, vel_n, ene_n, cur_state, tar_state, hop):
    kine_n = calc_kine(vel_n, mass)
    mass = mass.reshape([-1, 1])
    a = 0.5 * np.sum(np.multiply(mass, np.square(nac_tar_cur)))
    b = -1.0 * np.sum(np.multiply(mass, np.multiply(nac_tar_cur, vel_n)))
    c = 0.5 * np.sum(np.multiply(mass, np.square(vel_n)))
    c = c + ene_n[tar_state - 1] - ene_n[cur_state - 1] - kine_n
    d = b * b - 4.0 * a * c
    if d < 0:
        print("not enough dimension energy --- hopping rejected.")
        hop = False
    return a, b, c, d, hop, kine_n


def revel_nac(mass, nac_n, vel_n, ene_n, cur_state, tar_state, hop, force_n):
    nac_tar_cur = nac_n[tar_state - 1][cur_state - 1]
    force_cur = force_n[cur_state - 1]
    force_tar = force_n[tar_state - 1]
    a, b, c, d, hop, kine_n = calc_abcd(mass, nac_tar_cur, vel_n, ene_n, cur_state, tar_state, hop)
    if hop:
        factor = 0.0
        if b > 0:
            factor = (-b + np.sqrt(d)) / (2.0 * a)
        else:
            factor = (-b - np.sqrt(d)) / (2.0 * a)
        vel_n_n = vel_n - factor * nac_tar_cur
        kine_n_n = calc_kine(vel_n_n, mass)
        print("velocity adjusted along NAC vector(hop):")
        print('  %12.4f (V_1) + %12.4f (K_1)  ==  %12.4f (V_2) + %12.4f (K_2)' %
              (ene_n[cur_state - 1] * au2kcal, kine_n * au2kcal, ene_n[tar_state - 1] * au2kcal, kine_n_n * au2kcal))
        print("--- hopping identified.")
        print("%9s %1d %4s %1d" % (" @hopping", cur_state, " -> ", tar_state))
        ene_diff_kcal = (ene_n[cur_state - 1] - ene_n[tar_state - 1]) * au2kcal
        print("%29s %20.5f" % (" @potential change (kcal/mol)", ene_diff_kcal))
        cur_state = tar_state
        return vel_n_n, cur_state
    elif not hop:
        mom = calc_mom(vel_n, mass)
        t1 = dot_nacv_vel(force_cur, nac_tar_cur)
        t2 = dot_nacv_vel(force_tar, nac_tar_cur)
        t3 = dot_nacv_vel(mom, nac_tar_cur)
        # The hopping failed but met the conditions to reverse the velocity and try the hopping again
        if t1 * t2 < 0 and t2 * t3 < 0:
            factor = -b / a
            vel_n_n = vel_n - factor * nac_tar_cur
            kine_n_n = calc_kine(vel_n_n, mass)
            print("velocity adjusted along NAC vector(not hop):")
            print("  %12.4f (V_1) + %12.4f (K_1)  ==  %12.4f (V_2) + %12.4f (K_2)" %
                  (
                  ene_n[cur_state - 1] * au2kcal, kine_n * au2kcal, ene_n[tar_state - 1] * au2kcal, kine_n_n * au2kcal))
            return vel_n_n, cur_state
        # If the conditions are not met, maintain the original velocity
        else:
            print("adjusting conditions not met.")
            return vel_n, cur_state


def grad_elec(n_states, nac, den, vel, ene):
    elec_grad = np.zeros([n_states, n_states], dtype=np.complex128)
    num = np.zeros(1, dtype=np.complex128) + 1j
    for j in range(n_states):
        for k in range(n_states):
            elec_grad[j, k] = -1 * den[j, k] * num * (ene[j] - ene[k])
            for l in range(n_states):
                nacv_jl = dot_nacv_vel(nac[j, l, :, :], vel)
                nacv_lk = dot_nacv_vel(nac[l, k, :, :], vel)
                elec_grad[j, k] = elec_grad[j, k] - (den[l, k] * nacv_jl - den[j, l] * nacv_lk)
    return elec_grad


def simple_phase_correction(n_states, nac, nac_n):
    for i in range(n_states):
        for j in range(n_states):
            dot_nacv = np.sum(np.multiply(nac[i, j, :, :], nac_n[i, j, :, :]))
            if dot_nacv < 0:
                nac_n[i, j, :, :] *= -1
    return nac_n
    

def decoherence(n_states, vel, mass, ene, cur_index, den, sub_time):
    # Granucci's Energy Based Decoherence
    # first part, calculate decoherence rate
    rate = np.zeros(n_states)
    ekin = calc_kine(vel, mass)
    for i in range(n_states):
        # deco_para is set as 0.1
        rate[i] = np.abs(ene[cur_index] - ene[i]) / (0.1 / ekin + 1)
    # second part, do decoherence
    if rate[cur_index] != 0:
        print("DEBUG ERROR -- Decoherence rate of current state in not zero.")
    # Decoherence for non-active density element
    for j in range(n_states):
        for k in range(n_states):
            den[j, k] = den[j, k] * np.exp(-(rate[j] + rate[k]) * sub_time)
    # Calculate New Population
    den_new = den[cur_index, cur_index] + 1
    for j in range(n_states):
        den_new -= den[j, j]
    ratio = np.sqrt(np.real(den_new)) / np.sqrt(np.real(den[cur_index, cur_index]))
    den[cur_index, cur_index] = den[cur_index, cur_index] * ratio * ratio
    # Decoherence for coupled-active density element
    for j in range(n_states):
        if j != cur_index:
            den[j, cur_index] *= ratio
            den[cur_index, j] *= ratio
    return den


def correct_prob(n_states, prob, cur_index):
    # first, turn negative to 0, overflow to 1 (this rarely happens)
    for i in range(n_states):
        if prob[i] < 0:
            prob[i] = 0
        if prob[i] > 1:
            prob[i] = 1
    # second, normalize the probability
    norm = np.sum(prob)
    # assign probability of current state
    if norm < 1:
        prob[cur_index] = 1 - norm
    else:
        for i in range(n_states):
            prob[i] /= norm
    return prob


def prop_elec(time_step, step_elec, mass, ene, ene_n, vel, vel_n, nac, nac_n, den, n_states, cur_state):
    cur_index = cur_state - 1
    sub_time = time_step / step_elec
    prob = np.zeros(n_states)
    # propagate electronic coefficients using RK4
    for n_step in range(step_elec):
        n_step += 1
        # interpolate all variables at this time
        ene_now = ene + (n_step - 1) * (ene_n - ene) / step_elec
        vel_now = vel + (n_step - 1) * (vel_n - vel) / step_elec
        nac_now = nac + (n_step - 1) * (nac_n - nac) / step_elec
        vel_last = vel_now
        nac_last = nac_now
        den_last = den
        # then get k1
        egrad1 = grad_elec(n_states, nac_now, den, vel_now, ene_now)
        # interpolate all variables at time t+1/2dt
        vel_now = vel + (n_step - 0.5) * (vel_n - vel) / step_elec
        ene_now = ene + (n_step - 0.5) * (ene_n - ene) / step_elec
        nac_now = nac + (n_step - 0.5) * (nac_n - nac) / step_elec
        # get k2
        den2 = den + egrad1 * sub_time / 2
        egrad2 = grad_elec(n_states, nac_now, den2, vel_now, ene_now)
        # interpolate all variables at time t+1/2dt
        vel_now = vel + (n_step - 0.5) * (vel_n - vel) / step_elec
        ene_now = ene + (n_step - 0.5) * (ene_n - ene) / step_elec
        nac_now = nac + (n_step - 0.5) * (nac_n - nac) / step_elec
        # get k3
        den3 = den + egrad2 * sub_time / 2
        egrad3 = grad_elec(n_states, nac_now, den3, vel_now, ene_now)
        # interpolate all variables at time t+dt
        vel_now = vel + (n_step) * (vel_n - vel) / step_elec
        ene_now = ene + (n_step) * (ene_n - ene) / step_elec
        nac_now = nac + (n_step) * (nac_n - nac) / step_elec
        # get k4
        den4 = den + egrad3 * sub_time
        egrad4 = grad_elec(n_states, nac_now, den4, vel_now, ene_now)
        # do level-4 Runge-Kutta
        den = den + sub_time * (egrad1 + 2 * egrad2 + 2 * egrad3 + egrad4) / 6
        # decoherence
        den = decoherence(n_states, vel, mass, ene, cur_index, den, sub_time)
        # Calculate probability
        temprob_t = np.zeros(n_states)
        temprob_t_dt = np.zeros(n_states)
        for i in range(n_states):
            if i == cur_index:
                prob[i] = 0
            else:
                nacv_last = dot_nacv_vel(nac_last[cur_index, i], vel_last)
                nacv_now = dot_nacv_vel(nac_now[cur_index, i], vel_now)
                temprob_t[i] = np.real(nacv_last * np.conj(den_last[cur_index, i])) / np.real(den_last[cur_index, cur_index])
                temprob_t_dt[i] = np.real(nacv_now * np.conj(den[cur_index, i])) / np.real(den[cur_index, cur_index])
                prob[i] = prob[i] + (temprob_t[i] + temprob_t_dt[i]) * sub_time
    # correct probability
    prob = correct_prob(n_states, prob, cur_index)
    return prob, den

