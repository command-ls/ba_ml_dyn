import numpy as np
import torch
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
import os
import torch.nn as nn
from torch.nn import functional
from torch.nn.init import xavier_uniform_, constant_
from torch.utils.data import Dataset, Subset, DataLoader
from functools import partial
from ase.db import connect
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
zeros_initializer = partial(constant_, val=0.0)
torch.set_default_dtype(torch.float32)


class SimpleEnvironmentProvider():
    def get_environment(self, atoms):
        n_atoms = atoms.get_global_number_of_atoms()
        if n_atoms == 1:
            neighborhood_idx = -np.ones((1, 1), dtype=np.float32)
            offsets = np.zeros((n_atoms, 1, 3), dtype=np.float32)
        else:
            neighborhood_idx = np.tile(np.arange(n_atoms, dtype=np.float32)[np.newaxis], (n_atoms, 1))
            neighborhood_idx = neighborhood_idx[~np.eye(n_atoms, dtype=bool)].reshape(n_atoms, n_atoms - 1)
            offsets = np.zeros((neighborhood_idx.shape[0], neighborhood_idx.shape[1], 3), dtype=np.float32)
        return neighborhood_idx, offsets


def get_center_of_mass(atoms):
    masses = atoms.get_masses()
    return np.dot(masses, atoms.arrays["positions"]) / masses.sum()


def _convert_atoms(atoms, output=None):
    environment_provider = SimpleEnvironmentProvider()
    centering_function = get_center_of_mass
    if output is None:
        inputs = {}
    else:
        inputs = output
    inputs["_atomic_numbers"] = atoms.numbers.astype(np.int32)
    positions = atoms.positions.astype(np.float32)
    positions -= centering_function(atoms)
    inputs["_positions"] = positions
    nbh_idx, offsets = environment_provider.get_environment(atoms)
    inputs["_neighbors"] = nbh_idx.astype(np.int32)
    return inputs


def dict_data_to_tensor(data):
    torch_properties = {}
    for name, prop in data.items():
        if prop.dtype in [np.int32]:
            torch_properties[name] = torch.IntTensor(prop)
        elif prop.dtype in [np.float32]:
            torch_properties[name] = torch.FloatTensor(prop.copy())
        else:
            print("please check!")
    return torch_properties


class AtomsData(Dataset):
    ENCODING = "utf-8"
    def __init__(self, dbpath, available_properties=None, load_only=None, units=None):
        self.dbpath = dbpath
        self._load_only = load_only
        self._available_properties = self._get_available_properties(available_properties)
        if units is None:
            units = [1.0] * len(self.available_properties)
        self.units = dict(zip(self.available_properties, units))

    @property
    def available_properties(self):
        return self._available_properties
    @property
    def load_only(self):
        if self._load_only is None:
            return self.available_properties
        return self._load_only

    def get_properties(self, idx, load_only=None):
        if load_only is None:
            load_only = self.available_properties
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()
        properties = {}
        for pname in load_only:
            properties[pname] = row.data[pname].astype(np.float32)
        properties = _convert_atoms(at, output=properties)
        return at, properties

    def get_atoms(self, idx):
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()
        return at

    def __len__(self):
        with connect(self.dbpath) as conn:
            return conn.count()
    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.load_only)
        properties["_idx"] = np.array([idx], dtype=np.int32)
        return dict_data_to_tensor(properties)

    def _get_available_properties(self, properties):
        if os.path.exists(self.dbpath) and len(self) != 0:
            with connect(self.dbpath) as conn:
                atmsrw = conn.get(1)
                db_properties = list(atmsrw.data.keys())
        else:
            db_properties = None
        if properties is not None:
            if db_properties is None or set(db_properties) == set(properties):
                return properties
        if db_properties is not None:
            return db_properties


class AtomsDataSubset(Subset):
    def __init__(self, dataset, indices):
        super(AtomsDataSubset, self).__init__(dataset, indices)
        self._load_only = None

    @property
    def available_properties(self):
        return self.dataset.available_properties

    @property
    def load_only(self):
        if self._load_only is None:
            return self.dataset.load_only
        return self._load_only

    def get_properties(self, idx, load_only=None):
        return self.dataset.get_properties(self.indices[idx], load_only)

    def set_load_only(self, load_only):
        self._load_only = list(load_only)


def train_test_split(data, num_train=None, num_val=None, split_file=None):
    if split_file is not None and os.path.exists(split_file):
        S = np.load(split_file)
        train_idx = S["train_idx"].tolist()
        val_idx = S["val_idx"].tolist()
        test_idx = S["test_idx"].tolist()
    else:
        num_train = num_train if num_train > 1 else num_train * len(data)
        num_val = num_val if num_val > 1 else num_val * len(data)
        num_train = int(num_train)
        num_val = int(num_val)
        idx = np.random.permutation(len(data))
        train_idx = idx[:num_train].tolist()
        val_idx = idx[num_train: num_train + num_val].tolist()
        test_idx = idx[num_train + num_val:].tolist()
        if split_file is not None:
            np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    train = AtomsDataSubset(data, train_idx)
    val = AtomsDataSubset(data, val_idx)
    test = AtomsDataSubset(data, test_idx)
    return train, val, test


def _collate_ase_atoms(examples):
    properties = examples[0]
    max_size = {prop: np.array(val.size(), dtype=np.int32) for prop, val in properties.items()}
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(max_size[prop], np.array(val.size(), dtype=np.int32))
    batch = {p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(examples[0][p].type()) for p, size in max_size.items()}
    has_atom_mask = "_atom_mask" in batch.keys()
    has_neighbor_mask = "_neighbor_mask" in batch.keys()
    if not has_neighbor_mask:
        batch["_neighbor_mask"] = torch.zeros_like(batch["_neighbors"]).float()
    if not has_atom_mask:
        batch["_atom_mask"] = torch.zeros_like(batch["_atomic_numbers"]).float()
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val
        if not has_neighbor_mask:
            nbh = properties["_neighbors"]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch["_neighbor_mask"][s] = mask
            batch["_neighbors"][s] = nbh * mask.long()
        if not has_atom_mask:
            z = properties["_atomic_numbers"]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch["_atom_mask"][s] = z > 0
    return batch


class StatisticsAccumulator:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0
    def add_sample(self, sample_value):
        n_batch = sample_value.size(0)
        for i in range(n_batch):
            self._add_sample(sample_value[i, :])
    def _add_sample(self, sample_value):
        self.count += 1
        delta_old = sample_value - self.mean
        self.mean += delta_old / self.count
        delta_new = sample_value - self.mean
        self.M2 += delta_old * delta_new
    def get_statistics(self):
        mean = self.mean
        stddev = torch.sqrt(self.M2 / self.count)
        return mean, stddev
    def get_mean(self):
        return self.mean
    def get_stddev(self):
        return torch.sqrt(self.M2 / self.count)


class AtomsLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=_collate_ase_atoms, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        super(AtomsLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)
    def get_statistics(self, property_names, divide_by_atoms=False, single_atom_ref=None):
        if type(property_names) is not list:
            property_names = [property_names]
        if type(divide_by_atoms) is not dict:
            divide_by_atoms = {prop: divide_by_atoms for prop in property_names}
        if single_atom_ref is None:
            single_atom_ref = {prop: None for prop in property_names}
        with torch.no_grad():
            statistics = {prop: StatisticsAccumulator() for prop in property_names}
            for row in self:
                for prop in property_names:
                    self._update_statistic(divide_by_atoms[prop], single_atom_ref[prop], prop, row, statistics[prop])
            means = {prop: s.get_mean() for prop, s in statistics.items()}
            stddevs = {prop: s.get_stddev() for prop, s in statistics.items()}
        return means, stddevs
    def _update_statistic(self, divide_by_atoms, single_atom_ref, property_name, row, statistics):
        property_value = row[property_name]
        if single_atom_ref is not None:
            z = row["_atomic_numbers"]
            p0 = torch.sum(torch.from_numpy(single_atom_ref[z]).float(), dim=1)
            property_value -= p0
        if divide_by_atoms:
            mask = torch.sum(row["_atom_mask"], dim=1, keepdim=True).view([-1, 1] + [1] * (property_value.dim() - 2))
            property_value /= mask
        statistics.add_sample(property_value)


class Dense(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, activation=None, weight_init=xavier_uniform_, bias_init=zeros_initializer):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation
        super(Dense, self).__init__(in_features, out_features, bias)
    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)
    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)
        return y


class GetItem(nn.Module):
    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key
    def forward(self, inputs):
        return inputs[self.key]


class Standardize(nn.Module):
    def __init__(self, mean, stddev, eps=1e-9):
        super(Standardize, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.register_buffer("eps", torch.ones_like(stddev) * eps)
    def forward(self, input):
        y = (input - self.mean) / (self.stddev + self.eps)
        return y


class ScaleShift(nn.Module):
    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
    def forward(self, input):
        y = input * self.stddev + self.mean
        return y


class Aggregate(nn.Module):
    def __init__(self, axis, mean=False, keepdim=True):
        super(Aggregate, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim
    def forward(self, input, mask=None):
        if mask is not None:
            input = input * mask[..., None]
        y = torch.sum(input, self.axis)
        if self.average:
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N
        return y


class CosineCutoff(nn.Module):
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


def shifted_softplus(x):
    return functional.softplus(x) - np.log(2.0)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=None, n_layers=2, activation=shifted_softplus):
        super(MLP, self).__init__()
        if n_hidden is None:
            c_neurons = n_in
            self.n_neurons = []
            for i in range(n_layers):
                self.n_neurons.append(c_neurons)
                c_neurons = max(n_out, c_neurons // 2)
            self.n_neurons.append(n_out)
        else:
            if type(n_hidden) is int:
                n_hidden = [n_hidden] * (n_layers - 1)
            self.n_neurons = [n_in] + n_hidden + [n_out]
        layers = [Dense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation) for i in range(n_layers - 1)]
        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activation=None))
        self.out_net = nn.Sequential(*layers)
    def forward(self, inputs):
        return self.out_net(inputs)


class CFConv(nn.Module):
    def __init__(self, n_in, n_filters, n_out, filter_network, cutoff, axis=2):
        super(CFConv, self).__init__()
        activation = shifted_softplus
        cutoff_network = CosineCutoff(cutoff)
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=False)
    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)
        W = self.filter_network(f_ij)
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)
        y = self.in2f(x)
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y)
        return y


class SchNetInteraction(nn.Module):
    def __init__(self, n_atom_basis, n_gauss, n_filters, cutoff):
        super(SchNetInteraction, self).__init__()
        self.filter_network = nn.Sequential(Dense(n_gauss, n_filters, activation=shifted_softplus),
                                            Dense(n_filters, n_filters))
        self.cfconv = CFConv(n_atom_basis, n_filters, n_atom_basis, self.filter_network, cutoff)
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)
    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        return v


def gaussian_smearing(distances, offset, widths):
    coef = -0.5 / torch.pow(widths, 2)
    diff = distances[:, :, :, None] - offset[None, None, None, :]
    gauss = torch.exp(coef * torch.pow(diff, 2))
    return gauss


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, n_gauss=50, trainable=False):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, n_gauss)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
    def forward(self, distances):
        return gaussian_smearing(distances, self.offsets, self.width)


def atom_distances(positions, neighbors, neighbor_mask=None):
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device, dtype=torch.long)[:, None, None]
    pos_xyz = positions[idx_m, neighbors[:, :, :], :]
    dist_vec = pos_xyz - positions[:, :, None, :]
    distances = torch.norm(dist_vec, 2, 3)
    if neighbor_mask is not None:
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        distances = tmp_distances
    return distances


class AtomDistances(nn.Module):
    def __init__(self):
        super(AtomDistances, self).__init__()
    def forward(self, positions, neighbors, neighbor_mask=None):
        return atom_distances(positions, neighbors, neighbor_mask=neighbor_mask)


class SchNet(nn.Module):
    def __init__(self, n_atom_basis=128, n_filters=128, n_interactions=3, cutoff=5.0, n_gauss=25, coupled_interactions=False, num_embeddings=100, train_gauss=False):
        super(SchNet, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.embedding = nn.Embedding(num_embeddings, n_atom_basis, padding_idx=0)
        self.distances = AtomDistances()
        self.distance_expansion = GaussianSmearing(0.0, cutoff, n_gauss, trainable=train_gauss)
        if coupled_interactions:
            self.interactions = nn.ModuleList([SchNetInteraction(n_atom_basis, n_gauss, n_filters, cutoff)] * n_interactions)
        else:
            self.interactions = nn.ModuleList([SchNetInteraction(n_atom_basis, n_gauss, n_filters, cutoff) for _ in range(n_interactions)])
    def forward(self, inputs):
        atomic_numbers = inputs["_atomic_numbers"]
        positions = inputs["_positions"]
        neighbors = inputs["_neighbors"].long()
        neighbor_mask = inputs["_neighbor_mask"]
        x = self.embedding(atomic_numbers)
        r_ij = self.distances(positions, neighbors, neighbor_mask=neighbor_mask)
        f_ij = self.distance_expansion(r_ij)
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v
        return x


class Atomwise(nn.Module):
    def __init__(self, n_in, n_out=1, n_layers=2, n_neurons=None, activation=shifted_softplus, property="y", derivative=None, negative_dr=False, create_graph=False, mean=None, stddev=None, outnet=None):
        super(Atomwise, self).__init__()
        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.derivative = derivative
        self.negative_dr = negative_dr
        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev
        if outnet is None:
            self.out_net = nn.Sequential(GetItem("representation"), MLP(n_in, n_out, n_neurons, n_layers, activation))
        else:
            self.out_net = outnet
        self.standardize = ScaleShift(mean, stddev)
        self.atom_pool = Aggregate(axis=1, mean=False)
    def forward(self, inputs):
        atom_mask = inputs["_atom_mask"]
        yi = self.out_net(inputs)
        yi = self.standardize(yi)
        y = self.atom_pool(yi, atom_mask)
        result = {self.property: y}
        create_graph = True if self.training else self.create_graph
        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = torch.autograd.grad(result[self.property], inputs["_positions"], grad_outputs=torch.ones_like(result[self.property]), create_graph=create_graph, retain_graph=True)[0]
            result[self.derivative] = sign * dy
        return result


class AtomisticModel(nn.Module):
    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        self.requires_dr = any([om.derivative for om in self.output_modules])
    def forward(self, inputs):
        if self.requires_dr:
            inputs["_positions"].requires_grad_()
        inputs["representation"] = self.representation(inputs)
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs

