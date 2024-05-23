import torch
from torch import nn
from copy import deepcopy as cp
seed = 10
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)


def init_seq(self):
    self.model_n1 = nn.Sequential()
    self.model_n2 = nn.Sequential()
    self.model_c1 = nn.Sequential()
    self.model_c2 = nn.Sequential()
    self.model_c3 = nn.Sequential()
    self.model_c4 = nn.Sequential()
    self.model_c5 = nn.Sequential()
    self.model_c6 = nn.Sequential()
    self.model_c7 = nn.Sequential()
    self.model_c8 = nn.Sequential()
    self.model_c9 = nn.Sequential()
    self.model_c10 = nn.Sequential()
    self.model_c11 = nn.Sequential()
    self.model_c12 = nn.Sequential()
    self.model_c13 = nn.Sequential()
    self.model_c14 = nn.Sequential()
    self.model_h1 = nn.Sequential()
    self.model_h2 = nn.Sequential()
    self.model_h3 = nn.Sequential()
    self.model_h4 = nn.Sequential()
    self.model_h5 = nn.Sequential()
    self.model_h6 = nn.Sequential()
    self.model_h7 = nn.Sequential()
    self.model_h8 = nn.Sequential()
    self.model_h9 = nn.Sequential()
    self.model_h10 = nn.Sequential()
    self.model_h11 = nn.Sequential()
    self.model_h12 = nn.Sequential()


def model_add_module(model_list, fea_in, hidden_num):
    for model in model_list:
        model.add_module(f'Linear {1}', nn.Linear(fea_in, hidden_num))
        model.add_module(f'Activator {1}', nn.ReLU())
        model.add_module(f'Linear {-1}', nn.Linear(hidden_num, 1))


def model_init_wb(model_list):
    for model in model_list:
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                layer.bias.data.fill_(0.0)


def model_same_wb(model_list_refer, model_list_n, model_list_c, model_list_h):
    model_n1 = model_list_refer[0]
    model_c1 = model_list_refer[1]
    model_h1 = model_list_refer[2]
    for model_n in model_list_n:
        model_n[0].weight = cp(model_n1[0].weight)
        model_n[0].bias = cp(model_n1[0].bias)
        model_n[2].weight = cp(model_n1[2].weight)
        model_n[2].bias = cp(model_n1[2].bias)
    for model_c in model_list_c:
        model_c[0].weight = cp(model_c1[0].weight)
        model_c[0].bias = cp(model_c1[0].bias)
        model_c[2].weight = cp(model_c1[2].weight)
        model_c[2].bias = cp(model_c1[2].bias)
    for model_h in model_list_h:
        model_h[0].weight = cp(model_h1[0].weight)
        model_h[0].bias = cp(model_h1[0].bias)
        model_h[2].weight = cp(model_h1[2].weight)
        model_h[2].bias = cp(model_h1[2].bias)


def calc_ene_pred(self, des_list):
    ene_atom_n1 = self.model_n1(des_list[0])
    ene_atom_n2 = self.model_n2(des_list[1])
    ene_atom_c1 = self.model_c1(des_list[2])
    ene_atom_c2 = self.model_c2(des_list[3])
    ene_atom_c3 = self.model_c3(des_list[4])
    ene_atom_c4 = self.model_c4(des_list[5])
    ene_atom_c5 = self.model_c5(des_list[6])
    ene_atom_c6 = self.model_c6(des_list[7])
    ene_atom_c7 = self.model_c7(des_list[8])
    ene_atom_c8 = self.model_c8(des_list[9])
    ene_atom_c9 = self.model_c9(des_list[10])
    ene_atom_c10 = self.model_c10(des_list[11])
    ene_atom_c11 = self.model_c11(des_list[12])
    ene_atom_c12 = self.model_c12(des_list[13])
    ene_atom_c13 = self.model_c13(des_list[14])
    ene_atom_c14 = self.model_c14(des_list[15])
    ene_atom_h1 = self.model_h1(des_list[16])
    ene_atom_h2 = self.model_h2(des_list[17])
    ene_atom_h3 = self.model_h3(des_list[18])
    ene_atom_h4 = self.model_h4(des_list[19])
    ene_atom_h5 = self.model_h5(des_list[20])
    ene_atom_h6 = self.model_h6(des_list[21])
    ene_atom_h7 = self.model_h7(des_list[22])
    ene_atom_h8 = self.model_h8(des_list[23])
    ene_atom_h9 = self.model_h9(des_list[24])
    ene_atom_h10 = self.model_h10(des_list[25])
    ene_atom_h11 = self.model_h11(des_list[26])
    ene_atom_h12 = self.model_h12(des_list[27])
    ene_pred = [ene_atom_n1 + ene_atom_n2 + ene_atom_c1 + ene_atom_c2 + ene_atom_c3 + ene_atom_c4 + ene_atom_c5 +
                ene_atom_c6 + ene_atom_c7 + ene_atom_c8 + ene_atom_c9 + ene_atom_c10 + ene_atom_c11 + ene_atom_c12 +
                ene_atom_c13 + ene_atom_c14 + ene_atom_h1 + ene_atom_h2 + ene_atom_h3 + ene_atom_h4 + ene_atom_h5 +
                ene_atom_h6 + ene_atom_h7 + ene_atom_h8 + ene_atom_h9 + ene_atom_h10 + ene_atom_h11 + ene_atom_h12][0]
    return ene_pred


def calc_force_pred(ene_pred, des_list, der_list):
    de_dD_n1 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[0], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_n2 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[1], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c1 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[2], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c2 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[3], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c3 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[4], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c4 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[5], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c5 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[6], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c6 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[7], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c7 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[8], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c8 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[9], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c9 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[10], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c10 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[11], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c11 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[12], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c12 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[13], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c13 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[14], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_c14 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[15], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h1 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[16], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h2 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[17], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h3 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[18], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h4 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[19], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h5 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[20], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h6 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[21], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h7 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[22], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h8 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[23], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h9 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[24], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h10 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[25], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h11 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[26], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]
    de_dD_h12 = torch.autograd.grad(outputs=ene_pred, inputs=des_list[27], grad_outputs=torch.ones_like(ene_pred), create_graph=True)[0]

    force_pred_n1 = -torch.einsum('jkl,l->jk', [der_list[0], de_dD_n1])
    force_pred_n2 = -torch.einsum('jkl,l->jk', [der_list[1], de_dD_n2])
    force_pred_c1 = -torch.einsum('jkl,l->jk', [der_list[2], de_dD_c1])
    force_pred_c2 = -torch.einsum('jkl,l->jk', [der_list[3], de_dD_c2])
    force_pred_c3 = -torch.einsum('jkl,l->jk', [der_list[4], de_dD_c3])
    force_pred_c4 = -torch.einsum('jkl,l->jk', [der_list[5], de_dD_c4])
    force_pred_c5 = -torch.einsum('jkl,l->jk', [der_list[6], de_dD_c5])
    force_pred_c6 = -torch.einsum('jkl,l->jk', [der_list[7], de_dD_c6])
    force_pred_c7 = -torch.einsum('jkl,l->jk', [der_list[8], de_dD_c7])
    force_pred_c8 = -torch.einsum('jkl,l->jk', [der_list[9], de_dD_c8])
    force_pred_c9 = -torch.einsum('jkl,l->jk', [der_list[10], de_dD_c9])
    force_pred_c10 = -torch.einsum('jkl,l->jk', [der_list[11], de_dD_c10])
    force_pred_c11 = -torch.einsum('jkl,l->jk', [der_list[12], de_dD_c11])
    force_pred_c12 = -torch.einsum('jkl,l->jk', [der_list[13], de_dD_c12])
    force_pred_c13 = -torch.einsum('jkl,l->jk', [der_list[14], de_dD_c13])
    force_pred_c14 = -torch.einsum('jkl,l->jk', [der_list[15], de_dD_c14])
    force_pred_h1 = -torch.einsum('jkl,l->jk', [der_list[16], de_dD_h1])
    force_pred_h2 = -torch.einsum('jkl,l->jk', [der_list[17], de_dD_h2])
    force_pred_h3 = -torch.einsum('jkl,l->jk', [der_list[18], de_dD_h3])
    force_pred_h4 = -torch.einsum('jkl,l->jk', [der_list[19], de_dD_h4])
    force_pred_h5 = -torch.einsum('jkl,l->jk', [der_list[20], de_dD_h5])
    force_pred_h6 = -torch.einsum('jkl,l->jk', [der_list[21], de_dD_h6])
    force_pred_h7 = -torch.einsum('jkl,l->jk', [der_list[22], de_dD_h7])
    force_pred_h8 = -torch.einsum('jkl,l->jk', [der_list[23], de_dD_h8])
    force_pred_h9 = -torch.einsum('jkl,l->jk', [der_list[24], de_dD_h9])
    force_pred_h10 = -torch.einsum('jkl,l->jk', [der_list[25], de_dD_h10])
    force_pred_h11 = -torch.einsum('jkl,l->jk', [der_list[26], de_dD_h11])
    force_pred_h12 = -torch.einsum('jkl,l->jk', [der_list[27], de_dD_h12])
    force_pred = [force_pred_n1 + force_pred_n2 + force_pred_c1 + force_pred_c2 +
                  force_pred_c3 + force_pred_c4 + force_pred_c5 + force_pred_c6 +
                  force_pred_c7 + force_pred_c8 + force_pred_c9 + force_pred_c10 +
                  force_pred_c11 + force_pred_c12 + force_pred_c13 + force_pred_c14 +
                  force_pred_h1 + force_pred_h2 + force_pred_h3 + force_pred_h4 +
                  force_pred_h5 + force_pred_h6 + force_pred_h7 + force_pred_h8 +
                  force_pred_h9 + force_pred_h10 + force_pred_h11 + force_pred_h12][0]
    return force_pred


class BPNN(nn.Module):
    def __init__(self, fea_in, hidden_num):
        super(BPNN, self).__init__()
        init_seq(self)
        model_list = [self.model_n1, self.model_n2, self.model_c1, self.model_c2, self.model_c3, self.model_c4,
                      self.model_c5,
                      self.model_c6, self.model_c7, self.model_c8, self.model_c9, self.model_c10, self.model_c11,
                      self.model_c12,
                      self.model_c13, self.model_c14, self.model_h1, self.model_h2, self.model_h3, self.model_h4,
                      self.model_h5,
                      self.model_h6, self.model_h7, self.model_h8, self.model_h9, self.model_h10, self.model_h11,
                      self.model_h12]
        model_list_refer = [self.model_n1, self.model_c1, self.model_h1]
        model_list_n = [self.model_n2]
        model_list_c = [self.model_c2, self.model_c3, self.model_c4, self.model_c5, self.model_c6, self.model_c7,
                        self.model_c8,
                        self.model_c9, self.model_c10, self.model_c11, self.model_c12, self.model_c13, self.model_c14]
        model_list_h = [self.model_h2, self.model_h3, self.model_h4, self.model_h5, self.model_h6, self.model_h7,
                        self.model_h8,
                        self.model_h9, self.model_h10, self.model_h11, self.model_h12]
        model_add_module(model_list, fea_in, hidden_num)
        model_init_wb(model_list)
        model_same_wb(model_list_refer, model_list_n, model_list_c, model_list_h)

    def forward(self, des_list, *der_list):
        if not der_list:
            ene_pred = calc_ene_pred(self, des_list)
            return ene_pred
        else:
            der_list = der_list[0]
            ene_pred = calc_ene_pred(self, des_list)
            force_pred = calc_force_pred(ene_pred, des_list, der_list)
            return [ene_pred, force_pred]
