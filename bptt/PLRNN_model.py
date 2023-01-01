import torch.nn as nn
import torch as tc
import math



class PLRNN(nn.Module):
    """
    Piece-wise Linear Recurrent Neural Network (Durstewitz 2017)
    """

    LATENT_MODELS = ['PLRNN', 'clipped-PLRNN', 'dendr-PLRNN']

    def __init__(self, dim_x: int, dim_z: int, n_bases: int, clip_range: float,
                 latent_model: str, layer_norm: bool,obs_model=None):
        super(PLRNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.n_bases = n_bases
        self.use_bases = False
        self.obs_model = obs_model

        if latent_model == 'PLRNN':
            if n_bases > 0:
                Warning("Chosen model is PLRNN, the n_bases Parameter has no effect here!")
            self.latent_step = PLRNN_Step(dz=self.d_z, clip_range=clip_range, layer_norm=layer_norm)
        else:
            if latent_model == 'clipped-PLRNN':
                assert n_bases > 0, "n_bases has to be a positive integer for clipped-PLRNN!"
                self.latent_step = PLRNN_Clipping_Step(self.n_bases, dz=self.d_z, clip_range=clip_range, layer_norm=layer_norm)
                self.use_bases = True
            elif latent_model == 'dendr-PLRNN':
                assert n_bases > 0, "n_bases has to be a positive integer for dendr-PLRNN!"
                self.latent_step = PLRNN_Basis_Step(self.n_bases, dz=self.d_z, clip_range=clip_range, layer_norm=layer_norm)
                self.use_bases = True
            else:
                raise NotImplementedError(f"{latent_model} is not yet implemented. Use one of: {self.LATENT_MODELS}.")

    def get_latent_parameters(self):
        AW = self.latent_step.AW
        A = tc.diag(AW)
        W = AW - tc.diag(A)

        h = self.latent_step.h
        return A, W, h

    def get_basis_expansion_parameters(self):
        alphas = self.latent_step.alphas
        thetas = self.latent_step.thetas
        return alphas, thetas

    def get_parameters(self):
        params = self.get_latent_parameters()
        if self.use_bases:
            params += self.get_basis_expansion_parameters()
        return params

    def forward(self, x, n=None, z0=None):
        '''creates forced trajectories'''

        # switch dimensions for performance reasons
        x_ = x.permute(1, 0, 2)
        T, b, dx = x_.size()
        B = self.obs_model.weight

        if n is None:
            n = T + 1
        A,W = split_diag_offdiag(self.latent_step.AW)
        h = self.latent_step.h
        B_tilde = tc.pinverse(B)

        # initial state
        if z0 is None:
            z = (B_tilde@x_[0].float().T).T
        else:
            z = z0
        zs = tc.empty(size=(T, b, self.d_z), device=x.device)

        for t in range(0, T):
            # interleave observation every n time steps
            if (t % n == 0 and t!=0):
                z = (B_tilde @ x_[t].float().T).T
            z = self.latent_step(z,A,W,h)
            zs[t] = z

        return zs.permute(1, 0, 2)

    def generate(self,T,data,z0=None,n_repeat=1):
        '''creates freely generated (unforced) trajectories'''

        Z= []
        len = data.shape[0]
        step=int(len/n_repeat)
        x= data[::step]
        B = self.obs_model.weight
        A, W = split_diag_offdiag(self.latent_step.AW)
        h = self.latent_step.h
        B_tilde = tc.pinverse(B)

        if z0 is None:
            z = (B_tilde@x.float().T).T
        else:
            z= z0
        Z.append(z)

        for t in range(T-1):
            z = self.latent_step(z,A,W,h)
            Z.append(z)

        Z = tc.stack(Z, dim=1)
        shape = (n_repeat * T, self.d_z)
        Z = tc.reshape(Z, shape)
        return Z

class Latent_Step(nn.Module):
    def __init__(self, clip_range, dz, ds,layer_norm):
        super(Latent_Step, self).__init__()
        self.clip_range = clip_range
        self.nonlinearity = nn.ReLU()
        self.dz = dz
        self.ds = ds

        if self.ds is not None:
            self.C = nn.Parameter(tc.randn(self.dz, self.ds), requires_grad=True)
        if layer_norm==1:
            self.norm = lambda z: z - z.mean(dim=-1, keepdim=True)

        else:
            print("no Layer Norm")

            self.norm = nn.Identity()

    def init_AW_uniform(self):
        AW = tc.empty(self.dz, self.dz)
        r = 1 / math.sqrt(self.dz)
        nn.init.uniform_(AW, -r, r)
        return nn.Parameter(AW, requires_grad=True)

    def init_h_uniform(self):
        h = tc.empty(self.dz)
        r = 1 / math.sqrt(self.dz)
        nn.init.uniform_(h, -r, r)
        return nn.Parameter(h, requires_grad=True)

    def init_AW_random_max_ev(self):
        AW = tc.eye(self.dz) + 0.1 * tc.randn(self.dz, self.dz)
        max_ev = tc.max(tc.abs(tc.eig(AW)[0]))
        return nn.Parameter(AW / max_ev, requires_grad=True)

    def init_AW(self):
        # from: Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network with ReLU Nonlinearity
        matrix_random = tc.randn(self.dz, self.dz)
        matrix_positive_normal = 1 / (self.dz * self.dz) * matrix_random @ matrix_random.T
        matrix = tc.eye(self.dz) + matrix_positive_normal
        max_ev = tc.max(tc.abs(tc.eig(matrix)[0]))
        matrix_spectral_norm_one = matrix / max_ev

        return nn.Parameter(matrix_spectral_norm_one, requires_grad=True)

    def clip_z_to_range(self, z):
        if self.clip_range is not None:
            z = tc.max(z, -self.clip_range * tc.ones_like(z))
            z = tc.min(z, self.clip_range * tc.ones_like(z))
        return z

    def add_input(self, s):
        if s is not None:
            input = tc.einsum('ij,bj->bi', (self.C, s))
            if input.shape[0] > 1:  # for batch-wise processing
                input = input[1:]
        else:
            input = 0
        return input


class PLRNN_Step(Latent_Step):
    def __init__(self, dz, ds=None, clip_range=None,layer_norm=False):
        super(PLRNN_Step, self).__init__(clip_range=clip_range, dz=dz, ds=ds,layer_norm=layer_norm)
        self.AW = self.init_AW_uniform()
        self.h = self.init_h_uniform()
    def forward(self, z, A=None, W=None, h = None):
        if A is None or W is None:
            A, W = split_diag_offdiag(self.AW)
        if h is None:
            h = self.h
        z_activated = tc.relu(self.norm(z))
        z = A * z + z_activated @ W.t() + h
        return z

class PLRNN_Basis_Step(Latent_Step):
    def __init__(self, db, dz=None, ds=None, clip_range=None,layer_norm=False):
        super(PLRNN_Basis_Step, self).__init__(clip_range=clip_range, dz=dz, ds=ds,layer_norm=layer_norm)
        self.AW = self.init_AW_uniform()

        self.h = nn.Parameter(tc.randn(self.dz), requires_grad=True)
        self.db = db
        self.thetas = nn.Parameter(tc.randn(self.dz, self.db), requires_grad=True)
        self.alphas = nn.Parameter(tc.randn(self.db), requires_grad=True)

    def forward(self, z, A=None, W=None):
        if A is None or W is None:
            A, W = split_diag_offdiag(self.AW)
        z_norm = self.norm(z).unsqueeze_(-1)
        # thresholds are broadcasted into the added dimension of z
        be = tc.sum(self.alphas * tc.relu(z_norm + self.thetas), dim=-1)
        z = A * z + be @ W.t() + self.h
        return self.clip_z_to_range(z)


class PLRNN_Clipping_Step(Latent_Step):
    def __init__(self, db, dz=None, ds=None, clip_range=None,layer_norm=False):
        super(PLRNN_Clipping_Step, self).__init__(clip_range=clip_range, dz=dz, ds=ds,layer_norm=layer_norm)
        self.AW = self.init_AW_uniform()
        self.h = self.init_h_uniform() #nn.Parameter(tc.randn(self.dz), requires_grad=True)
        self.db = db
        self.thetas = nn.Parameter(tc.rand(self.dz, self.db), requires_grad=True)
        self.alphas = nn.Parameter(tc.rand(self.db), requires_grad=True)

    def forward(self, z,A=None,W=None,h=None):
        if A is None or W is None:
            A, W = split_diag_offdiag(self.AW)

        z_norm = self.norm(z).unsqueeze(-1)
        be_clip = tc.sum(self.alphas * (tc.relu(z_norm + self.thetas) - tc.relu(z_norm)), dim=-1)
        z = A * z + be_clip @ W.t() + self.h
        return z


def split_diag_offdiag(AW):
    diag = tc.diag(AW)
    off_diag = AW- tc.diag(diag)
    #diag = tc.tanh(diag)
    return diag, off_diag
