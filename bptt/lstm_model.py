import torch.nn as nn
import torch as tc
import math



class LSTM(nn.Module):
    """
    LSTM
    """

    def __init__(self, dim_x, dim_z,obs_model=None):
        super(LSTM, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        n = 1
        self.latent_step = nn.LSTM(input_size=self.d_x,hidden_size=self.d_z,num_layers=n,batch_first=True)
        self.obs_model = obs_model
        layer_norm = 0
        if layer_norm==1:
            self.norm = nn.LayerNorm(self.d_z,elementwise_affine=False)
        else:
            self.norm = nn.Identity()

    def forward(self, x,n, z0= None):
        '''creates forced trajectories'''

        b, T, dx = x.size()
        B = self.obs_model.weight#.detach()

        # no forcing if n is not specified
        if n is None:
            n = T + 1
        if z0 is None:
            z = (tc.pinverse(B)@x[:,0].float().T).T
            c = tc.zeros((1, b, self.d_z))
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b,1,self.d_x)
        else:
            z = z0
            c = tc.zeros((1, b, self.d_z))
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b,1,self.d_x)

        zs = tc.empty(size=(b, T, self.d_z))

        for t in range(0, T):
            # interleave observation every n time steps
            # for truncated BPTT copy in (*)
            if (t % n == 0 and t !=0):
                z = (tc.pinverse(B) @ x[:,t].float().T).T # (*) copy this out
                c = tc.zeros((1, b, self.d_z)) # (*) c.detach()
                h = z.reshape((1, b, self.d_z)) # (*)  h.detach()
            output, (h, c) = self.latent_step(inp, (h, self.norm(c)))
            zs[:,t] = output.squeeze(1)

        return zs

    def generate(self, T, data, z0=None,n_repeat=1):
        '''creates freely generated (unforced) trajectories'''

        Z = []
        len, dx = data.size()
        b=n_repeat
        step = int(len / n_repeat)
        x_ = data[::step]

        B = self.obs_model.weight  # .detach()
        # no interleaving obs. if n is not specified
        if z0 is None:
            z = (tc.pinverse(B) @ x_.float().T).T
            c = tc.zeros((1, b, self.d_z))
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b, 1, self.d_x)
        else:
            z = z0
            c = tc.zeros((1, b, self.d_z))
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b, 1, self.d_x)

        Z.append(z.unsqueeze(1))
        for t in range(T - 1):
            output, (h, c) = self.latent_step(inp, (h, self.norm(c)))
            Z.append(output)

        Z = tc.stack(Z, dim=1)
        shape = (n_repeat * T, self.d_z)
        Z = tc.reshape(Z, shape)
        return Z













