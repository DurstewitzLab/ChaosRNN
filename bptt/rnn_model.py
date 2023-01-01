import torch.nn as nn
import torch as tc
import math



class RNN(nn.Module):
    """
    RNN
    """

    def __init__(self, dim_x, dim_z,obs_model=None):
        super(RNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.latent_step = nn.RNN(input_size=self.d_x,hidden_size=self.d_z,num_layers=1,batch_first=True)
        self.obs_model = obs_model



    def forward(self, x,n, z0= None):
        '''creates forced trajectories'''

        b, T, dx = x.size()

        B = self.obs_model.weight#.detach()
        # no forcing obs. if n is not specified
        if n is None:
            n = T + 1
        if z0 is None:

            z = (tc.pinverse(B)@x[:,0].float().T).T
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b,1,self.d_x)
        else:
            z = z0
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros_like(x)


        zs = tc.empty(size=(b, T, self.d_z))

        for t in range(0, T):
            # interleave observation every n time steps
            if (t % n == 0 and t!=0):
                z = (tc.pinverse(B) @ x[:,t].float().T).T
                h = z.reshape((1, b, self.d_z))
            output, h = self.latent_step(inp, h)
            zs[:,t] = output.squeeze(1)

        return zs

    def generate(self, T, data, z0=None,n_repeat=1):
        '''creates freely generated (unforced) trajectories'''

        Z = []

        len, dx = data.size()
        b = n_repeat
        step = int(len / n_repeat)
        x_ = data[::step]

        B = self.obs_model.weight

        if z0 is None:
            z = (tc.pinverse(B) @ x_.float().T).T
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b, 1, self.d_x)
        else:
            z = z0
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b, 1, self.d_x)

        for t in range(0, T):
            output, h = self.latent_step(inp, h)

            Z.append(output)
        Z = tc.stack(Z, dim=1)
        shape = (n_repeat * T, self.d_z)
        Z = tc.reshape(Z, shape)
        return Z











