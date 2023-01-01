import os
import torch as tc
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from bptt import saving
from bptt import PLRNN_model
from bptt import lstm_model
from bptt import rnn_model


def load_args(model_path):
    args_path = os.path.join(model_path, 'hypers.pkl')
    args = np.load(args_path, allow_pickle=True)
    return args


class Model(tc.nn.Module):
    def __init__(self, args=None, data_set=None):
        super().__init__()
        self.latent_model = None
        self.args = args
        self.data_set = data_set

        if args is not None:
            self.init_from_args(data_set)

    def forward(self, x, n=None,z0=None,grads=False):
        hidden_out = self.latent_model(x, n, z0)
        output = self.output_layer(hidden_out)
        if grads:
           return output,hidden_out
        else:
            return output

    def get_parameters(self):
        return list(self.latent_model.parameters())

    def get_num_trainable(self):
        '''
        Return the number of trainable parameters
        '''
        return sum([p.numel() if p.requires_grad else 0
                    for p in self.get_parameters()])

    def init(self, model, args):
        self.latent_model = model
        self.args = args

    def init_from_args(self, data_set):
        self.output_layer = self.init_obs_model(self.args.fix_obs_model)
        if self.args.load_model_path is not None:
            self.init_from_model_path(self.args.load_model_path)
        else:

            if self.args.model == "PLRNN":
                self.latent_model = PLRNN_model.PLRNN(self.args.dim_x, self.args.dim_z, self.args.n_bases, clip_range=self.args.clip_range, latent_model=self.args.latent_model,layer_norm=self.args.layer_norm,obs_model=self.output_layer)

            if self.args.model == "LSTM":
                self.latent_model = lstm_model.LSTM(self.args.dim_x, self.args.dim_z,obs_model=self.output_layer)

            if self.args.model == "RNN":
                self.latent_model = rnn_model.RNN(self.args.dim_x, self.args.dim_z, obs_model=self.output_layer)

    def init_obs_model(self, fix_output_layer):
        output_layer = nn.Linear(self.args.dim_z, self.args.dim_x, bias=False)
        return output_layer


    def load_obs_model(self, fix_output_layer):
        output_layer = nn.Linear(self.args['dim_z'], self.args['dim_x'], bias=False)

        return output_layer

    def init_from_model_path(self, model_path, epoch=None):
        self.args = load_args(model_path)


        self.output_layer = self.load_obs_model(self.args['fix_obs_model'])

        state_dict = self.load_statedict(model_path, 'model', epoch=epoch)
        self.latent_model = self.load_model(state_dict,self.output_layer)




    def load_model(self, state_dict,obs_model):
        try:
            clip_range = self.args['clip_range']
        except:
            clip_range = None

        if self.args['model'] == "PLRNN":
            latent_model = PLRNN_model.PLRNN(dim_x=self.args['dim_x'], dim_z=self.args['dim_z'],
                                  n_bases=self.args['n_bases'], clip_range=clip_range, latent_model=self.args['latent_model'],layer_norm=self.args['layer_norm'],obs_model=obs_model)
        elif self.args['model'] == "LSTM":
            latent_model = lstm_model.LSTM(self.args['dim_x'], self.args['dim_z'], obs_model=obs_model)
        elif self.args['model']== "RNN":
            latent_model = rnn_model.RNN(self.args['dim_x'], self.args['dim_z'], obs_model=obs_model)

        latent_model.load_state_dict(state_dict)
        return latent_model

    def load_statedict(self, model_path, model_name, epoch=None):
        if epoch is None:
            epoch = self.args['n_epochs']
        path = os.path.join(model_path, '{}_{}.pt'.format(model_name, str(epoch)))
        state_dict = tc.load(path)

        return state_dict

    def eval(self):
        self.latent_model.eval()

    def train(self):
        self.latent_model.train()
    def generate_free_trajectory(self,data,T,z0=None,n_repeat=1):
        latent_traj = self.latent_model.generate(T,data, z0,n_repeat)
        obs_traj = self.output_layer(latent_traj).squeeze(0)
        return obs_traj,latent_traj





    def plot_simulated(self,data, time_steps):
        X, Z = self.generate_free_trajectory(data,time_steps)
        fig = plt.figure()
        plt.title('simulated')
        plt.axis('off')
        plot_list = [X, Z]
        names = ['x', 'z']
        for i, x in enumerate(plot_list):
            fig.add_subplot(len(plot_list), 1, i + 1)
            plt.plot(x)
            plt.title(names[i])
        plt.xlabel('time steps')

    def plot_obs_simulated(self,data):
        time_steps = len(data)
        X_simulated, Z = self.generate_free_trajectory(data,time_steps)
        fig = plt.figure()
        plt.title('observations')
        plt.axis('off')
        n_units = data.shape[1]
        max_units = min([n_units, 10])
        max_time_steps = 1000
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(data[:max_time_steps, i])
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(X_simulated[:max_time_steps, i])
            ax.set_ylim(lim)
        plt.legend(['data', 'x simulated'])
        plt.xlabel('time steps')

    def plot_model_parameters(self):
        print_state_dict(self.gen_model.state_dict())


def print_state_dict(state_dict):
    for i, par_name in enumerate(state_dict.keys()):
        par = state_dict[par_name]
        if len(par.shape) == 1:
            par = np.expand_dims(par, 1)
        plt.figure()
        axes = plt.gca()
        plt.title(par_name)
        saving.plot_par_to_axes(axes, par)






