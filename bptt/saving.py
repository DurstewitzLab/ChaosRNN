import torch as tc
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorboardX import utils as tb_utils
import seaborn as sns

import main_eval
import utils


class Saver:
    def __init__(self, writer, save_path, args, data_set, regularizer):
        self.writer = writer
        self.save_path = save_path
        self.args = args
        self.data_set = data_set
        self.model = None
        self.current_epoch = None
        self.current_model = None
        self.regularizer = regularizer
        self.initial_save()

    def initial_save(self):
        if self.args.use_tb:
            self.save_dataset()

    def save_dataset(self):
        dataset_snippet = self.data_set.data[:1000]
        plt.plot(dataset_snippet)
        plt.title('Observations')
        plt.xlabel('time steps')
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='data set', global_step=None)
        plt.close()

    def epoch_save(self, model, epoch):
        tc.save(model.latent_model.state_dict(), os.path.join(self.save_path, 'model_{}.pt'.format(epoch)))
        self.current_epoch = epoch
        self.current_model = model

        if self.args.use_tb:
            with tc.no_grad():
                self.save_loss_terms()
                self.save_metrics()
                if epoch % (self.current_model.args.n_epochs / 10) == 0:
                #self.save_inferred()
                    self.save_simulated()
                    self.save_parameters()

                #self.plot_as_image()

    def save_loss_terms(self):
        batch_index = 0
        batch_data = tc.tensor(self.data_set.data[:1000].reshape((1,1000,self.args.dim_x))) #self.data_set.data[batch_index]

        latent_model = self.current_model.latent_model
        if self.current_model.args.model =="PLRNN":
            latent_model_parameters = latent_model.get_latent_parameters()
            loss_reg = self.regularizer.loss(latent_model_parameters)
            self.writer.add_scalar(tag='loss_regularization', scalar_value=loss_reg, global_step=self.current_epoch)

            A, W, h = latent_model_parameters
            L2A = tc.linalg.norm(tc.diag(A), 2)
        else: L2A = float('nan')


        loss_func = tc.nn.MSELoss()
        pred = self.current_model(batch_data, self.args.n_interleave)

        loss = loss_func(pred, batch_data)

        total_norm = tc.nn.utils.clip_grad_norm_(self.current_model.get_parameters(), 1.)


        self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=self.current_epoch)
        self.writer.add_scalar(tag='L2-norm A', scalar_value=L2A, global_step=self.current_epoch)
        self.writer.add_scalar(tag='total_grad_norm', scalar_value=total_norm, global_step=self.current_epoch)

        print("Epoch {}/{}: Loss {:.6f}, L2-norm A {:.1f}".format(
            str(self.current_epoch).zfill(4), self.args.n_epochs,
            float(loss), float(L2A)))

    def save_metrics(self):
        """Evaluate metrics on a subset of the training data, then save them to tensorboard"""
        for metric in self.args.metrics:
            data_batch = utils.read_data(self.args.data_path)
            data_subset = data_batch#[:1000]
            metric_value = main_eval.eval_model_on_data_with_metric(model=self.current_model, data=data_subset, metric=metric)
            metric_value = metric_value[0]  # only take first metric value, e.g. mse 1 step ahead, and klz mc
            tag = 'metric_{}'.format(metric)
            self.writer.add_scalar(tag=tag, scalar_value=metric_value, global_step=self.current_epoch)
            print("{}: {:.1f}".format(metric, metric_value))

    def save_parameters(self):
        par_dict = {**dict(self.current_model.latent_model.state_dict())}
        par_to_tb(par_dict, epoch=self.current_epoch, writer=self.writer)



    # def save_simulated(self):
    #     data = tc.tensor(self.data_set.data[:1000])
    #     time_steps = len(data)
    #
    #     self.current_model.plot_simulated(data,time_steps)
    #     figure = plt.gcf()
    #     save_figure_to_tb(figure, self.writer, text='curve trial simulated', global_step=self.current_epoch)
    #     plt.close()
    #     self.current_model.plot_obs_simulated(data)
    #     save_plot_to_tb(self.writer, text='curve trial simulated against data'.format(0),
    #                     global_step=self.current_epoch)
    #     plt.close()
    def save_simulated(self):
        T = 1000
        data = tc.tensor(self.data_set.data[:T])

        self.current_model.plot_simulated(data, T)
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='curve trial simulated',
                          global_step=self.current_epoch)
        plt.close()

        self.current_model.plot_obs_simulated(data)
        save_plot_to_tb(self.writer, text='curve trial simulated against data'.format(0),
                        global_step=self.current_epoch)
        plt.close()

    def save_figure_to_tb(figure, writer, text, global_step=None):
        image = tb_utils.figure_to_image(figure)
        writer.add_image(text, image, global_step=global_step)

    def get_min_max(self, values):
        list_ = list(values)
        indices = [i for i in range(len(list_)) if list_[i] == 1]
        return min(indices), max(indices)


    def plot_as_image(self):
        time_steps = 1000
        data_generated = self.current_model.gen_model.get_observed_time_series(time_steps=time_steps + 1000)
        data_generated = data_generated[1000:1000 + time_steps]
        data_ground_truth = self.data_set.data[0][:time_steps]
        data_generated = data_generated[:(data_ground_truth.shape[0])]  # in case trial data is shorter than time_steps

        plt.subplot(121)
        plt.title('ground truth')
        plt.imshow(data_ground_truth, aspect=0.05, origin='lower', interpolation='none', cmap='Blues_r')
        plt.xlabel('observations')
        plt.ylabel('time steps')
        plt.subplot(122)
        plt.title('simulated')
        plt.imshow(data_generated, aspect=0.05, origin='lower', interpolation='none', cmap='Blues_r')
        plt.ylabel('time steps')
        plt.xlabel('observations')

        save_plot_to_tb(self.writer, text='curve image'.format(), global_step=self.current_epoch)


def initial_condition_trial_to_tb(gen_model, epoch, writer):
    for i in range(len(gen_model.z0)):
        trial_z0 = gen_model.z0[i].unsqueeze(0)
        x = gen_model.get_observed_time_series(800, trial_z0)  # TODO magic length of trial
        plt.figure()
        plt.title('trial {}'.format(i))
        plt.plot(x)
        figure = plt.gcf()
        save_figure_to_tb(figure, writer, text='curve_trial{}'.format(i + 1), global_step=epoch)


def data_plot(x):
    x = x.cpu().detach().numpy()
    plt.ylim(top=4, bottom=-4)
    plt.xlim(right=4, left=-4)
    plt.scatter(x[:, 0], x[:, -1], s=3)
    plt.title('{} time steps'.format(len(x)))
    return plt.gcf()


def save_plot_to_tb(writer, text, global_step=None):
    figure = plt.gcf()
    save_figure_to_tb(figure, writer, text, global_step)


def save_figure_to_tb(figure, writer, text, global_step=None):
    image = tb_utils.figure_to_image(figure)
    writer.add_image(text, image, global_step=global_step)


def save_data_to_tb(data, writer, text, global_step=None):
    if type(data) is list:
        for i in range(len(data)):
            plt.figure()
            plt.title('trial {}'.format(i))
            plt.plot(data[i])
            figure = plt.gcf()
            save_figure_to_tb(figure=figure, writer=writer, text='curve_trial{}_data'.format(i), global_step=None)
    else:
        plt.figure()
        plt.plot(data)
        figure = plt.gcf()
        # figure = data_plot(data)
        save_figure_to_tb(figure=figure, writer=writer, text=text, global_step=global_step)


def par_to_tb(par_dict, epoch, writer):
    for key in par_dict.keys():
        par = par_dict[key]
        if len(par.shape) == 1:
            par = np.expand_dims(par, 1)
        par_to_image(par, par_name=key)
        save_plot_to_tb(writer, text='par_{}'.format(key), global_step=epoch)
        plt.close()


def par_to_image(par, par_name):
    plt.figure()
    # plt.title(par_name)
    sns.set_context('paper', font_scale=1.)
    sns.set_style('white')
    max_dim = max(par.shape)
    use_annot = not (max_dim > 20)
    sns.heatmap(data=par, annot=use_annot, linewidths=float(use_annot), cmap='Blues_r', square=True, fmt='.2f',
                yticklabels=False, xticklabels=False)
