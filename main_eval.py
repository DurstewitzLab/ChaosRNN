import os
import numpy as np
import torch as tc
import pandas as pd
from glob import glob

import utils
from evaluation import mse
from evaluation import klx
from evaluation import klz
from bptt import models
from evaluation.pse import power_spectrum_error, power_spectrum_error_per_dim

DATA_GENERATED = None
PRINT = True

'''Code to evaluate multiple models. See __main__ at bottom.'''

def get_generated_data(model,data):
    """
    Use global variable as a way to draw trajectories only once for evaluating several metrics, for speed.
    """
    global DATA_GENERATED

    data = data
    n_repeat = 1
    ts_len = int(100000/n_repeat)
    X_= []

    X, Z = model.generate_free_trajectory(data, ts_len,None,n_repeat)
    ''' linear regression for 2D Lorenz'''
    # Z = Z.detach()
    # try:
    #     B3D = np.linalg.lstsq(Z[0:10], data[0:10])[0]
    #     DATA_GENERATED = Z @ B3D
    #     print(DATA_GENERATED.shape)
    # except np.linalg.LinAlgError:
    #     print("Error")
    #     DATA_GENERATED = tc.ones((100000,3))*float('nan')

    DATA_GENERATED = X.detach()
    return DATA_GENERATED


def printf(x):
    if PRINT:
        print(x)


class Evaluator(object):
    def __init__(self, init_data):
        model_ids, data, save_path = init_data
        self.model_ids = model_ids
        self.save_path = save_path

        self.data = tc.tensor(data)

        self.name = NotImplementedError
        self.dataframe_columns = NotImplementedError

    def metric(self, model):
        return NotImplementedError

    def evaluate_metric(self):
        metric_dict = dict()
        assert self.model_ids is not None
        for model_id in self.model_ids:
            model = self.load_model(model_id)
            metric_dict[model_id] = self.metric(model)
        self.save_dict(metric_dict)

    def load_model(self, model_id):
        model = models.Model()
        model.init_from_model_path(model_id)
        model.eval()
        return model

    def save_dict(self, metric_dict):
        df = pd.DataFrame.from_dict(data=metric_dict, orient='index')
        df.columns = self.dataframe_columns
        utils.make_dir(self.save_path)
        df.to_csv('{}/{}.csv'.format(self.save_path, self.name), sep='\t')


class EvaluateKLx(Evaluator):
    def __init__(self, init_data):
        super(EvaluateKLx, self).__init__(init_data)
        self.name = 'klx'
        self.dataframe_columns = ('klx',)
    def metric(self, model):
        data_gen = get_generated_data(model,tc.tensor(self.data).clone().detach())

        klx_value = klx.klx_metric(x_gen=data_gen, x_true=tc.tensor(self.data),n_bins=10)
        printf('\tKLx {}'.format(klx_value))
        return [np.array(klx_value)]





class EvaluateMSE(Evaluator):
    def __init__(self, init_data):
        super(EvaluateMSE, self).__init__(init_data)
        self.name = 'mse'
        self.n_steps = 25
        self.dataframe_columns = tuple(['{}'.format(i) for i in range(1, 1 + self.n_steps)])

    def metric(self, model):
        mse_results = mse.n_steps_ahead_pred_mse(model, self.data, n_steps=self.n_steps)
        for step in [1, 5, 25]:
            printf('\tMSE-{} {}'.format(step, mse_results[step-1]))
        return mse_results

class EvaluateKLz(Evaluator):
    '''Calculates D_stsp - GMM, which in this code is referred to as klz'''
    def __init__(self, init_data):
        super(EvaluateKLz, self).__init__(init_data)
        self.name = 'klz'
        self.dataframe_columns = ('klz_mc',)

    def metric(self, model):
        klz_mc = klz.calc_kl_with_unit_covariance(model, self.data)
        klz_mc = float(klz_mc.detach().numpy())
        printf('\tKLz mc {}'.format(klz_mc))
        return [klz_mc]

class EvaluatePSE(Evaluator):
    '''Calculates the Hellinger distance D_H, which in this code is referred to as pse'''

    def __init__(self, init_data):
        super(EvaluatePSE, self).__init__(init_data)
        self.name = 'pse'
        n_dim = self.data.shape[1]
        self.dataframe_columns = tuple(['mean'] + ['dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        data_gen = get_generated_data(model,tc.tensor(self.data))
        n_repeat = 50
        time_steps =int(100000/n_repeat)#10000 ##2000
        data = self.data
        dim_x = data_gen.shape[1]
        x_gen = tc.reshape(data_gen, shape=(n_repeat, time_steps, dim_x))
        n_batches =int(len(data)/time_steps)#n_repeat

        ts_len = int(tc.tensor(data).shape[0]/n_repeat)
        print(ts_len)
        x_gen = tc.reshape(data_gen, shape=(n_repeat, time_steps, dim_x))[:,:ts_len]
        x_true = tc.reshape(tc.tensor(data), shape=(n_repeat, ts_len, dim_x))


        pse_per_dim = power_spectrum_error_per_dim(x_gen=x_gen, x_true=x_true)
        pse = power_spectrum_error(x_gen=x_gen, x_true=x_true)

        printf('\tPSE {}'.format(pse))
        printf('\tPSE per dim {}'.format(pse_per_dim))
        return [pse] + pse_per_dim


class SaveArgs(Evaluator):
    def __init__(self, init_data):
        super(SaveArgs, self).__init__(init_data)
        self.name = 'args'
        self.dataframe_columns = ('dim_x', 'dim_z', 'n_bases')

    def metric(self, model):
        args = model.args
        return [args['dim_x'], args['dim_z'], args['n_bases']]


def gather_eval_results(eval_dir='save', save_path='save_eval', metrics=None):
    """Pre-calculated metrics in individual model directories are gathered in one csv file"""
    if metrics is None:
        metrics = ['klx', 'pse']
    metrics.append('args')
    model_ids = get_model_ids(eval_dir)
    for metric in metrics:
        paths = [os.path.join(model_id, '{}.csv'.format(metric)) for model_id in model_ids]
        data_frames = []
        for path in paths:
            try:
                data_frames.append(pd.read_csv(path, sep='\t', index_col=0))
            except:
                print('Warning: Missing model at path: {}'.format(path))
        data_gathered = pd.concat(data_frames)
        utils.make_dir(save_path)
        metric_save_path = '{}/{}.csv'.format(save_path, metric)
        data_gathered.to_csv(metric_save_path, sep='\t')


def choose_evaluator_from_metric(metric_name, init_data):
    if metric_name == 'mse':
        EvaluateMetric = EvaluateMSE(init_data)
    elif metric_name == 'klx':
        EvaluateMetric = EvaluateKLx(init_data)
    elif metric_name == 'pse':
        EvaluateMetric = EvaluatePSE(init_data)
    elif metric_name == 'klz':
        EvaluateMetric = EvaluateKLz(init_data)

    else:
        raise NotImplementedError
    return EvaluateMetric


def eval_model_on_data_with_metric(model, data, metric):
    init_data = (None, data, None)
    EvaluateMetric = choose_evaluator_from_metric(metric, init_data)
    EvaluateMetric.data = data
    metric_value = EvaluateMetric.metric(model)
    return metric_value


def is_model_id(path):
    """Check if path ends with a three digit, e.g. save/test/001 """
    run_nr = path.split('/')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers


def get_model_ids(path):
    """
    Get model ids from a directory by recursively searching all subdirectories for files ending with a number
    """
    assert os.path.exists(path)
    if is_model_id(path):
        model_ids = [path]
    else:
        all_subfolders = glob(os.path.join(path, '**/*'), recursive=True)
        model_ids = [file for file in all_subfolders if is_model_id(file)]
    assert model_ids, 'could not load from path: {}'.format(path)
    return model_ids


def eval_model(args):
    save_path = args.load_model_path
    evaluate_model_path(args, model_path=save_path, metrics=args.metrics)


def evaluate_model_path(data_path, model_path=None, metrics=None):
    """Evaluate a single model in directory model_path w.r.t. metrics and save results in csv file in model_path"""
    model_ids = [model_path]
    data = tc.tensor(utils.read_data(data_path))
    init_data = (model_ids, data, model_path)
    Save = SaveArgs(init_data)
    Save.evaluate_metric()
    global DATA_GENERATED
    DATA_GENERATED = None

    for metric_name in metrics:
        EvaluateMetric = choose_evaluator_from_metric(metric_name=metric_name, init_data=(model_ids, data, model_path))
        EvaluateMetric.evaluate_metric()


def evaluate_all_models(eval_dir, data_path, metrics):
    model_paths = get_model_ids(eval_dir)
    n_models = len(model_paths)
    print('Evaluating {} models'.format(n_models))
    for i, model_path in enumerate(model_paths):
        print('{} of {}'.format(i+1, n_models))
        try:
            evaluate_model_path(data_path=data_path, model_path=model_path, metrics=metrics)

        except FileNotFoundError:
            print('Error in model evaluation {}'.format(model_path))
    return


if __name__ == '__main__':
    '''
    Evaluates multiple models.
    data_path: path to original data
    eval_dir: path to folder containing the models to be evaluated
    '''
    data_path = 'datasets/Lorenz/lorenz_data_chaos.npy'

    eval_dir = 'results/Rebuttal/Windowing/Lorenz'
    eval_dir = 'results/Rebuttal/Random/Lorenz'
    eval_dir = 'results/Rebuttal/BPTT/Lorenz/annealing/longSL/Lorenz'
    eval_dir = 'results/Rebuttal/BPTT/Lorenz/TruncBPTT/Lorenz/'

    #eval_dir = '/Users/jonas/Master/Theo_Neuroscience/ZI/BPTT/BPTT_training/results/Rebuttal/GradClip/seqLen1000/maxClip/Lorenz'

    ''' Metrics:
    pse := D_H
    KLx : D_stsp - binning | use for low-dimensional data (N<5)
    KLz := D_stsp - GMM  | use for high-dimensional data (N>)
    '''
    metrics = ['klx'] #choose from klx, pse, klz
    evaluate_all_models(eval_dir=eval_dir, data_path=data_path, metrics=metrics)
    gather_eval_results(eval_dir=eval_dir, metrics=metrics,save_path=eval_dir+"/eval/")
