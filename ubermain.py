from multitasking import Argument, create_tasks_from_arguments, run_settings


def ubermain(n_runs):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6
    """
    args = []
    args.append(Argument('experiment', ['Rebuttal/BPTT/Lorenz/annealing/longSL']))
    args.append(Argument('data_path', ['datasets/Lorenz/lorenz_data_chaos.npy'],add_to_name_as="dataset"))
    #args.append(Argument('data_path', ['datasets/duffing_data_chaos.npy'],add_to_name_as="dataset"))

    args.append(Argument('model', ["LSTM"], add_to_name_as='Model_'))
    args.append(Argument('latent_model', ["PLRNN"], add_to_name_as='_'))

    args.append(Argument('dim_z', [30], add_to_name_as='z'))
    #args.append(Argument('gradient_clipping', [1,10,100,1000], add_to_name_as='gc_'))
    #args.append(Argument('windowing', [0], add_to_name_as='WindowOn'))
    args.append(Argument('random', [0], add_to_name_as='RandomOn'))
    #args.append(Argument('deltaTau', [50,70,100], add_to_name_as='dTau'))
    #n_interleave := forcing/learning interval tau
    args.append(Argument('n_interleave', [46], add_to_name_as='Gamma'))
    args.append(Argument('seq_len', [1000], add_to_name_as='seqLen'))

    #args.append(Argument('layer_norm',[False],add_to_name_as="L_norm"))
    args.append(Argument('n_epochs', [3000]))

    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


if __name__ == '__main__':
    n_runs =3 # 5
    n_cpu = 3 # 45
    args = ubermain(n_runs)
    run_settings(create_tasks_from_arguments(args), n_cpu)
