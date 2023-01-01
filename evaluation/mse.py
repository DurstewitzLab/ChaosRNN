import torch as tc


def get_input(inputs, step, n_steps):
    if inputs is not None:
        input_for_step = inputs[step:(-n_steps + step), :]
    else:
        input_for_step = None
    return input_for_step


def get_ahead_pred_obs(model, data, n_steps, inputs):
    B = model.output_layer.weight

    time_steps = len(data) - n_steps
    x_data = data[:-n_steps, :]
    #z = tc.randn((len(x_data),model.latent_model.d_z))
    #z[:,:model.latent_model.d_x] = x_data
    B = model.output_layer.weight
    z = (tc.pinverse(B) @ x_data.float().T).T
    #z = ((tc.pinverse(B @ B.T) @ B).T @ x_data.float().T).T
    x_pred = []
    b = x_data.shape[0]
    for step in range(n_steps):
        if model.args['model'] == 'LSTM':

            c = tc.zeros((1, b, model.latent_model.d_z))
            h = z.reshape((1, -1, model.latent_model.d_z))
            inp = tc.zeros(b, 1, model.latent_model.d_x)
            output, (h, c) = model.latent_model.latent_step(inp, (h, c))
            z = output.squeeze(0)
        elif model.args['model'] == 'RNN':

            h = z.reshape((1, -1, model.latent_model.d_z))
            inp = tc.zeros(b, 1, model.latent_model.d_x)
            output, h  = model.latent_model.latent_step(inp,h)
            z = output.squeeze(0)
        elif model.args['model'] =="PLRNN":
            z = model.latent_model.latent_step(z)
        x_pred.append(model.output_layer(z))
    x_pred = tc.cat(x_pred)
    x_pred = tc.reshape(x_pred, shape=(n_steps, time_steps, -1))
    return x_pred


def construct_ground_truth(data, n_steps):
    time_steps = len(data) - n_steps
    x_true = [data[step:, :] for step in range(1, n_steps + 1)]
    x_true = tc.stack([x[:time_steps, :] for x in x_true])
    return x_true


def squared_error(x_pred, x_true):
    return tc.pow(x_pred - x_true, 2)


def n_steps_ahead_pred_mse(model, data, n_steps, inputs=None):
    with tc.no_grad():
        x_pred = get_ahead_pred_obs(model, data, n_steps, inputs)
        x_true = construct_ground_truth(data, n_steps)
        import matplotlib.pyplot as plt
        plt.plot(x_pred[5,:,0],'b-')
        plt.plot(x_true[5,:,0],'g-')
        plt.show()
        mean_squared_error = squared_error(x_pred, x_true).mean([1, 2]).numpy()
    return mean_squared_error
