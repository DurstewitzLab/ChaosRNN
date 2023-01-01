
import numpy as np
'''Calculation of D_H, to evaluate temporal similarity between true and simulated data'''

SMOOTHING_SIGMA = 2 #choose depending on data
FREQUENCY_CUTOFF = 500

def convert_to_decibel(x):
    x = 20 * np.log10(x)
    return x[0]


def ensure_length_is_even(x):
    n = len(x)
    if n % 2 != 0:
        x = x[:-1]
        n = len(x)
    x = np.reshape(x, (n,))
    return x


def fft_smoothed(x):
    """
    :param x: input signal
    :return fft: smoothed power spectrum
    """
    x = ensure_length_is_even(x)
    fft_real = np.fft.rfft(x,norm='ortho')
    fft_magnitude = np.abs(fft_real)**2 * 2 / len(x)
    fft_smoothed = kernel_smoothen(fft_magnitude, kernel_sigma=SMOOTHING_SIGMA)

    return fft_smoothed

def get_average_spectrum(trajectories):
    '''average trajectories to fulfill conditions for the application
    of the Hellinger distance '''
    spectrum = []
    for trajectory in trajectories:
        trajectory = (trajectory-trajectory.mean())/trajectory.std()
        fft= fft_smoothed(trajectory)
        spectrum.append(fft)
    spectrum = np.nanmean(np.array(spectrum),axis=0)

    return spectrum


def power_spectrum_error_per_dim(x_gen, x_true):
    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]
    dim_x = x_gen.shape[2]
    pse_corrs_per_dim = []
    for dim in range(dim_x):
        spectrum_true = get_average_spectrum(x_true[:, :, dim])
        spectrum_gen = get_average_spectrum(x_gen[:, :, dim])
        spectrum_true = spectrum_true[:FREQUENCY_CUTOFF]
        spectrum_gen = spectrum_gen[:FREQUENCY_CUTOFF]
        BC = np.trapz(np.sqrt(spectrum_true*spectrum_gen))
        hellinger_dist = np.sqrt(1-BC)

        pse_corrs_per_dim.append(hellinger_dist)

    return pse_corrs_per_dim


def power_spectrum_error(x_gen, x_true):
    pse_errors_per_dim = power_spectrum_error_per_dim(x_gen, x_true)
    return np.array(pse_errors_per_dim).mean(axis=0)


def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothen data with Gaussian kernel
    @param kernel_sigma: standard deviation of gaussian, kernel_size is adapted to that
    @return: internal data is modified but nothing returned
    """
    kernel = get_kernel(kernel_sigma)
    data_final = data.copy()
    data_conv = np.convolve(data[:], kernel)
    pad = int(len(kernel) / 2)
    data_final[:] = data_conv[pad:-pad]
    data = data_final
    return data


def gauss(x, sigma=1):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-1 / 2 * (x / sigma) ** 2)


def get_kernel(sigma):
    size = sigma * 10 + 1
    kernel = list(range(size))
    kernel = [float(k) - int(size / 2) for k in kernel]
    kernel = [gauss(k, sigma) for k in kernel]
    kernel = [k / np.sum(kernel) for k in kernel]
    return kernel


