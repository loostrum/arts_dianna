#!/usr/bin/env python3
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

# model parameters
NFREQ = 32
NTIME = 64


def preprocess_data(data):
    nfreq_data, ntime_data = data.shape
    assert nfreq_data % NFREQ == 0
    # reshape nfreq axis
    data = data.reshape(NFREQ, -1, ntime_data).mean(axis=1)
    # reshape time axis
    factor, remainder = divmod(ntime_data, NTIME)
    # remove remainder from start/end to keep center the same
    start = remainder//2
    end = ntime_data - remainder//2
    if (remainder%2 != 0):
        start += 1
    data = data[:, start:end]
    data = data.reshape(NFREQ, NTIME, -1).mean(axis=2)
    # normalize per channel
    data -= np.median(data, axis=-1)[:, None]
    with np.errstate(invalid='ignore'):
        data /= np.std(data, axis=-1)[:, None]
    data[np.isnan(data)] = 0.
    return data


if __name__ == '__main__':
    files = glob.glob('data/raw/*.hdf5')

    for input_file in files:
        with h5py.File(input_file) as fp:
            raw_data = fp['data_freq_time'][...]
        preprocessed_data = preprocess_data(raw_data)
        output_file = input_file.replace('raw', 'processed').replace('hdf5', 'npy')
        np.save(output_file, preprocessed_data)

        # make figure
        if True:
            fig, ax = plt.subplots()
            plt.imshow(preprocessed_data, origin='lower', aspect='auto', cmap='viridis')
            plt.savefig(output_file.replace('.npy', '.jpg'))
