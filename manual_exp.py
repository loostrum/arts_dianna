#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import h5py

from preprocess import preprocess_data

model = load_model('model/20190416freq_time.hdf5')


def load_train_data(ind):
    with h5py.File('data/data_apertif.hdf5') as f:
        data = f['data_freq_time'][ind]
        label = f['labels'][ind]
        print(f'Label for train data at index {ind}: {label}')
    return preprocess_data(data)


def run_model(data):
    # add extra "colour channels" axis at the end and run through model,
    # output is [p_noise, p_frb]
    return model.predict(data[..., None])


def create_masked_data(data_frb, n):
    nfreq, ntime = data_frb.shape

    # mask sets of n consecutive timesteps
    nmask = ntime - n + 1
    masked = np.zeros((nmask, nfreq, ntime))
    for t in range(0, nmask):
        masked[t] = data_frb.copy()
        masked[t][..., t:t+n] = 0.

    return masked


if __name__ == '__main__':
    # data_frb = np.load('data/processed/FRB211024.npy')
    data_frb = load_train_data(33598)  # FRB plus a few RFI channels

    n_step_to_mask = 4
    masked = create_masked_data(data_frb, n_step_to_mask)

    p_frb = run_model(masked)[:, 1]
    p_frb_original = run_model(data_frb[None, ...])[0, 1]
    print(f'{p_frb_original=:.2f}')
    print(p_frb)

    exp = p_frb_original - p_frb
    norm = plt.Normalize(p_frb_original - 1, p_frb_original)
    cmap = plt.get_cmap('viridis')

    # plot input data
    fig, ax = plt.subplots()
    ax.imshow(data_frb, origin='lower', aspect='auto')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Input data')

    # plot FRB data as timeseries
    fig, ax = plt.subplots()
    ax.plot(data_frb.sum(axis=0), c='k')
    for idx, weight in enumerate(exp):
        start = idx
        stop = start + n_step_to_mask
        c = cmap(norm(weight))
        ax.axvspan(start, stop, color=c, alpha=.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Brightness')
    ax.set_xlim(0, data_frb.shape[1])

    plt.show()
