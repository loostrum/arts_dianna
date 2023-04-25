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


def create_masked_data_time(data_frb, n):
    nfreq, ntime = data_frb.shape

    # mask sets of n consecutive timesteps
    nmask = ntime - n + 1
    masked = np.zeros((nmask, nfreq, ntime))
    for t in range(0, nmask):
        masked[t] = data_frb.copy()
        masked[t][..., t:t+n] = 0.

    return masked


def create_masked_data_freq(data_frb, n):
    nfreq, ntime = data_frb.shape

    # mask sets of n consecutive channels
    nmask = nfreq - n + 1
    masked = np.zeros((nmask, nfreq, ntime))
    for t in range(0, nmask):
        masked[t] = data_frb.copy()
        masked[t][t:t+n, ...] = 0.

    return masked


if __name__ == '__main__':
    # data_frb = np.load('data/processed/FRB211024.npy')
    # data_frb = load_train_data(33598)  # FRB plus a few RFI channels
    data_frb = load_train_data(36918)  # narrow-bandwidth FRB plus a few RFI channels (but masked by median filter)

    p_frb_original = run_model(data_frb[None, ...])[0, 1]
    print(f'{p_frb_original=:.2f}')
    norm = plt.Normalize(p_frb_original - 1, p_frb_original)
    cmap = plt.get_cmap('viridis')

    n_time_to_mask = 4
    n_freq_to_mask = 8

    # Time-based masking
    print("Time-based:")
    masked_time = create_masked_data_time(data_frb, n_time_to_mask)
    p_frb_time = run_model(masked_time)[:, 1]
    print(p_frb_time)
    exp_time = p_frb_original - p_frb_time

    # Freq-based masking
    print("Freq-based:")
    masked_freq = create_masked_data_freq(data_frb, n_freq_to_mask)
    p_frb_freq = run_model(masked_freq)[:, 1]
    print(p_frb_freq)
    exp_freq = p_frb_original - p_frb_freq

    # plot input data
    fig, ax = plt.subplots()
    ax.imshow(data_frb, origin='lower', aspect='auto')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Input data')

    fig, ax = plt.subplots()
    # freq-based masking
    for idx, weight in enumerate(exp_freq):
        start = idx
        stop = start + n_freq_to_mask
        c = cmap(norm(weight))
        ax.axhspan(start, stop, color=c, alpha=.5)
    ax.set_xlim(0, data_frb.shape[1])
    ax.set_ylim(0, data_frb.shape[0])
    ax.set_title('Frequency-based explanation')

    # plot FRB data as timeseries and add time-based explanation
    fig, ax = plt.subplots()
    ax.plot(data_frb.sum(axis=0), c='k')
    # time-based masking
    for idx, weight in enumerate(exp_time):
        start = idx
        stop = start + n_time_to_mask
        c = cmap(norm(weight))
        ax.axvspan(start, stop, color=c, alpha=.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Brightness')
    ax.set_title('Timeseries + explanation')
    ax.set_xlim(0, data_frb.shape[1])

    plt.show()
