#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import dianna
from dianna.visualization.image import plot_image
from dianna.visualization.timeseries import plot_timeseries

model = load_model('model/20190416freq_time.hdf5')


def run_model(data):
    # add extra "colour channels" axis at the end and run through model,
    # output is [p_noise, p_frb]
    return model.predict(data[..., None])


if __name__ == '__main__':
    data_frb = np.load('data/processed/FRB211024.npy')

    # run image-based dianna for class FRB
    heatmaps = dianna.explain_timeseries(run_model, data_frb, method='RISE', labels=[0, 1],
                                         n_masks=4096, p_keep=0.5, feature_res=8,
                                         batch_size=1024, mask_type=lambda _: 0)

    # plot heatmap for FRB class
    print(f'{heatmaps.shape=}')
    # still need to figure out timeseries viz
    plot_image(heatmaps[1], show_plot=False)
    plt.title('heatmap for class FRB')

    # plt.figure()
    # plt.imshow(data_frb, origin='lower', aspect='auto')
    # plt.xlabel('Time')
    # plt.ylabel('Freq')
    # plt.title('Input data')
    plt.show()
