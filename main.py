#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import dianna
from dianna.visualization.image import plot_image
from dianna.visualization.timeseries import plot_timeseries

from utils import run_model, load_train_data


if __name__ == '__main__':
    # data_frb = np.load('data/processed/FRB211024.npy')
    # data_frb = load_train_data(33598)  # FRB plus a few RFI channels
    data_frb = load_train_data(36918)  # narrow-bandwidth FRB plus a few RFI channels (but masked by median filter)

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
