#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import dianna
from dianna.visualization.image import plot_image
from dianna.visualization.timeseries import plot_timeseries

from utils import run_model, load_train_data


if __name__ == '__main__':
    data_frb = np.load('data/processed/FRB211024.npy')
    # data_frb = load_train_data(33598)  # FRB plus a few RFI channels
    # data_frb = load_train_data(36918)  # narrow-bandwidth FRB plus a few RFI channels (but masked by median filter)

    # run image-based dianna for class FRB
    heatmaps = dianna.explain_timeseries(run_model, data_frb, method='RISE', labels=[0, 1],
                                         n_masks=4096, p_keep=.05, feature_res=8,
                                         batch_size=1024, mask_type=lambda _: 0)
    # select heatmap for class FRB
    heatmap = heatmaps[1]

    navg = 4
    segments = []

    assert heatmap.shape[0] % navg == 0
    heatmap_ds = heatmap.reshape((heatmap.shape[0]//navg, navg, heatmap.shape[1])).mean(axis=1)
    data_frb_ds = data_frb.reshape((heatmap.shape[0]//navg, navg, heatmap.shape[1])).mean(axis=1)

    for channel_number, heatmap_channel in enumerate(heatmap_ds):
        for i in range(len(heatmap_channel) - 1):
            segments.append({
                'index': i,
                'start': i,
                'stop': i + 1,
                'weight': heatmap_channel[i],
                'channel': channel_number})

    x_label = 'Time step'
    df = 300. / heatmap_ds.shape[0]
    flo = 1220 + .5*df
    y_label = []
    for channel_number in range(0, heatmap_ds.shape[0]):
        y_label.append(f'{flo+df*channel_number:.0f}')

    plot_timeseries(range(heatmap.shape[1]), data_frb_ds, segments,
                    x_label=x_label, y_label=y_label, show_plot=False)

    fig = plt.gcf()
    fig.suptitle('Explanation for FRB 211024')

    # plot input data
    # plt.figure()
    # plt.imshow(data_frb, origin='lower')
    # plt.xlabel('Time')
    # plt.ylabel('Freq')
    # plt.title('Input data')
    plt.show()
