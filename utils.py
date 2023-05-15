
import h5py
from tensorflow.keras.models import load_model

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