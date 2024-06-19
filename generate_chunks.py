import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os import makedirs

def generate_chunks(df, chunk_size, chunk_stride, cols):

    from numpy.lib.stride_tricks import sliding_window_view

    gaps = list((g := df.timestamp.diff().gt(pd.Timedelta(minutes=1)))[g].index)
    c = []
    window_start_date = []
    start = 0
    for gap in gaps:
        tdf = df.iloc[start:gap, :]
        if len(tdf) < chunk_size:
            start = gap
            continue
        vals = tdf[cols].values
        sliding_vals = sliding_window_view(vals, (chunk_size, len(cols))).squeeze(1)[::chunk_stride, :, :]
        window_start_date.append(sliding_window_view(tdf.timestamp.values, chunk_size)[::chunk_stride,[0,-1]])
        c.append(sliding_vals)
        start = gap
    tdf = df.iloc[start:, :]
    if len(tdf) >= chunk_size:
        vals = tdf[cols].values
        sliding_vals = sliding_window_view(vals, (chunk_size, len(cols))).squeeze(1)[::chunk_stride, :, :]
        c.append(sliding_vals)
        window_start_date.append(sliding_window_view(tdf.timestamp.values, chunk_size)[::chunk_stride,[0,-1]])

    c = np.concatenate(c)
    return c, np.concatenate(window_start_date)

def load_data():
    with open("data/train_chunk_dates.pkl", "rb") as chunk_dates_file:
        train_chunk_dates = pkl.load(chunk_dates_file)
    with open("data/train_chunks.pkl", "rb") as chunk_dates_file:
        train_chunks = pkl.load(chunk_dates_file).astype(np.float32)
    with open("data/test_chunk_dates.pkl", "rb") as chunk_dates_file:
        test_chunk_dates = pkl.load(chunk_dates_file)
    with open("data/test_chunks.pkl", "rb") as chunk_dates_file:
        test_chunks = pkl.load(chunk_dates_file).astype(np.float32)

    return train_chunks, train_chunk_dates, test_chunks, test_chunk_dates

def generate_data():
    print("Read dataset")

    final_metro = pd.read_csv('MetroPT2.csv')
    final_metro['timestamp'] = pd.to_datetime(final_metro['timestamp'])
    final_metro = final_metro.sort_values('timestamp')
    final_metro.reset_index(drop=True, inplace=True)

    analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
                    'Oil_temperature', 'Flowmeter', 'Motor_current']

    print('Separated into training and test')

    cutoff_date = np.datetime64('2022-06-01T00:00:00.000000000')
    metro_train = final_metro[final_metro['timestamp'] < cutoff_date]
    metro_test = final_metro[final_metro['timestamp'] >= cutoff_date]

    print("Calculate chunks")

    train_chunks, train_chunk_dates = generate_chunks(metro_train, 1800, 60, analog_sensors)
    test_chunks, test_chunk_dates = generate_chunks(metro_test, 1800, 5*60, analog_sensors)

    scaler = StandardScaler()
    train_chunks = np.array(list(map(lambda x: scaler.fit_transform(x), train_chunks)))
    test_chunks = np.array(list(map(lambda x: scaler.fit_transform(x), test_chunks)))

    print("Finished scaling")

    makedirs('data', exist_ok=True)

    with open("data/train_chunk_dates.pkl", "wb") as pklfile:
        pkl.dump(train_chunk_dates, pklfile)

    with open("data/test_chunk_dates.pkl", "wb") as pklfile:
        pkl.dump(test_chunk_dates, pklfile)

    with open("data/train_chunks.pkl", "wb") as pklfile:
        pkl.dump(train_chunks, pklfile)

    with open("data/test_chunks.pkl", "wb") as pklfile:
        pkl.dump(test_chunks, pklfile)

    print("Finished saving")

if __name__ == '__main__':
    generate_data()