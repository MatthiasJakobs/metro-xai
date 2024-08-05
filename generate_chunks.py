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

def load_data(version=2, scaler=1):
    with open(f"data/pt{version}_train_chunk_dates.pkl", "rb") as chunk_dates_file:
        train_chunk_dates = pkl.load(chunk_dates_file)
    with open(f"data/pt{version}_test_chunk_dates.pkl", "rb") as chunk_dates_file:
        test_chunk_dates = pkl.load(chunk_dates_file)
    if scaler == 1:
        with open(f"data/pt{version}_train_chunks.pkl", "rb") as chunk_dates_file:
            train_chunks = pkl.load(chunk_dates_file).astype(np.float32)
        with open(f"data/pt{version}_test_chunks.pkl", "rb") as chunk_dates_file:
            test_chunks = pkl.load(chunk_dates_file).astype(np.float32)
    else:
        with open(f"data/pt{version}_train_chunks_2.pkl", "rb") as chunk_dates_file:
            train_chunks = pkl.load(chunk_dates_file).astype(np.float32)
        with open(f"data/pt{version}_test_chunks_2.pkl", "rb") as chunk_dates_file:
            test_chunks = pkl.load(chunk_dates_file).astype(np.float32)

    return train_chunks, train_chunk_dates, test_chunks, test_chunk_dates

def generate_data(version=2):
    print(f'=== MetroPT{version} ===')
    print("Read dataset")

    final_metro = pd.read_csv(f'MetroPT{version}.csv')
    final_metro['timestamp'] = pd.to_datetime(final_metro['timestamp'])
    final_metro = final_metro.sort_values('timestamp')
    final_metro.reset_index(drop=True, inplace=True)

    if version == 2:
        analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
                        'Oil_temperature', 'Flowmeter', 'Motor_current']
        additional_sensors = ['COMP']
        cutoff_date = np.datetime64('2022-06-01T00:00:00.000000000')
    else:
        analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
        additional_sensors = ['COMP', 'DV_eletric', 'Towers', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
        cutoff_date = np.datetime64('2020-04-01T00:00:00.000000000')

    print('Separated into training and test')

    metro_train = final_metro[final_metro['timestamp'] < cutoff_date]
    metro_test = final_metro[final_metro['timestamp'] >= cutoff_date]

    print("Calculate chunks")

    train_chunks, train_chunk_dates = generate_chunks(metro_train, 1800, 60, analog_sensors+additional_sensors)
    test_chunks, test_chunk_dates = generate_chunks(metro_test, 1800, 5*60, analog_sensors+additional_sensors)

    makedirs('data', exist_ok=True)

    with open(f"data/pt{version}_train_chunks_unnormalized.pkl", "wb") as pklfile:
        pkl.dump(train_chunks, pklfile)

    with open(f"data/pt{version}_test_chunks_unnormalized.pkl", "wb") as pklfile:
        pkl.dump(test_chunks, pklfile)

    scaler = StandardScaler()
    sc1_train_chunks = np.array(list(map(lambda x: scaler.fit_transform(x), train_chunks)))
    sc1_test_chunks = np.array(list(map(lambda x: scaler.fit_transform(x), test_chunks)))

    n_channels = train_chunks.shape[-1]
    scaler.fit(train_chunks[::30].reshape(-1, n_channels))
    sc2_train_chunks = np.array(list(map(lambda x: scaler.transform(x), train_chunks)))
    sc2_test_chunks = np.array(list(map(lambda x: scaler.transform(x), test_chunks)))

    print("Finished scaling")

    with open(f"data/pt{version}_train_chunk_dates.pkl", "wb") as pklfile:
        pkl.dump(train_chunk_dates, pklfile)

    with open(f"data/pt{version}_test_chunk_dates.pkl", "wb") as pklfile:
        pkl.dump(test_chunk_dates, pklfile)

    with open(f"data/pt{version}_train_chunks.pkl", "wb") as pklfile:
        pkl.dump(sc1_train_chunks, pklfile)

    with open(f"data/pt{version}_test_chunks.pkl", "wb") as pklfile:
        pkl.dump(sc1_test_chunks, pklfile)

    with open(f"data/pt{version}_train_chunks_2.pkl", "wb") as pklfile:
        pkl.dump(sc2_train_chunks, pklfile)

    with open(f"data/pt{version}_test_chunks_2.pkl", "wb") as pklfile:
        pkl.dump(sc2_test_chunks, pklfile)

    print("Finished saving")

if __name__ == '__main__':
    generate_data(version=2)
    generate_data(version=3)
    load_data(version=2)
    load_data(version=3)