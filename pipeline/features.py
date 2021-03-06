from os import mkdir
from os.path import exists, join
import itertools
from functools import partial

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import pearsonr, differential_entropy

from tqdm import tqdm

from mne.connectivity import spectral_connectivity, phase_slope_index


def get_func_by_method(method_name):
    methods = {
        'coh': partial(get_mne_spec_con_feats, band=None, method='coh'),
        'coh-alpha': partial(get_mne_spec_con_feats, band='alpha', method='coh'),
        'coh-beta': partial(get_mne_spec_con_feats, band='beta', method='coh'),
        'coh-theta': partial(get_mne_spec_con_feats, band='theta', method='coh'),
        'env': partial(get_envelope_feats, band=None),
        'env-alpha': partial(get_envelope_feats, band='alpha'),
        'env-beta': partial(get_envelope_feats, band='beta'),
        'env-theta': partial(get_envelope_feats, band='theta'),
        'bands': get_bands_feats,
        'psi': get_psi_feats,
        'fr-assym': get_frontal_asymmetry,
        'asMap': get_asMap
    }
    if method_name in methods:
        return methods[method_name]
    else:
        raise ValueError('Features method is not in allowed list')


def calc_features_dict(df, method_names):
    d = {}
    for method_name in method_names:
        f = get_func_by_method(method_name)
        d.update(f(df))
    return d


band_bounds = {
    'theta' : [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'gamma': [30, 45],
    '4_6': [2, 4],
    '6_8': [2, 4],
    '8_10': [2, 4],
    '10_12': [2, 4],
}


def merge_dfs(dfs):
    res_df = None

    for i, df in enumerate(dfs):

        df = df.copy()

        if i == 0:
            res_df = df
        else:
            del df['target']
            res_df = res_df.merge(df, on='fn')

    return res_df

def get_merged_df(base_path, feature_methods):
    """
    Loads merged dataframe from csv-files for each path.
    Arguments are paths for csv-files followed by the list of feature methods.
    """
    features = [pd.read_csv(get_feature_path(method, base_path)) for method in feature_methods]
    return merge_dfs(features)

def get_col_name(method, band, ch_1, ch_2=None):
    band_name = 'nofilt' if band is None else band
    s = method + '_' + band_name + '_' + ch_1
    if ch_2:
        s += '_' + ch_2
    return s


def get_feature_path(method_name, path):
    return join(path, method_name.replace('-', '_') + '.csv')


def get_filter(sfreq=125., band='alpha'):

    f_low_lb = band_bounds[band][0] - 1
    f_low_ub = band_bounds[band][0]
    f_high_lb = band_bounds[band][1]
    f_high_ub = band_bounds[band][1] + 1

    nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate

    freq = [0., f_low_lb, f_low_ub, f_high_lb, f_high_ub, nyq]
    gain = [0, 0, 1, 1, 0, 0]
    n = int(round(1 * sfreq)) + 1

    filt = signal.firwin2(n, freq, gain, nyq=nyq)

    return filt

def get_frontal_asymmetry(df, bands=('alpha', 'beta', 'gamma', 'theta'), sfreq=125., nperseg=1024):
    feats=get_bands_feats(df, bands=bands, sfreq=sfreq, nperseg=nperseg)
    frontal_assymetry ={}
    for band in bands:
        f3 =feats[get_col_name('bands', band, "F3")]
        f4 =feats[get_col_name('bands', band, "F4")]
        frontal_assymetry[get_col_name('bands', band, "F3_f4")] =(f3-f4)/(f3+f4)
    return frontal_assymetry

def get_asMap(df, bands=('alpha', 'beta', 'gamma', 'theta')):
    feats=get_bands_entropy(df, bands=bands)
    frontal_assymetry ={}
    channels = df.columns
    for band in bands:
        for channel1 in channels:
            for i in range(0, int(len(channels)/2)+1):
                channel2= channels[i]
                if channel1!=channel2:
                    c1 =feats[get_col_name('bands', band, channel1)]
                    c2 =feats[get_col_name('bands', band, channel2)]
                    frontal_assymetry[get_col_name('bands', band, f"{channel1}_{channel2}")] =(c1-c2)/(c1+c2)

    return frontal_assymetry

def get_bands_entropy(df, bands=('alpha', 'beta', 'gamma', 'theta'), sfreq=125., nperseg=1024):
    channels = df.columns

    feats = {}

    for ch in channels:
        freqs, times,psds = signal.stft(df[ch],sfreq)
        #freqs, psds = signal.welch(df[ch], sfreq, nperseg=nperseg)
        psd_df = pd.DataFrame(data={'freqs': freqs, 'psds': np.abs(psds).tolist()})
        for band in bands:
            feats[get_col_name('bands', band, ch)]=differential_entropy(np.concatenate(psd_df.loc[
                    (psd_df['freqs'] >= band_bounds[band][0]) &
                    (psd_df['freqs'] <= band_bounds[band][1]),
                    'psds'].to_numpy()))


    return feats



def get_bands_feats(df, bands=('alpha', 'beta'), sfreq=125., nperseg=1024):

    channels = df.columns

    feats = {}

    for ch in channels:
        freqs, psds = signal.welch(df[ch], sfreq, nperseg=nperseg)
        psd_df = pd.DataFrame(data={'freqs': freqs, 'psds': psds})
        total = psd_df['psds'].sum()
        for band in bands:
            feats[get_col_name('bands', band, ch)] = (
                psd_df.loc[
                    (psd_df['freqs'] >= band_bounds[band][0]) &
                    (psd_df['freqs'] <= band_bounds[band][1]),
                    'psds']
                .sum()
                / total
            )
    return feats


def get_mne_spec_con_feats(df, sfreq=125., band=None, method='coh'):

    electrodes = df.columns

    res = spectral_connectivity(
        df[electrodes].values.T.reshape(1, len(electrodes), -1),
        method=method, sfreq=sfreq, verbose=False)

    data = res[0]
    freqs = res[1]

    def filter(arr):
        if band is None:
            return arr
        else:
            start_idx = np.where(freqs > band_bounds[band][0])[0][0]
            end_idx = np.where(freqs < band_bounds[band][1])[0][-1] + 1
            return arr[start_idx:end_idx]

    d = {}

    idx_electrodes_dict = {i: e for i, e in enumerate(electrodes)}

    for idx_1, idx_2 in itertools.combinations(range(len(electrodes)), 2):
        el_1 = idx_electrodes_dict[idx_1]
        el_2 = idx_electrodes_dict[idx_2]
        d[get_col_name(method, band, el_1, el_2)] = filter(data[idx_2, idx_1]).mean()

    return d


def get_envelope_feats(df, sfreq=125., band='alpha'):

    electrodes = df.columns

    df = df.copy()
    new_df = pd.DataFrame()
    if band is not None:
        filt = get_filter(sfreq, band)
    else:
        filt = None

    for el in electrodes:
        sig = df[el]
        if filt is not None:
            sig = np.convolve(filt, df[el], 'valid')
        sig = hilbert(sig)
        sig = np.abs(sig)
        new_df[el + '_env'] = sig

    d = {}

    idx_electrodes_dict = {i: e for i, e in enumerate(electrodes)}

    for idx_1, idx_2 in itertools.combinations(range(len(electrodes)), 2):
        el_1 = idx_electrodes_dict[idx_1]
        el_2 = idx_electrodes_dict[idx_2]
        series_1 = new_df[el_1 + '_env']
        series_2 = new_df[el_2 + '_env']
        d[get_col_name('env', band, el_1, el_2)] = pearsonr(series_1, series_2)[0]

    return d


def get_envelope_std(df, sfreq=125., band='alpha'):
    channels = df.columns

    if band is not None:
        filt = get_filter(sfreq, band)
    else:
        filt = None

    d = {}

    for ch in channels:
        sig = df[ch]
        if filt is not None:
            sig = np.convolve(filt, df[ch], 'valid')
        sig = hilbert(sig)
        sig = np.abs(sig)
        d[ch + '_env_var'] = sig.std() / sig.mean()

    return d



def get_psi_feats(df, sfreq=125., band='alpha'):

    electrodes = df.columns

    df = df.copy()
    alpha_filter = get_filter(sfreq=sfreq, band=band)

    df = df[electrodes]
    for el in electrodes:
        df[el] = np.convolve(alpha_filter, df[el], 'same')

    vals = df.values
    vals = vals.transpose(1, 0)
    vals = vals[None, :, :]

    psi, freqs, times, n_epochs, _ = phase_slope_index(vals, sfreq=sfreq, verbose=False)
    d = {}
    for i in range(psi.shape[0]):
        for j in range(i):
            d[get_col_name('psi', band, electrodes[i], electrodes[j])] = psi[i, j, 0]
    return d


# Legacy

def get_feature_build_func(method_name, verbose=None, df_filter_func=None):

    def unity_func(x):
        return x

    if df_filter_func is None:
        df_filter_func = unity_func

    methods = {
        'coh': partial(get_mne_spec_con_feats, band=None, method='coh'),
        'coh-alpha': partial(get_mne_spec_con_feats, band='alpha', method='coh'),
        'coh-beta': partial(get_mne_spec_con_feats, band='beta', method='coh'),
        'coh-theta': partial(get_mne_spec_con_feats, band='theta', method='coh'),
        'env': partial(get_envelope_feats, band=None),
        'env-alpha': partial(get_envelope_feats, band='alpha'),
        'env-beta': partial(get_envelope_feats, band='beta'),
        'env-theta': partial(get_envelope_feats, band='theta'),
        'bands': get_bands_feats,
        'psi': get_psi_feats,
        'fr-assym': get_frontal_asymmetry,
        'asMap': get_asMap
    }

    f = methods[method_name]

    def wrapped(data_path, out_path):

        print('Started features stage -', method_name)

        path_file_path = join(data_path, 'path_file.csv')
        path_df = pd.read_csv(path_file_path)
        # required columns check
        assert all([col in path_df.columns for col in ['fn', 'target']])

        features_path = get_feature_path(method_name, out_path)

        new_rows = []

        for i, row in tqdm(path_df.iterrows(), total=len(path_df)):
            if verbose and i % 10 == 0:
                print('At file {}'.format(i + 1))
            try:
                path = join(data_path, row['fn'])
                df = pd.read_csv(path, index_col='time')
                df =df.reindex(sorted(df.columns), axis=1)
                if "Unnamed: 0" in df.columns:
                    del df["Unnamed: 0"]
                df = df_filter_func(df)
                new_row = f(df)
            except AssertionError:
                print('Error in file ' + row['fn'])
                continue
            except FileNotFoundError:
                print('Not found - ' + row['fn'])
                continue

            for col in ['fn', 'target']:
                new_row[col] = row[col]
            new_rows.append(new_row)
        res_df = pd.DataFrame(new_rows)
        res_df.to_csv(features_path, index=False)

    return wrapped


def merge_segment_dfs(df_list):
    df_copies = [df.copy() for df in df_list]
    for id, df in enumerate(df_copies):
        df['segment_id'] = id
    return pd.concat(df_copies, ignore_index=True)
