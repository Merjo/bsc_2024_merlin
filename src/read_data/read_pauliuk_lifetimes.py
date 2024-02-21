import os
import pandas as pd
from src.tools.config import cfg
from src.read_data.load_data import load_lifetimes


def load_pauliuk_lifetimes():
    lifetime_path = os.path.join(cfg.data_path, 'original', 'Pauliuk', 'Pauliuk_lifetimes.csv')
    df = pd.read_csv(lifetime_path)
    if cfg.region_data_source == 'Pauliuk':
        pass
    elif cfg.region_data_source == 'REMIND':
        df = _map_pauliuk_lifetimes_to_remind_regions(df)
    else:
        raise RuntimeError(f'Pauliuk lifetimes can not be mapped to region source {cfg.region_data_source}.'
                           f'Change lifetime data source or implement region mapping to desired region source.')
    df = df.set_index('region')
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)

    mean = df.to_numpy()
    std_dev = mean * 0.3

    return mean, std_dev


def _map_pauliuk_lifetimes_to_remind_regions(df):
    map_path = os.path.join(cfg.data_path, 'original', 'iso3_codes', 'pauliuk_REMIND_map.csv')
    df_map = pd.read_csv(map_path)
    df = pd.merge(df_map, df, left_on='pauliuk_region', right_on='region', how='outer')
    df = df.drop(columns=['pauliuk_region', 'region'])
    df = df.rename(columns={'remind_region': 'region'})

    return df


def _test():
    mean, std_dev = load_lifetimes('Pauliuk')

    print(mean.shape)
    print(std_dev.shape)


if __name__ == '__main__':
    _test()
