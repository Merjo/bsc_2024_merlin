import os
import pandas as pd
from src.tools.config import cfg
from src.read_data.load_data import load_gdp
from src.tools.country_mapping import map_iso3_codes, split_joint_country_data_for_parameters
from src.tools.tools import group_country_data_to_regions


def get_region_sector_splits():
    df_gdp = load_gdp(country_specific=True, per_capita=False)
    df_splits = _read_pauliuk_splits()
    df_gdp = df_gdp[df_gdp.index.isin(df_splits.index)]
    df_splits = df_splits[df_splits.index.isin(df_gdp.index)]

    df_splits = _restructure_splits(df_splits)

    df = pd.merge(df_splits, df_gdp, left_on='country', right_on='country')
    df.index = df_splits.index
    df = df.iloc[:, 1:].multiply(df['sector_split'], axis=0)
    df = group_country_data_to_regions(df_by_country=df, is_per_capita=False, data_split_into_categories=True)
    df_cat_sum = df.groupby(level=0).sum()
    df = df / df_cat_sum

    sector_splits = df.to_numpy().transpose()
    old_shape = sector_splits.shape
    new_shape = (old_shape[0], int(old_shape[1] / cfg.n_use_categories), cfg.n_use_categories)
    sector_splits = sector_splits.reshape(new_shape)

    return sector_splits


def _restructure_splits(df_splits):
    countries = df_splits.index
    df_splits = pd.melt(df_splits.reset_index(), id_vars=['country'], value_name='sector_split',
                        value_vars=['Transportation', 'Machinery', 'Construction', 'Products'])
    df_splits.index = pd.MultiIndex.from_product([pd.Index(cfg.in_use_categories, name='category'), countries])
    df_splits = df_splits.drop(columns=['country', 'variable'])
    df_splits = df_splits.swaplevel()
    df_splits = df_splits.sort_index()

    return df_splits


def _read_pauliuk_splits():
    splits_path = os.path.join(cfg.data_path, 'original', 'Pauliuk', 'Supplementary_Table_23.xlsx')
    df_splits = pd.read_excel(splits_path,
                              engine='openpyxl',
                              sheet_name='Supplementray_Table_23',
                              skiprows=3,
                              usecols='A:E')

    df_splits = map_iso3_codes(df_splits, country_name_column='Country name')
    df_splits = split_joint_country_data_for_parameters(df_splits)
    return df_splits


def _test():
    sector_splits = get_region_sector_splits()
    print(sector_splits.shape)


if __name__ == '__main__':
    _test()
