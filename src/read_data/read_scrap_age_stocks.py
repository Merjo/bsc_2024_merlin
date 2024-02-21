import os
import pandas as pd
from src.tools.config import cfg
from src.read_data.load_data import load_regions


def get_scrap_age_stocks(country_specific: bool = True, do_current_not_future: bool = True):
    df = _read_scrap_age_stocks_original()
    if do_current_not_future:
        df = df.loc[:, :2008]
    if country_specific:
        df_regions = load_regions('Pauliuk')
        df = df_regions.reset_index().merge(df.reset_index(), on='region')
        df = df.drop(columns=['region'])
        df = df.set_index(['country', 'category'])
    return df


def _read_scrap_age_stocks_original():
    scrap_age_data_path = os.path.join(cfg.data_path, 'original', 'Pauliuk',
                                       'GlobalSteel_DataExtract.xls')
    sheet_names = ['Transportation', 'Machinery', 'Construction', 'Products']

    region_dict = {
        'North America': 'NAM',
        'Latin America': 'LAM',
        'Western Europe': 'WEU',
        'Eastern Europe and CIS': 'CIS',
        'Africa': 'AFR',
        'Middle East': 'MES',
        'India': 'IND',
        'China': 'CHA',
        'Developed Asia and Ocenia': 'DAO',
        'Developing Asia': 'DVA'
    }
    dfs = [pd.read_excel(
        io=scrap_age_data_path,
        sheet_name=sheet_name,
        skiprows=1,
        usecols='A:K').set_index('Year').transpose().rename(index=region_dict) for sheet_name in sheet_names]
    df = pd.concat(dfs, keys=cfg.in_use_categories)
    df = df.loc[:, 1900:2101]
    df = df.reorder_levels([-1, 0])
    df = df.sort_index()
    df.index.names = ['region', 'category']
    return df


# -- TEST FILE FUNCTION --

def _test():
    from src.read_data.load_data import load_stocks
    df = load_stocks('ScrapAge', country_specific=False, per_capita=True, recalculate=True)
    print(df)


if __name__ == "__main__":
    _test()
