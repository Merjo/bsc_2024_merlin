import os
import pandas as pd
from src.tools.config import cfg
from src.tools.tools import transform_per_capita
from src.tools.split_data_into_subregions import split_areas_by_gdp


def get_iedb_country_stocks():
    df = _get_current_iedb_stocks(perCapita=True)
    return df


# -- DATA ASSEMBLY FUNCTIONS --


def _get_current_iedb_stocks(perCapita=False):
    df_original = _read_iedb_stocks_by_category_original()
    df = _clean_iedb_stocks(df_original)
    df_iso3_map = _read_pauliuk_iso3_map()
    df = _reformat_iedb_stocks(df, df_iso3_map)
    areas_to_split = ['Belgium-Luxembourg', 'Czechoslovakia', 'Fmr USSR', 'Fmr Yugoslavia',
                      'Netherlands Antilles', 'So. African Customs Union']
    df = split_areas_by_gdp(areas_to_split, df, df_iso3_map, data_is_by_category=True)
    if perCapita:
        df = transform_per_capita(df, total_from_per_capita=False, country_specific=True)

    return df


def _reformat_iedb_stocks(df_original, df_iso3_map):
    df_original = df_original.pivot(index=['country_name', 'category'], columns='year', values='stock')
    df_original = df_original.reset_index()
    df = pd.merge(df_iso3_map, df_original, on='country_name')
    df = df.drop(columns='country_name')
    df = df.set_index(['country', 'category'])

    return df


def _clean_iedb_stocks(df_pauliuk):
    df_pauliuk = df_pauliuk.rename(columns={'aspect 3 : time': 'year',
                                            'aspect 4 : commodity': 'category_description',
                                            'aspect 5 : region': 'country_name',
                                            'value': 'stock'})
    df_pauliuk['stock'] = df_pauliuk['stock'] * 1000.  # convert grom Giga grams to tons
    df_cat_names = pd.DataFrame.from_dict({
        'category_description': ['vehicles and other transport equipment',
                                 'industrial machinery',
                                 'buildings - construction - infrastructure',
                                 'appliances - packaging - other'],
        'category': cfg.in_use_categories
    })

    df_pauliuk = pd.merge(df_cat_names, df_pauliuk, on='category_description')
    df_pauliuk = df_pauliuk.drop(columns=['category_description'])
    return df_pauliuk


def _read_iedb_stocks_by_category_original():
    pauliuk_data_path = os.path.join(cfg.data_path, 'original', 'unifreiburg_ie_db',
                                     '2_IUS_steel_200R_4Categories.xlsx')
    df_pauliuk = pd.read_excel(
        io=pauliuk_data_path,
        engine='openpyxl',
        sheet_name='Data',
        usecols=['aspect 3 : time', 'aspect 4 : commodity', 'aspect 5 : region', 'value'])

    return df_pauliuk


def _read_iedb_stocks_aggregated_original():
    pauliuk_data_path = os.path.join(cfg.data_path, 'original', 'unifreiburg_ie_db',
                                     '2_IUS_steel_200R.xlsx')
    df_pauliuk = pd.read_excel(
        io=pauliuk_data_path,
        engine='openpyxl',
        sheet_name='Data',
        usecols=['aspect 3 : time', 'aspect 5 : region', 'value'])

    # clean up
    df_pauliuk = df_pauliuk.rename(columns={'aspect 3 : time': 'year',
                                            'aspect 5 : region': 'country_name',
                                            'value': 'stock'})
    df_pauliuk['stock'] = df_pauliuk['stock'] * 1000.  # convert grom Giga grams to tons

    # reformat
    df_iso3_map = _read_pauliuk_iso3_map()
    df_pauliuk = df_pauliuk.pivot(index=['country_name'], columns='year', values='stock')
    df_pauliuk = df_pauliuk.reset_index()
    df = pd.merge(df_iso3_map, df_pauliuk, on='country_name')
    df = df.drop(columns='country_name')
    df = df.set_index(['country'])

    return df


def _read_pauliuk_iso3_map():
    pauliuk_iso3_path = os.path.join(cfg.data_path, 'original', 'unifreiburg_ie_db', 'Pauliuk_countries.csv')
    df_iso3 = pd.read_csv(pauliuk_iso3_path)

    df_iso3 = df_iso3.rename(columns={
        'country': 'country_name',
        'iso3c': 'country'
    })

    return df_iso3


# -- TEST FILE FUNCTION --

def _test():
    from src.read_data.load_data import load_stocks
    df = load_stocks('IEDatabase', country_specific=False, per_capita=True, recalculate=True)
    print(df)


if __name__ == "__main__":
    _test()
