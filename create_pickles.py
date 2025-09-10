
import pandas as pd
import numpy as np


def randomise_close_date() -> pd.Series:
    # Reproducibility
    np.random.seed(42)

    # Create a range of all dates in 2017
    dates = pd.date_range(start=pd.Timestamp(2017, 1, 1), end=pd.Timestamp(2017, 12, 31), freq="D")

    # Randomly sample 20 dates from that range
    random_dates = np.random.choice(dates, size=10)

    # Convert to a Series
    return pd.Series(random_dates, dtype='datetime64[ns]')


def create_sales_pickle(downloads_path: str):

    sales_file_name = 'sales_pipeline.csv'

    sales_dtypes = {
        'opportunity_id': 'string',
        'sales_agent': 'string',
        'product': 'string',
        'account': 'string',
        'deal_stage': 'string',
        # 'engage_date': 'datetime64[ns]',
        # 'close_date': 'datetime64[ns]',
        'close_value': 'float64'}

    sales_metadata = {
        'name': 'sales',
        'source_file': 'sales_pipeline.csv',
        'source_path': 'C:/Users/FrankoBuneta1/Downloads/',
        'source_type': 'CSV',
        'source_url': 'https://www.kaggle.com/datasets/agungpambudi/crm-sales-predictive-analytics'}

    sales = pd.read_csv(
        filepath_or_buffer=downloads_path + sales_file_name,
        dtype=sales_dtypes)

    sales.attrs = sales_metadata

    sales = sales.astype({
        'engage_date': 'datetime64[ns]',
        'close_date': 'datetime64[ns]'})

    # Reduce the dataset for the purpose of the presentation
    select_columns = [
        'opportunity_id', 'account', 'deal_stage',
        'close_date', 'close_value']
    sales = sales[select_columns]
    sales['revenue'] = sales['close_value']

    # Only 10 rows as sample
    sales = pd.concat([sales[100:109], sales[-1:]], ignore_index=True)

    close_value_not_na = ~sales['close_value'].isna()
    sales.loc[close_value_not_na, 'close_date'] = randomise_close_date()

    sales.to_pickle('pickles/sales.pkl')

    # Return select accounts, so that only those are included
    # in the accounts dataset
    return list(sales['account'].dropna().unique())


def create_accounts_pickle(downloads_path: str, accounts_selection: list):
    """
    Pickle should be available on GitHub, but you can download the csv files
    and create it yourself.
    """
    accounts_file_name = 'accounts.csv'

    accounts_dtypes = {
        'account': 'string',
        'sector': 'string',
        'year_established': 'int64',
        'revenue': 'float64',
        'employees': 'int64',
        'office_location': 'string',
        'subsidiary_of': 'string'}

    accounts_metadata = {
        'name': 'accounts',
        'source_file': 'accounts.csv',
        'source_path': 'C:/Users/FrankoBuneta1/Downloads/',
        'source_type': 'CSV',
        'source_url': 'https://www.kaggle.com/datasets/agungpambudi/crm-sales-predictive-analytics'}

    accounts = pd.read_csv(
        filepath_or_buffer=downloads_path + accounts_file_name,
        dtype=accounts_dtypes)

    accounts.attrs = accounts_metadata

    select_columns = ['account', 'sector', 'revenue', 'employees']
    accounts = accounts[select_columns]
    accounts = accounts[accounts['account'].isin(accounts_selection)]

    # Create duplicate record
    accounts = pd.concat([accounts, accounts[-1:]], ignore_index=True)

    accounts.to_pickle('pickles/accounts.pkl')


def run(downloads_path: str):

    accounts_selection = create_sales_pickle(downloads_path)
    create_accounts_pickle(downloads_path, accounts_selection)
