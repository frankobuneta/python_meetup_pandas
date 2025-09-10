
import json
import typing as tp

import pandas as pd


class PandasDtypes:

    @staticmethod
    def is_datetime(data_series: pd.Series) -> bool:

        return pd.api.types.is_datetime64_dtype(data_series)

    @staticmethod
    def is_float(data_series: pd.Series) -> bool:

        # Float can be float64 or Float64
        return pd.api.types.is_float_dtype(data_series)

    @staticmethod
    def is_int(data_series: pd.Series) -> bool:

        # Int can be int64 or Int64
        return pd.api.types.is_integer_dtype(data_series)

    def to_int(self, data_series: pd.Series) -> pd.Series:

        if self.is_int(data_series):
            return data_series

        try:
            return data_series.astype('int64')
        except pd.errors.IntCastingNaNError as _:
            return data_series.astype('Int64')

    @staticmethod
    def view(data_frame: pd.DataFrame) -> dict:

        return {
            series_name: f'{column_data.dtypes}'
            for series_name, column_data in data_frame.items()}

    def view_datetimes(self, data_frame: pd.DataFrame) -> list:

        return [
            series_name for series_name, data_series in
            data_frame.items() if self.is_datetime(data_series)]

    def view_ints(self, data_frame: pd.DataFrame) -> list:

        return [
            series_name for series_name, data_series in
            data_frame.items() if self.is_int(data_series)]

    def view_floats(self, data_frame: pd.DataFrame) -> list:

        return [
            series_name for series_name, data_series in
            data_frame.items() if self.is_float(data_series)]

    def view_objects(self, data_frame: pd.DataFrame) -> list:

        return [
            series_name for series_name, column_dtype in
            self.view(data_frame).items()
            if column_dtype == 'object']

    def cast_objects(self, data_frame: pd.DataFrame) -> pd.DataFrame:

        object_dtypes_to_convert = {
            series_name: 'string' for series_name in
            self.view_objects(data_frame)}

        return data_frame.astype(object_dtypes_to_convert)


class MergeErrorDuplicateRows(pd.errors.MergeError):
    pass


class MergeErrorDuplicateColumns(pd.errors.MergeError):
    pass


class PandasTools:

    def __init__(self):

        self._dtype = PandasDtypes()

    def merge_safe(
            self,
            left_df: pd.DataFrame,
            right_df: pd.DataFrame,
            on: tp.Union[list, str],
            how: tp.Literal['left', 'right', 'inner', 'outer', 'cross'] = 'left',
            indicator: tp.Union[str, bool] = False) -> pd.DataFrame:

        # It is also possible to just kwargs everything after indicator

        err_msg = f'\nLeft: {left_df.attrs["name"]}\n'
        err_msg += f'Right: {right_df.attrs["name"]}\n'

        # Convert on to list
        if isinstance(on, str):
            on = [on]

        # Are there any duplicate column names in left_df, right_df?

        duplicates_result = self.duplicate_columns(
            self.columns(left_df),
            self.columns(right_df),
            on)

        if len(duplicates_result) > 0:
            err_msg += f'Duplicate columns: {duplicates_result}'
            raise MergeErrorDuplicateColumns(err_msg)

        right_int_cols = self._dtype.view_ints(right_df)

        # Testing rows
        row_count = self.rows(left_df)

        merge_df = pd.merge(
            left_df, right_df, on=on, how=how, indicator=indicator)

        if row_count < self.rows(merge_df):
            err_msg += f'Duplicate rows'
            raise MergeErrorDuplicateRows(err_msg)

        for column_name in right_int_cols:
            merge_df[column_name] = self._dtype.to_int(merge_df[column_name])

        return merge_df

    def columns(
            self,
            data_frame: pd.DataFrame,
            contains: tp.Optional[str] = None,
            case_sensitive: bool = False) -> list:

        return self._list_contains(
            list(data_frame.columns), contains, case_sensitive)

    def duplicate_columns(
            self,
            left_columns: list,
            right_columns: list,
            exclude_list: tp.Optional[list] = None) -> list:

        if exclude_list:
            left_columns = self._exclude_items(
                left_columns, exclude_list)
            right_columns = self._exclude_items(
                right_columns, exclude_list)

        return list(set(left_columns) & set(right_columns))

    def exclude_columns(
            self,
            data_frame: pd.DataFrame,
            exclude_list: tp.Union[list, str]) -> pd.DataFrame:

        if isinstance(exclude_list, str):
            exclude_list = [exclude_list]

        return data_frame[self._exclude_items(self.columns(data_frame), exclude_list)]

    @staticmethod
    def rows(
            data_frame: pd.DataFrame) -> int:

        return data_frame.shape[0]

    @staticmethod
    def _list_contains(
            items_list: list,
            contains: tp.Optional[str] = None,
            case_sensitive: bool = False) -> list:

        if not contains:
            return items_list

        if case_sensitive is False:
            contains = contains.lower()

        return [
            item for item in items_list if
            contains in (item if case_sensitive else item.lower())]

    @staticmethod
    def _exclude_items(
            items_list: list,
            exclude_list: list) -> list:

        return [
            item for item in items_list
            if item not in exclude_list]


def tidy_attrs(data_frame: pd.DataFrame):

    print(json.dumps(data_frame.attrs, indent=4))


def tidy_sales(data_frame: pd.DataFrame) -> 'pd.io.formats.style.Styler':

    def highlight_column(data_series: pd.Series):

        row_count = data_series.shape[0]
        # Function to highlight column, you need to hardcode column name
        if data_series.name == 'revenue':
            return ['background-color: lightgreen'] * row_count
        else:
            return [''] * row_count

    def highlight_na(item_value):

        # Function to highlight NA cells in column B
        if pd.isna(item_value):
            return 'background-color: yellow'
        else:
            return ''

    return (
        data_frame.style
        .apply(highlight_column)  # Highlight column
        .map(highlight_na, subset=['account'])  # Highlight NA in account
        .format({
            'close_value': '{:.2f}',
            'revenue': '{:.2f}',
            'close_date':
                lambda x: x.strftime('%d.%m.%Y')
                if pd.isna(x) is False else ''  # Show date without time, Croatian format
        }))


def tidy_accounts(data_frame: pd.DataFrame) -> 'pd.io.formats.style.Styler':

    def highlight_duplicates(row):

        column_name = 'account'
        if (row[column_name] in
                data_frame[column_name][data_frame[column_name].duplicated()].values):
            return ['background-color: yellow'] * len(row)
        else:
            return [""] * len(row)

    return data_frame.style.apply(highlight_duplicates, axis=1)
