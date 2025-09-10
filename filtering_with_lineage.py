
import typing as tp

import pandas as pd


class PandasFilterUnknownOperator(ValueError):
    pass


@pd.api.extensions.register_dataframe_accessor('case')
class PandasFilter:

    def __init__(self, data_frame: pd.DataFrame):

        self._data = data_frame
        self.operators: dict[str, tp.Callable] = {
            'equal': self.eq,
            'greater than': self.gt,
            'lesser than': self.lt,
            'is null': self.isna,
            'not': self.not_series,
            'and': self.and_all,
            'or': self.or_all}

        self.lineage = []
        self._log = []

    def _get_series(self, series_name: tp.Union[pd.Series, str]) -> pd.Series:

        if isinstance(series_name, str):
            return self._data[series_name]
        else:
            return series_name

    def _get_null_series(
            self,
            series_null_name: tp.Optional[str] = None,
            dtype: tp.Optional[str] = None) -> pd.Series:

        rows = self._data.shape[0]
        return pd.Series([pd.NA] * rows, index=self._data.index, name=series_null_name, dtype=dtype)

    def eq(self, series_name: str, criteria: tp.Any) -> pd.Series:

        series_data = self._get_series(series_name)
        # noinspection PyTypeChecker
        return series_data == criteria

    def gt(self, series_name: str, criteria: tp.Any) -> pd.Series:

        series_data = self._get_series(series_name)
        # noinspection PyTypeChecker
        return series_data > criteria

    def lt(self, series_name: str, criteria: tp.Any) -> pd.Series:

        series_data = self._get_series(series_name)
        # noinspection PyTypeChecker
        return series_data < criteria

    def isna(self, series_name: str) -> pd.Series:

        series_data = self._get_series(series_name)
        return series_data.isna()

    def not_series(self, series_name: str) -> pd.Series:

        series_data = self._get_series(series_name)
        return ~series_data

    @staticmethod
    def and_all(series_list: list[pd.Series]):

        series_result = series_list[0]

        for series_data in series_list[1:]:
            series_result &= series_data

        return series_result

    @staticmethod
    def or_all(series_list: list[pd.Series]):

        series_result = series_list[0]

        for series_data in series_list[1:]:
            series_result |= series_data

        return series_result

    def mask(
            self,
            function_name: str,
            input_values: tp.Any,
            criteria: tp.Optional[tp.Any] = None) -> pd.Series:

        if function_name not in self.operators:
            err_msg = f'Unknown operator: {function_name}'
            raise PandasFilterUnknownOperator(err_msg)

        if function_name in ['and', 'or']:
            input_values = [self.mask(*input_sub_values) for input_sub_values in input_values]
            self._update_logical_log(function_name, self._get_next_logical_id(), len(input_values))
        else:
            self._log.append({
                'data_source': None,
                'logical': None,
                'logical_id': None,
                'function_name': function_name,
                'input_values':  f'{input_values.name}' if isinstance(input_values, pd.Series) else f'{input_values}',
                'criteria': f'{criteria}' if criteria else None,
                'output': 'tbu'})

        func = self.operators[function_name]

        if criteria is not None:
            return func(input_values, criteria)
        else:
            return func(input_values)

    def if_then(
            self,
            mask_list: list,
            output: tp.Any,
            series_output: tp.Optional[pd.Series] = None,
            series_output_name: tp.Optional[str] = None,
            dtype: tp.Optional[str] = None) -> pd.Series:

        if series_output is None:
            series_output = self._get_null_series(series_output_name, dtype)

        output_mask = self.mask(*mask_list) & series_output.isna()
        series_output[output_mask] = output

        self._update_output_log(output)

        return series_output

    def else_if_all(self, else_if_data: dict):

        self._log = []
        series_name = else_if_data.get('series_name', None)
        dtype = else_if_data.get('dtype', None)

        series_output = self._get_null_series(series_name, dtype)

        # If-then statements loop
        for mask_list, output in else_if_data['statements']:
            series_output = self.if_then(
                mask_list, output, series_output)

        # Else
        self.if_then(['is null', series_output], else_if_data['else'])

        self._data[series_output.name] = series_output

        self._update_output_name_log(series_name)
        self.lineage += self._log

    def _update_logical_log(self, function_name: str, logical_id: int, update_dicts: int):

        for row in self._log[-update_dicts:]:
            row.update({'logical': function_name, 'logical_id': logical_id})

    def _update_output_log(self, output: tp.Any):

        if isinstance(output, pd.Series):
            output = output.name
        else:
            output = f'{output}'

        for row in self._log:
            if row['output'] == 'tbu':
                row['output'] = output

    def _update_output_name_log(self, series_name: tp.Optional[str]):

        for row in self._log:
            if 'name' in self._data.attrs.keys():
                data_source = self._data.attrs['name']
            else:
                data_source = None
            row.update({'output_name': series_name, 'data_source': data_source})

    def _get_next_logical_id(self) -> int:

        logical_ids = [row.get('logical_id', 0) for row in self._log]
        return max([item or 0 for item in logical_ids]) + 1
