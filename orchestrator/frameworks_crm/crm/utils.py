from .models.parameters import ExcelExporter
import numpy as np
import pandas as pd


def export_train_test_data_to_excel(filename, train_index, test_index, data, header):
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'



    with ExcelExporter(filename) as workbook:
        index_test_format = workbook.add_format({'bg_color': '#A6F9AE', 'num_format': 'dd/mm/yy', 'bold': True})
        values_test_format = workbook.add_format({'bg_color': '#A6F9AE', 'num_format': r'0.000'})

        index_train_format = workbook.add_format({'bg_color': '#FFD276', 'num_format': 'dd/mm/yy', 'bold': True})
        values_train_format = workbook.add_format({'bg_color': '#FFD276', 'num_format': r'0.000'})

        for name in data:
            workbook.index_format = index_train_format
            workbook.values_format = values_train_format
            sheet = workbook.write_matrix_to_worksheet(name, train_index, _get_matrix(data[name][0]), header)

            workbook.index_format = index_test_format
            workbook.values_format = values_test_format
            workbook.write_matrix_to_worksheet(sheet, test_index, _get_matrix(data[name][1]),
                                               header=None, loc=(len(train_index), 0))

def _get_matrix(data):
    data_as_matrix = None
    if type(data) is np.ndarray:
        data_as_matrix = data
    elif type(data) is pd.DataFrame:
        data_as_matrix = data.values
    else:
        raise ValueError(f'Invalid data passed as an argument')
    return data_as_matrix