# import datetime
import pandas as pd
from frameworks_shelf_algo.class_Shelf.support_functions import _get_path
from frameworks_shelf_algo.class_Shelf.data_processor_shelf import DataProcessorShelf
from frameworks_shelf_algo.class_Shelf.config import ConfigShelf
from frameworks_shelf_algo.class_Shelf.calculator import CalculatorShelf


class DataPostProcessorShelf:
    def __init__(self, config: ConfigShelf):
        # print('DataPostProcessorShelf init')
        self._config = config
        self._data_processor = DataProcessorShelf(self._config)
        self._calculator = CalculatorShelf(self._config)
        # print('DataPostProcessorShelf CalculatorShelf done')
        df_shelf_result, df_shelf_result_liq, df_shelf_fact, df_shelf_fact_liq = \
            self._get_input_data(self._data_processor, self._calculator)
        self._path = _get_path(config.oilfield)
        self.df_shelf_ksp_format = pd.DataFrame()
        self.df_shelf_ksp_format = self._run(df_shelf_result, df_shelf_result_liq, df_shelf_fact, df_shelf_fact_liq)
        print('PostProcessor done')

    def _get_input_data(self, data_processor, calculator):
        df_shelf_fact = data_processor.df_fact_test_prd
        df_shelf_fact_liq = data_processor.df_fact_test_prd_liq
        df_shelf_result = calculator.df_result
        df_shelf_result_liq = calculator.df_result_liq
        return df_shelf_result, df_shelf_result_liq, df_shelf_fact, df_shelf_fact_liq

    def _run(self, df_shelf_result, df_shelf_result_liq, df_shelf_fact, df_shelf_fact_liq):
        path = self._path / 'welllist.feather'
        df_wellist = pd.read_feather(path)

        df_shelf_ksp_format = pd.DataFrame()
        df_shelf_ksp_format['dt'] = df_shelf_result.index

        ois_list = df_shelf_result.columns

        for ois_num in ois_list:
            indx = 0
            for ois_orig in df_wellist.ois:
                if str(ois_num) == str(ois_orig) and df_wellist.loc[indx, 'npath'] == 0:
                    n_well = df_wellist.loc[indx, 'num']
                    f_liq = str(n_well) + '_liq_true'
                    pr_liq = str(n_well) + '_liq_pred'
                    f_oil = str(n_well) + '_oil_true'
                    pr_oil = str(n_well) + '_oil_pred'
                    colmn = int(ois_num)
                    df_shelf_ksp_format[f_liq] = df_shelf_fact_liq[colmn]
                    df_shelf_ksp_format[pr_liq] = df_shelf_result_liq[ois_num]
                    df_shelf_ksp_format[f_oil] = df_shelf_fact[colmn]
                    df_shelf_ksp_format[pr_oil] = df_shelf_result[ois_num]
                indx = indx + 1

        df_shelf_ksp_format.set_index('dt', inplace=True)
        # df_shelf_ksp_format.to_excel(f'ResultsShelf.xlsx')

        return df_shelf_ksp_format