import datetime

from frameworks_crm.class_CRM.calculator import Calculator as CalculatorCRM
from frameworks_crm.class_CRM.config import ConfigCRM
from frameworks_crm.class_Fedot.fedot_model import FedotModel

if __name__ == '__main__':
    oilfield = 'Отдельное'
    date_start = datetime.date(2019, 4, 1)
    date_test = datetime.date(2020, 2, 1)
    date_end = datetime.date(2020, 4, 30)
    calc_CRM = True
    calc_CRMIP = False
    config = ConfigCRM(date_start_adapt=date_start,
                       date_end_adapt=date_test - datetime.timedelta(days=1),
                       date_end_forecast=date_end,
                       oilfield=oilfield,
                       calc_CRM=calc_CRM,
                       calc_CRMIP=calc_CRMIP)
    calculator_CRM = CalculatorCRM(config)
    # Результаты расчетов
    if calc_CRM:
        CRM = calculator_CRM.CRM
        pred_CRM = calculator_CRM.pred_CRM
    if calc_CRMIP:
        CRMIP = calculator_CRM.CRMIP
        pred_CRMIP = calculator_CRM.pred_CRMIP

    # Модель Fedot на основании коэф-тов взаимовлияния от CRM
    fedot_entity = FedotModel(oilfield=oilfield,
                              train_start=date_start,
                              train_end=date_test - datetime.timedelta(days=1),
                              predict_start=date_test,
                              predict_end=date_end,
                              wells_to_calc=calculator_CRM.f.columns,
                              coeff=calculator_CRM.f,
                              save_plots=True)
    stats = fedot_entity.statistics
