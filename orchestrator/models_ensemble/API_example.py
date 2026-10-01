import numpy as np
import pandas as pd
from loguru import logger

from models_ensemble.calculator import Calculator as CalculatorEnsemble
from models_ensemble.config import Config as ConfigEnsemble

logger.remove()
np.random.seed(1)

wellname = 'some_wellname'
data = pd.DataFrame(
    np.random.randint(-1, 3, size=(120, 4)),
    columns=['a', 'b', 'c', 'y_fact'],
    index=pd.date_range(start='1/1/2018', periods=120)
)

# Формат входных данных - список словарей с ключами:
# имя скважины 'wellname' и ее датафрейм 'df'
input_data = [
    {'wellname': wellname,
     'df': data},
]

if __name__ == '__main__':
    calculator_ensemble = CalculatorEnsemble(
        ConfigEnsemble(adaptation_days_number=30,
                       interval_probability=0.8,
                       name_of_y_true='y_fact'),
        input_data,
        logging=False,
    )
    result_train = calculator_ensemble.result_train
    result_test = calculator_ensemble.result_test
    weights = calculator_ensemble.weights

    data['ensemble'] = result_test[wellname]['ensemble']
    data['interval_upper'] = result_test[wellname]['interval_upper']
    data['interval_lower'] = result_test[wellname]['interval_lower']
