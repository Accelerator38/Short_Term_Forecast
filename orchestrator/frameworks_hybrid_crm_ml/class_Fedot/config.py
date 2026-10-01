import pandas as pd

class ConfigFedot:
    def __init__(self,
                 oilfield,
                 train_start,
                 train_end,
                 predict_start,
                 predict_end,
                 wells_norm,
                 coeff_f,
                 lags
                 ):
        self.oilfield = oilfield
        self.train_start = train_start
        self.train_end = train_end
        self.predict_start = predict_start
        self.predict_end = predict_end
        self.wells_norm = wells_norm
        self.coeff_f = coeff_f
        self.lags = lags

        # choose dates and discretization
        self.dates_predict = pd.date_range(predict_start, predict_end, freq="D")
        # 10% of dates predict
        self.date_discr = len(self.dates_predict) // 10
        while len(self.dates_predict) % self.date_discr == 1:
            self.date_discr += 1

