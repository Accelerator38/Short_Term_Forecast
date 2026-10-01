class ConfigShelf:
    """Конфигурация гиперпараметров для прогноза по темпам падения.

        oilfield: str, название месторождения

        shops: List, список цехов

        train_start: date,
        train_end: date,
            Начало и окончание периода адаптации всех моделей (не обязательно ППТП)

        predict_start: date,
        predict_end: date,
            Начало и окончание прогноза

        n_days_past: int, default = 30,
            Количество дней для вычисления темпа падения

        n_days_calc_avg: int, default = 5
            Количество дней для осреднения дебитов для вычисления темпа падения"""

    def __init__(self,
                 oilfield,
                 shops,
                 wells_ois,
                 train_start,
                 train_end,
                 predict_start,
                 predict_end,
                 n_days_past: int = 30,
                 n_days_calc_avg: int = 5
                 ):
        self.oilfield = oilfield
        self.shops = shops
        self.well_ois = wells_ois
        self.train_start = train_start
        self.train_end = train_end
        self.predict_start = predict_start
        self.predict_end = predict_end
        self.n_days_past = n_days_past
        self.n_days_calc_avg = n_days_calc_avg

        # print(oilfield,shops)
        # print("Config Shelf")
        # print(wells_ois)
        # print(train_start,train_end,predict_start,predict_end,
        #                        n_days_past,n_days_calc_avg)



