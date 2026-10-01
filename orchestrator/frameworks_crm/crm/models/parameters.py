import numpy as np
import xlsxwriter
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from logging import getLogger

logger = getLogger(__name__)


def estim_productivity(debit, pressure, neg_J=1.0): # оценка продуквтивности
    d_p = pressure[1:] - pressure[:-1]
    d_d = debit[1:] - debit[:-1]
    J = - d_d / d_p
    J = np.delete(J, np.where(np.isnan(J)))
    J = np.delete(J, np.where(np.isinf(J)))
    J = np.average(J)
    if J <= 0:
        logger.warning("Average productivity is negative!")
    return np.max([J, neg_J])

def estim_res_pres(production_pres, injection_pres): # оценка давления в добывающих и нагнетательных скважинах
    avr_prod_pres = np.average(production_pres[~np.isnan(production_pres)])
    avr_inj_pres = np.average(injection_pres[~np.isnan(injection_pres)])
    return np.mean(avr_inj_pres, avr_prod_pres)


def inverse_distance_mtrx(prod_coord, inj_coord, dist_thres=2*1e3): #матрица обратных расстояний
    pair_dist = pairwise_distances(X=inj_coord, Y=prod_coord)
    pair_dist[pair_dist > dist_thres] = 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        pair_dist =  1. / pair_dist
        pair_dist[pair_dist == np.inf] = 0.0
    return pair_dist


class ReverseSparseConverter: # преобразование в обратные разряжённую матрицу
    def __init__(self, indices, shape):
        self.indices = indices
        self.shape = shape

    def convert(self, data):
        return csr_matrix((data, self.indices), self.shape).toarray()


class ParamsLinker: # компановщик параметров
    def __init__(self, *args):
        self.shapes = []
        self.sizes = []
        for entity in args:
            self.shapes.append(entity.shape)
            self.sizes.append(entity.size)

    def _args_assertions(self, *args): #утвержение аргументов
        assert len(args) == len(self.shapes), 'number of arguments must agree'
        for i in range(len(args)):
            assert args[i].shape == self.shapes[i], 'shape for {} arg doesn\'t match'.format(i)

    def link(self, *args): #ссылка на аргументы
        self._args_assertions(*args)
        concated = np.concatenate([param.flatten() for param in args])
        return concated.reshape(concated.size)

    def split(self, X):# деление массива
        assert X.size == np.sum(self.sizes), 'invalid size of X'
        ret = []
        border = 0
        for i in range(len(self.shapes)):
            data = X[border: border + self.sizes[i]]
            border += self.sizes[i]
            ret.append(data.reshape(self.shapes[i]))
        return ret


class ExcelExporter(xlsxwriter.Workbook):

    def __init__(self, filename=None, options=None,
                 index_format={'bold': True}, header_format={'bold': True}, values_format={'num_format': r'0.000'}):
        super().__init__(filename, options)
        self.index_format = self.add_format(properties=index_format)
        self.header_format = self.add_format(properties=header_format)
        self.values_format = self.add_format(properties=values_format)

    def write_series_to_worksheet(self, worksheet, index, values, header=None, loc=(0, 0)):
        if type(worksheet) is str:
            worksheet = self.add_worksheet(worksheet)
        elif type(worksheet) is xlsxwriter.worksheet_class:
            worksheet = worksheet
        else:
            raise ValueError('Invalid worksheet parameter passed')

        if header is not None and len(header) != 2:
            raise ValueError('Invalid header passed during saving series to excel')
        row, col = loc

        if header is not None:
            for h in header:
                worksheet.write(row, col, str(h), self.header_format)
                col += 1
            row += 1
            col = loc[1]

        for i, v in zip(index, values):
            worksheet.write(row, col, i, self.index_format)
            worksheet.write(row, col+1, v, self.values_format)
            row += 1

        return worksheet

    def write_matrix_to_worksheet(self, worksheet, index, values, header, loc=(0, 0)):
        if type(worksheet) is str:
            worksheet = self.add_worksheet(worksheet)
        elif type(worksheet) is self.worksheet_class:
            worksheet = worksheet
        else:
            raise ValueError('Invalid worksheet parameter passed')

        if len(index) != len(values):
            raise ValueError('Invalid index passed during saving matrix to excel')

        row, col = loc
        if header is not None and len(header) == values.shape[1]:
            col += 1
            logger.warning('Header for index is missing')
        elif header is not None and len(header) != values.shape[1]+1:
            raise ValueError('Invalid header passed during saving matrix to excel')

        if header is not None:
            for h in header:
                worksheet.write(row, col, str(h), self.header_format)
                col += 1
            row += 1
            col = loc[1]

        for i, row_vals in zip(index, values):
            worksheet.write(row, col, i, self.index_format)
            for v in row_vals:
                col += 1
                worksheet.write(row, col, v, self.values_format)
            col = loc[1]
            row += 1

        return worksheet
