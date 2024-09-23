import csv
import xml.etree.ElementTree as ET

import pandas as pd

XNUM = 80
YNUM = 24


class XResult:
    def __init__(self, csvfile):
        self._dataframe = pd.read_csv(csvfile)

        work_list = self._dataframe["work_name"].values.tolist()
        self.work_list = list(set(work_list))

    def __call__(self, workname, x, y):
        _df = self._dataframe[self._dataframe['work_name'] == workname]

        _df = _df[_df['x'] == x]
        _df = _df[_df['y'] == y]

        if len(_df) == 1:
            return _df['val'].values[0]
        else:
            print(workname, x, y)
            print(_df)
            return None


def merge_results(xgx_csv: str, x_csv: str, out_file: str) -> None:
    xres = XResult(x_csv)

    fw = open(out_file, 'w')
    csv_writer = csv.writer(fw, lineterminator='\n')

    # xgx
    fx = open(xgx_csv, "r")
    log = csv.reader(fx, delimiter=',', lineterminator='\n')

    work_no = 0
    work_name = "initial"
    file_name = "initial"
    for row_cnt, row in enumerate(log):
        if row_cnt == 0:
            csv_writer.writerow(row + ["work_name",
                                       "work_no",
                                       "x_no",
                                       "y_no",
                                       "xval",
                                       "center_flg",
                                       "work_flg"])
            continue

        try:
            _file_name = row[0]
            x = float(row[6])
            y = float(row[7])
        except:
            print(row)
            raise ValueError

        try:
            _work_name = _file_name.split('_')[2]
        except:
            print(_file_name)
            raise ValueError

        if _file_name != file_name:
            if _work_name != work_name:
                work_name = _work_name
                work_no = 0
                file_name = _file_name
                print(work_name)
            else:
                work_no = 1

        _work_name = work_name.replace('-', '')
        _work_name = _work_name.upper()
        x_no = (x - 106.0) / 45.6
        y_no = (y - 423.8) / 53.0

        if _work_name not in xres.work_list:
            exist_work = "no"
        else:
            exist_work = "yes"

        if work_no == 0:
            x_no = XNUM - x_no -1
            y_no = YNUM - y_no - 1
        else:
            x_no = x_no
            y_no = y_no

        if exist_work=='yes':
            xval = xres(_work_name, round(x_no), round(y_no))
        else:
            xval = None

        if round(x_no) < 1 or round(x_no) > XNUM -2 or round(y_no) < 1 or round(y_no) > YNUM -2:
            is_center = "no"
        else:
            is_center = "yes"


        csv_writer.writerow(row + [_work_name,
                                   work_no,
                                   round(x_no),
                                   round(y_no),
                                   xval,
                                   is_center,
                                   exist_work])

    fx.close()
    fw.close()


def main():
    xgx_file = f'xgx_results1.csv'
    x_file = 'xray_result.csv'
    res_file = f'merge_results.csv'

    merge_results(xgx_file, x_file, res_file)


if __name__ == '__main__':
    main()
