import os
import csv
import shutil
import glob
import datetime

import numpy as np
import cv2
import optuna

from blob_analysis import blob_analysis


class OptunaApp:
    """
    Optunaを使用して画像処理パラメータの最適化を実行。
    OK/NGホルダにある画像データを処理し、判定精度を最大化する。
    目的変数を複数にすると収束できなかったため、現時点では目的変数は１つに固定。
    """

    def __init__(self, ok_dir: str, ng_dir: str,
                 out_dir: str = './log',
                 img_ext: str = 'jpg',
                 out_file: str = 'results.csv'):
        """
        :param ok_dir: OK画像データのホルダ
        :param ng_dir: NG画像データのホルダ
        :param out_dir: 出力ホルダ
        :param img_ext: 画像ファイルの拡張子
        :param out_file: 結果出力ファイル
        """
        self.ok_dir = ok_dir
        self.ng_dir = ng_dir
        self.out_dir = out_dir
        self.img_ext = img_ext
        self.res_file = out_file

        self.trial_cnt = None
        self.study = None
        self.best_score = 0
        self.params_dict = {}
        self.res_dict = {}

        self.memo = None

    def _objective(self, trial):
        """
        optunaの目的関数
        :param trial:
        :return:
        """

        self.trial_cnt += 1
        self._set_optuna_params(trial)

        score = self._calc_total_score()

        if score > self.best_score:
            self.best_score = score

        self._save_results()

        return score

    def _set_optuna_params(self, trial):
        """
        最適化したい変数を定義
        :param trial:
        :return:
        """

        # HSV parameter
        if self.best_score > 0.5:
            step = 5
            _max = 260
        else:
            step = 10
            _max = 260

        h_min = trial.suggest_float('h_min', 0, 180, step=step)
        h_max = trial.suggest_float('h_max', h_min, 180, step=step)
        s_min = trial.suggest_float('s_min', 0, _max, step=step)
        s_max = trial.suggest_float('s_max', s_min, _max, step=step)
        v_min = trial.suggest_float('v_min', 0, _max, step=step)
        v_max = trial.suggest_float('v_max', v_min, _max, step=step)
        s_min = min(255, s_min)
        s_max = min(255, s_max)
        v_min = min(255, v_min)
        v_max = min(255, v_max)

        mask_low = np.array([int(h_min), int(s_min), int(v_min)])
        mask_high = np.array([int(h_max), int(s_max), int(v_max)])

        # blob_area parameter
        blob_min = 300
        blob_margin = 0.1  # OK画像はNG画像より厳しい面積閾値で判定

        # 画像処理で使用しない変数には、最初に"_"を付ける。
        self.params_dict = {'mask_low': mask_low,
                            'mask_high': mask_high,
                            '_blob_min': blob_min,
                            '_blob_margin': blob_margin,
                            "_h_min": h_min,
                            "_h_max": h_max,
                            "_s_min": s_min,
                            "_s_max": s_max,
                            "_v_min": v_min,
                            "_v_max": v_max,
                            }

    def _pickup_params_dict(self, params_dict: dict):
        """
        params_dictの内、画像処理で使用する変数のみ抽出
        :param params_dict:
        :return:
        """

        _dict = {}
        for key, val in params_dict.items():
            if key[0] == '_': continue

            _dict[key] = val

        return _dict

    def _calc_score(self, is_ok: bool, output_log: bool = False):
        """
        画像処理を実行し、スコアを算出する
        :param is_ok: OKデータかどうか
        :param output_log: 処理画像を出力するかどうか
        :return:
        """

        _params_dict = self._pickup_params_dict(self.params_dict)

        blob_min = self.params_dict.get('_blob_min')
        if is_ok:
            blob_min *= self.params_dict.get('_blob_margin')

        img_dir = self.ok_dir if is_ok else self.ng_dir

        score = 0
        files = glob.glob(f'{img_dir}/**/*.{self.img_ext}', recursive=True)
        for file_cnt, filename in enumerate(files):
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            img_h = img.shape[0]
            img_w = img.shape[1]

            _, res, _, img_out = blob_analysis(img,
                                               mode='hsv',
                                               inverse_mask=True,
                                               blob_min=blob_min,
                                               output_rec=True,
                                               **_params_dict)

            if is_ok:
                # ブロブ数＋ブロブ面積の総和をペナルティとする
                _num = len(res)
                _area = 0
                for _res in res:
                    _area += _res[0] / img_h / img_w

                score -= _num + _area * 100
            else:
                # NG判定できればプラスとする
                if len(res) > 0: score += 1

            if output_log:
                sub_dir = 'OK' if is_ok else 'NG'
                outfile = f'{self.out_dir}/{self.trial_cnt:05d}/{sub_dir}/{os.path.basename(filename)}'
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                cv2.imwrite(outfile, img_out)

        # ファイル数で規格化
        if len(files) > 0:
            score = score / len(files)
        else:
            score = 0

        return score

    def _calc_total_score(self):
        """
        OK/NGデータでそれぞれスコアを算出し、トータルスコアを算出する
        :return:
        """

        ok_score = self._calc_score(is_ok=True)
        ng_score = self._calc_score(is_ok=False)

        score = ok_score + ng_score

        self.res_dict = {"trial": self.trial_cnt,
                         "score": round(score, 3),
                         "ok": round(ok_score, 3),
                         "ng": round(ng_score, 3)}

        # 条件を満たす場合、処理後画像を出力するため、再処理を実施
        if self.trial_cnt is not None and self.trial_cnt % 100 == 0:
            self._calc_score(is_ok=False, output_log=True)
        elif score > self.best_score:
            self._calc_score(is_ok=False, output_log=True)
        else:
            pass

        return score

    def _save_results(self):
        """
        結果をcsvファイルに出力する
        :return:
        """

        _params = {"trial": self.trial_cnt,
                   "score": self.res_dict.get("score"),
                   "ok": self.res_dict.get("ok"),
                   "ng": self.res_dict.get("ng"),
                   "memo": self.memo
                   }
        _params.update(self.params_dict)

        # ヘッダ出力
        if not os.path.isfile(self.res_file):
            fw = open(self.res_file, 'w')
            csvWriter = csv.writer(fw, lineterminator='\n')

            header = list(_params.keys())
            csvWriter.writerow(header)
            fw.close()

        # 結果出力
        fw = open(self.res_file, 'a')
        csvWriter = csv.writer(fw, lineterminator='\n')

        vals = list(_params.values())
        csvWriter.writerow(vals)
        fw.close()

    def run(self, trial_num, sql_db=None):

        # 結果ファイル等を初期化
        if os.path.isfile(self.res_file):
            os.remove(self.res_file)

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        _day_id = datetime.datetime.now()
        _day_id = _day_id.strftime('%Y%m%d_%H%M%S')

        self.memo = 'test'

        # 最適化定義
        self.trial_cnt = 0
        sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'{_day_id}_{self.memo}',
            storage=sql_db
        )

        # 初期値定義
        self.study.enqueue_trial({"h_min": 0,
                                  "h_max": 180,
                                  "s_min": 50,
                                  "s_max": 170,
                                  "v_min": 0,
                                  "v_max": 250})

        # 実行
        self.study.optimize(self._objective, n_trials=trial_num)


if __name__ == '__main__':
    app = OptunaApp(ok_dir='./image/good',
                    ng_dir='./image/NG',
                    img_ext='png',
                    out_file='optuna_results.csv')

    app.run(trial_num=2000,
            sql_db='sqlite:///opencv.db')
