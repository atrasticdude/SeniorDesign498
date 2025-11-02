import numpy as np
from scipy.signal import medfilt
from scipy import interpolate
from utils.helperfunction import BolusArrivalTime1D


class API(object):
    def __init__(self,dsa,fraction = 0.1):
        self.dsa = dsa
        self.x =self.x_time(dsa)
        self._x_inter = np.arange(0, np.max(self.x), 0.1)


    @staticmethod
    def y_concentration(dsa, roi = None):
        tdc = dsa[0][None, :, :] - dsa
        if roi is not None:
            tdc_roi = tdc.copy()
            mask = roi != 0
            tdc_roi[:, ~mask] = 0
            return tdc_roi
        return tdc

    @staticmethod
    def x_time(dsa):
        try:

            delta_t = np.asarray(dsa.FrameTimeVector, dtype=np.float32) / 1000
            if delta_t.size > 1 and delta_t[1] < 0.05:
                delta_t[1] = max(delta_t[1], 0.05)
        except AttributeError:
            n_frames = dsa.NumberOfFrames
            frame_rate = getattr(dsa, "RecommendedDisplayFrameRate", 10)
            delta_t = np.full(n_frames, 1 / frame_rate, dtype=np.float32)
            delta_t[0] = 0.0
        time_vector = np.cumsum(delta_t)
        return time_vector

    def time_density_curve(self,x,y):
        tdc_filtered = medfilt(y.astype(np.float32), kernel_size=1)
        baseline = tdc_filtered[0] * np.ones_like(tdc_filtered)
        tdc_inv = baseline - tdc_filtered
        tdc_inv[tdc_inv < 0] = 0
        f = interpolate.interp1d(x, tdc_filtered, kind='cubic', fill_value='extrapolate')
        y_interp = f(self.x_inter)
        return y_interp




    def API_Parameters(self,y, x = None):
        if x is None:
            x = self._x_inter

        if len(x) == 0 or len(y) == 0:
            raise Exception("Concentration or Time vector is empty")
        api = {}
        p_h = np.max(y, axis=0)
        api["PH"] = p_h
        p_h_i = np.argmax(y, axis=0)

        bai = BolusArrivalTime1D(y)
        if bai >= 0:
            ttp = x[p_h_i]
            api["TTP"] = ttp
        else:
            ttp = None
            api["TimeToPeak"] = ttp

        AUC_full = np.trapezoid(y, x)
        api["AUC"] = AUC_full

        half_max = p_h / 2.2
        mtt_index = np.where(y >= half_max)[0]
        mtt = None
        if len(mtt_index) > 0:
            start_mtt = x[mtt_index[0]]
            end_mtt = x[mtt_index[-1]]
            mtt = end_mtt - start_mtt
            api["MTT"] = mtt
        else:
            api["MTT"] =  mtt

        auc_interval = []
        multi = [0.5,1.0,1.0,2.0]
        if mtt and bai >= 0:
            bat_time = x[bai]
            for m in multi:
                end_t = bat_time + mtt * m
                end_index = np.searchsorted(x, end_t, side='right')
                auc = np.trapezoid(y[bai:min(end_index,len(x)),x[bai:min(end_index,len(x))]])
                # auc = np.trapezoid(y[bai:end_index], x[bai:end_index])
                auc_interval.append(auc)
            api["AUC_interval"] = auc_interval
        else:
            api["AUC_interval"] = None

        max_df = None
        if bai >= 0 and p_h_i > bai:
            dx = np.gradient(y[bai:p_h_i + 1])
            max_df = np.max(np.abs(dx))
            api["Max_Df"]  = max_df
        else:
            api["Max_Df"] = max_df

        if bai >= 0:
            api["BAT"] = x[bai]
        else:
            api["BAT"] = None

        return api



