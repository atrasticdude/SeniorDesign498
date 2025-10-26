import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy import interpolate


class API(object):
    def __init__(self,dsa,roi):
        self.dsa = dsa
        if roi:
            self.y = self.y_concentration(self.dsa,roi)
        else:
            self.y = self.y_concentration(self.dsa)
        self.x = self.x_time(self.dsa)
        self.x_inter = np.arange(0, np.max(self.x),0.1)
        self.tdc = self.time_density_curve(self.x,self.y)


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
        f = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')
        y_interp = f(self.x_inter)
        return y_interp




    def API_Parameters(self,x,y):
        if len(x) == 0 or len(y) == 0:
            raise Exception("Concentration or Time vector is empty")
        api = {}
        p_h = np.max(y, axis=0)
        api["PeakHeight"] = p_h
        p_h_i = np.argmax(y, axis=0)












