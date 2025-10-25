import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

class API(object):
    def __init__(self,dsa,roi):
        self.dsa = dsa
        if roi:
            self.y = self.y_concentration(self.dsa,roi)
        else:
            self.y = self.y_concentration(self.dsa)
        self.x = self.x_time(self.dsa)


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



    def API_Parameters(self,x,y):
        if len(x) == 0 or len(y) == 0:
            raise Exception("Concentration or Time vector is empty")
        api = {}
        p_h = np.max(y, axis=0)
        api["PeakHeight"] = p_h
        p_h_i = np.argmax(y, axis=0)












