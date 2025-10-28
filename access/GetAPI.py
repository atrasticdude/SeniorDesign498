from cffi.model import qualify
from scipy.signal import correlate

from utils.API import API
import numpy as np
from src.Load import Load
from utils.helperfunction import getindices


class getAPI(API):
    def __init__(self,dsa,mask,inlet,frac):
        super().__init__(dsa)
        self.mask = mask
        self.inlet = inlet
        self.threshold_fraction = frac
        self.time = self.x_time(self.dsa)
        self.get_tdc_inl_avg = self.get_tdc_inl_avg()
        self.tdc_inl_interp = self.time_density_curve(self.get_tdc_inl_avg)
        self.inlet_parameters = self.API_Parameters(self.tdc_inl_interp)

    def process_mask(self,inlet_tdc_inter):

        mask_y,mask_x = np.where(self.mask != 0)
        tdc_pixels = self.dsa[:,mask_y, mask_x]

        y_interp = np.apply_along_axis(lambda y: self.time_density_curve(self.time,y),0,tdc_pixels)
        max_change = np.max(y_interp, axis = 0)
        max_inlet_change = np.max(inlet_tdc_inter)
        max_change_factor = self.threshold_fraction * max_inlet_change
        y_valid= y_interp[:,max_change >= max_change_factor]
        valid_mask = max_change >= self.threshold_fraction * max_inlet_change
        qualifying_indices = list(zip(mask_y[valid_mask], mask_x[valid_mask]))
        qualifying_pixels = y_valid.shape[1]

        if qualifying_pixels == 0:
            print("No qualifying pixels after threshold")
            return None


        eps = 1e-8
        inlet_corr = inlet_tdc_inter - np.mean(inlet_tdc_inter)
        y_valid_corr = y_valid - np.mean(y_valid, axis = 0)
        corr = np.array([
            np.max(
                correlate(inlet_corr, y_valid_corr[:, i]) /
                (len(y_valid_corr[:, i]) * (np.std(inlet_corr) + eps) * (np.std(y_valid_corr[:, i]) + eps))
            )
            for i in range(y_valid_corr.shape[1])
        ])

        parameters = np.array([self.API_Parameters(y_valid[:, i]) for i in range(qualifying_pixels)])

        parameters = np.column_stack((corr, parameters))

        tdc_average = np.mean(y_valid, axis=1)
        results_mean = np.nanmean(parameters, axis=0)
        results_std = np.nanstd(parameters, axis=0)

        return results_mean, results_std, qualifying_pixels, qualifying_indices, tdc_average

    def get_tdc_inl_avg(self):
        tdc_inlet_average = self.dsa[:, self.inlet.astype(bool)].mean(axis=1)
        return tdc_inlet_average












