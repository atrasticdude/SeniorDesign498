
from scipy.signal import correlate
from utils.API import API
import numpy as np



class getAPI(API):
    def __init__(self,dsa,mask,inlet,dsa_temp,frac = 0.1, show_mask_stats = False):
        super().__init__(dsa,dsa_temp)
        self.mask = mask.astype(bool)
        self.inlet = inlet.astype(bool)
        self.threshold_fraction = frac
        self.time = self.x_time(dsa_temp,dsa.shape[0])
        self.inlet_tdc_average = self.get_tdc_inl_avg()
        self.inlet_tdc_inlet = self.time_density_curve(self._x,self.inlet_tdc_average)
        self.inlet_parameters = self.API_Parameters(self.inlet_tdc_inlet)
        if show_mask_stats:
           self.results_mean, self.results_std, self.qualifying_pixels, self.qualifying_indices, self.tdc_average = self.process_mask()




    def process_mask(self):

        mask_y,mask_x = np.where(self.mask != 0)
        tdc_pixels = self.dsa[:,mask_y, mask_x]

        y_interp = np.apply_along_axis(lambda y: self.time_density_curve(self._x,y),0,tdc_pixels)
        max_change = np.max(y_interp, axis = 0)
        max_inlet_change = np.max(self.inlet_tdc_inlet)
        max_change_factor = self.threshold_fraction * max_inlet_change
        y_valid= y_interp[:,max_change >= max_change_factor]
        valid_mask = max_change >= self.threshold_fraction * max_inlet_change
        qualifying_indices = list(zip(mask_y[valid_mask], mask_x[valid_mask]))
        qualifying_pixels = y_valid.shape[1]

        if qualifying_pixels == 0:
            print("No qualifying pixels after threshold")
            return (np.nan,np.nan,0,[],None)


        eps = 1e-8
        inlet_corr = self.inlet_tdc_inlet - np.mean(self.inlet_tdc_inlet)
        y_valid_corr = y_valid - np.mean(y_valid, axis = 0)
        corr = np.array([
            np.max(
                correlate(inlet_corr, y_valid_corr[:, i]) /
                (len(y_valid_corr[:, i]) * (np.std(inlet_corr) + eps) * (np.std(y_valid_corr[:, i]) + eps))
            )
            for i in range(y_valid_corr.shape[1])
        ])

        # parameters = np.array([self.API_Parameters(y_valid[:, i]) for i in range(qualifying_pixels)])
        # parameters = np.column_stack((corr, parameters))
        # tdc_average = np.mean(y_valid, axis=1)
        # results_mean = np.nanmean(parameters, axis=0)
        # results_std = np.nanstd(parameters, axis=0)
        #
        # Compute API parameters for each pixel
        parameters_list = []
        for i in range(qualifying_pixels):
            api_values = self.API_Parameters(y_valid[:, i])
            numeric_values = []
            for key in ["PH", "TTP", "AUC", "MTT", "Max_Df", "BAT"]:
                val = api_values.get(key)
                if val is None:
                    val = np.nan
                numeric_values.append(val)
            parameters_list.append(numeric_values)

        parameters = np.array(parameters_list)
        parameters = np.column_stack((corr, parameters))

        tdc_average = np.mean(y_valid, axis=1)
        results_mean = np.nanmean(parameters, axis=0)
        results_std = np.nanstd(parameters, axis=0)

        return results_mean, results_std, qualifying_pixels, qualifying_indices, tdc_average



    def get_tdc_inl_avg(self):
        tdc_inlet_average = self.dsa[:, self.inlet.astype(bool)].mean(axis=1)
        return tdc_inlet_average
    def get_inlet_API(self):
        return self.inlet_parameters










