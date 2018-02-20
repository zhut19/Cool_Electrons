from hax.minitrees import TreeMaker
from hax.paxroot import loop_over_dataset, function_results_datasets
from collections import defaultdict

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

class S2TrailingPeaks(TreeMaker):
    __version__ = '0.1.2'
    extra_branches = ['peaks.*']
    uses_array = False
    stop_after = np.inf

    peak_fields = ['type', 'area', 'area_fraction_top', 
                   'range_50p_area', 'range_80p_area', 'range_90p_area','rise_time',
                   'hit_time_mean', 'x','y','pattern_fit']
    event_cut_list = ['True']
    peak_cut_list = ['({obj}.type != "lone_hit")',
                     '({obj}.type != "unknown")',
                     '({obj}.detector == "tpc")',
                     '({obj}.hit_time_mean > s2_hit_time_mean)'
                    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peaktypes = {'unknown': 0, 's1': 1, 's2': 2, 'lone_hit': 3}

    def extract_data(self, event):

        if event.event_number == self.stop_after:
            raise hax.paxroot.StopEventLoop()

        if len(event.peaks) == 0:
            return [], dict()

        peak_data_list = []

        if len(event.interactions):
            interaction = event.interactions[0]
            s1 = event.peaks[interaction.s1]
            s2 = event.peaks[interaction.s2]
            s2_hit_time_mean = s2.hit_time_mean
        else:
            s2_hit_time_mean = 0

        # Loop over peaks
        for ix, peak in enumerate(event.peaks):

            if eval('&'.join(self.peak_cut_list).format(obj = 'peak')):

                # Basics
                current_peak = dict(type = peak.type,
                                    area = peak.area,
                                    area_fraction_top = peak.area_fraction_top,  
                                    hit_time_mean = peak.hit_time_mean,
                                    hit_time_mean_global = peak.hit_time_mean + event.start_time,
                                    rise_time = - peak.area_decile_from_midpoint[1],
                                    range_50p_area = list(peak.range_area_decile)[5],
                                    range_80p_area = list(peak.range_area_decile)[8],
                                    range_90p_area = list(peak.range_area_decile)[9],
                                   )

                # Position reconstruction information
                for rp in peak.reconstructed_positions:
                    if rp.algorithm == 'PosRecTopPatternFit':
                         current_peak.update(dict(x_tpf = rp.x, y_tpf = rp.x, goodness_of_fit_tpf = rp.goodness_of_fit))
                    if rp.algorithm == 'PosRecNeuralNet':
                         current_peak.update(dict(x_nn = rp.x, y_nn = rp.x, goodness_of_fit_nn = rp.goodness_of_fit))

                # Additional interaction dependent information
                if len(event.interactions):
                    current_peak.update(dict(delay_main_s1 = peak.hit_time_mean - s1.hit_time_mean,
                                             delay_main_s2 = peak.hit_time_mean - s2.hit_time_mean))

                peak_data_list.append(current_peak)

        return peak_data_list

    def process_event(self, event):
        result = self.extract_data(event)
        if not isinstance(result, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result))

        # Add the run and event_number to the result. This is required to make joins succeed later on.
        for i in range(len(result)):
            result[i]['run_number'] = self.run_number
            result[i]['event_number'] = event.event_number
            result[i]['global_event_number'] = self.run_number * 1000000 + event.event_number # id: rrrrreeeeee
        assert len(result) == 0 or isinstance(result[0], dict)
        self.cache.extend(result)
        self.check_cache(force_empty=False) # Default MultipleRowExtractor force_empty=True


class PreTriggerPeaks(TreeMaker):
    __version__ = '0.1.0'
    extra_branches = ['peaks.*']
    uses_array = False
    stop_after = np.inf

    peak_fields = ['type', 'area', 'area_fraction_top', 
                   'range_50p_area', 'range_80p_area', 'range_90p_area','rise_time',
                   'hit_time_mean', 'x','y','pattern_fit']
    event_cut_list = ['True']
    peak_cut_list = ['({obj}.type != "lone_hit")',
                     '({obj}.type != "unknown")',
                     '({obj}.detector == "tpc")',
                     '({obj}.hit_time_mean < event_trigger_time)'
                    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peaktypes = {'unknown': 0, 's1': 1, 's2': 2, 'lone_hit': 3}

    def extract_data(self, event):

        if event.event_number == self.stop_after:
            raise hax.paxroot.StopEventLoop()

        if len(event.peaks) == 0:
            return [], dict()

        peak_data_list = []
        event_data = dict()

        if len(event.interactions):
            interaction = event.interactions[0]
            s1 = event.peaks[interaction.s1]
            s2 = event.peaks[interaction.s2]
            s2_hit_time_mean = s2.hit_time_mean

            # Event data
            event_data = dict(s1_hit_time_mean_global = s1.hit_time_mean + event.start_time,
                              s2_hit_time_mean_global = s2.hit_time_mean + event.start_time,
                              event_start_time = event.start_time,
                              event_stop_time = event.stop_time)


        # Loop over peaks
        for ix, peak in enumerate(event.peaks):

            if eval('&'.join(self.peak_cut_list).format(obj = 'peak')):

                # Basics
                current_peak = dict(type = peak.type,
                                    area = peak.area,
                                    area_fraction_top = peak.area_fraction_top,  
                                    hit_time_mean = peak.hit_time_mean,
                                    hit_time_mean_global = peak.hit_time_mean + event.start_time,
                                    rise_time = - peak.area_decile_from_midpoint[1],
                                    range_50p_area = list(peak.range_area_decile)[5],
                                    range_80p_area = list(peak.range_area_decile)[8],
                                    range_90p_area = list(peak.range_area_decile)[9],
                                   )

                # Position reconstruction information
                for rp in peak.reconstructed_positions:
                    if rp.algorithm == 'PosRecTopPatternFit':
                         current_peak.update(dict(x_tpf = rp.x, y_tpf = rp.x, goodness_of_fit_tpf = rp.goodness_of_fit))
                    if rp.algorithm == 'PosRecNeuralNet':
                         current_peak.update(dict(x_nn = rp.x, y_nn = rp.x, goodness_of_fit_nn = rp.goodness_of_fit))

                # Additional interaction dependent information
                if len(event.interactions):
                    current_peak.update(dict(delay_main_s1 = peak.hit_time_mean - s1.hit_time_mean,
                                             delay_main_s2 = peak.hit_time_mean - s2.hit_time_mean))

                peak_data_list.append(current_peak)

        return peak_data_list

    def process_event(self, event):
        result = self.extract_data(event)
        if not isinstance(result, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result))

        # Add the run and event_number to the result. This is required to make joins succeed later on.
        for i in range(len(result)):
            result[i]['event_number'] = event.event_number
        assert len(result) == 0 or isinstance(result[0], dict)
        self.cache.extend(result)
        self.check_cache(force_empty=False) # Default MultipleRowExtractor force_empty=True