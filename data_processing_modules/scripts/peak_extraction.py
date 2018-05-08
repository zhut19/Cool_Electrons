from hax.minitrees import TreeMaker
from hax.paxroot import loop_over_dataset, function_results_datasets
from collections import defaultdict

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

class S2TrailingPeaks(TreeMaker):
    ''' This tree maker extract peaks that happen after the main S2 or in absence of main S2 all peaks.
        We are in paticular interested in photo ionization of impurities in liquid xe.
    '''
    __version__ = '0.1.2'
    extra_branches = ['peaks.*']
    uses_array = False
    stop_after = np.inf

    peak_fields = [
        'type', 'area', 'area_fraction_top',
        'range_50p_area', 'range_80p_area', 'range_90p_area','rise_time',
        'hit_time_mean', 'x','y','pattern_fit']

    event_cut_list = ['True']
    peak_cut_list = [
        '({obj}.type != "lone_hit")',
        '({obj}.type != "unknown")',
        '({obj}.detector == "tpc")',
        '({obj}.hit_time_mean > s2_hit_time_mean)']

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
                current_peak = dict(
                    type=peak.type,
                    area=peak.area,
                    left=peak.left,
                    area_fraction_top=peak.area_fraction_top,
                    hit_time_mean=peak.hit_time_mean,
                    hit_time_mean_global=peak.hit_time_mean + event.start_time,
                    rise_time= - peak.area_decile_from_midpoint[1],
                    range_50p_area=list(peak.range_area_decile)[5],
                    range_80p_area=list(peak.range_area_decile)[8],
                    range_90p_area=list(peak.range_area_decile)[9])

                # Position reconstruction information
                for rp in peak.reconstructed_positions:
                    if rp.algorithm == 'PosRecTopPatternFit':
                         current_peak.update(dict(
                            x_tpf=rp.x,
                            y_tpf=rp.y,
                            goodness_of_fit_tpf=rp.goodness_of_fit))

                    if rp.algorithm == 'PosRecNeuralNet':
                         current_peak.update(dict(
                            x_nn=rp.x,
                            y_nn=rp.y,
                            goodness_of_fit_nn=rp.goodness_of_fit))

                # Additional interaction dependent information
                if len(event.interactions):
                    current_peak.update(dict(
                        delay_main_s1=peak.hit_time_mean - s1.hit_time_mean,
                        delay_main_s2=peak.hit_time_mean - s2.hit_time_mean))

                peak_data_list.append(current_peak)

        return peak_data_list

    def process_event(self, event):
        result = self.extract_data(event)
        if not isinstance(result, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result))

        # Add the run and event_number to the result. This is required to make joins succeed later on.
        for i, _ in enumerate(result):
            result[i]['run_number'] = self.run_number
            result[i]['event_number'] = event.event_number
            # global_event_number: rrrrreeeeee
            result[i]['global_event_number'] = self.run_number * 1000000 + event.event_number 
        assert len(result) == 0 or isinstance(result[0], dict)
        self.cache.extend(result)
        self.check_cache(force_empty=False) # Default MultipleRowExtractor force_empty=True




class PreTriggerPeaks(TreeMaker):
    ''' This would eventually be the tree-maker we are trying to build
        that would extract small S2s in background.
        Looking back a few events before the current event. 
    '''
    __version__ = '0.1.0'
    extra_branches = ['peaks.*']
    uses_array = False
    stop_after = np.inf

    event_cut_list = ['True']
    peak_cut_list = [
        '({obj}.type != "lone_hit")',
        '({obj}.type != "unknown")',
        '({obj}.detector == "tpc")',
        '({obj}.right * 10 < trigger_time)',
        '({obj}.hit_time_mean < s1_center_time)',
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peaktypes = {'unknown': 0, 's1': 1, 's2': 2, 'lone_hit': 3}

    def extract_data(self, event):

        if event.event_number == self.stop_after:
            raise hax.paxroot.StopEventLoop()

        peak_data_list = []
        if len(event.peaks) == 0:
            return peak_data_list

        if len(event.interactions):
            interaction = event.interactions[0]
            s1 = event.peaks[interaction.s1]
            s2 = event.peaks[interaction.s2]
            s1_center_time = s1.center_time
            s2_center_time = s2.center_time

            # Event data
            event_data = dict(s1_hit_time_mean_global=s1.hit_time_mean + event.start_time,
                              s2_hit_time_mean_global=s2.hit_time_mean + event.start_time,
                              event_start_time=event.start_time,
                              event_stop_time=event.stop_time)
        else:
            s1_center_time = 1000e3

        trigger_time = 1000e3
        peak_pool = [p for p in event.peaks if p.detector == 'tpc' and p.right * 10 < trigger_time]

        # Loop over peaks
        for ix, peak in enumerate(peak_pool):

            if eval('&'.join(self.peak_cut_list).format(obj = 'peak')):

                # Basics
                current_peak = dict(
                    type=peak.type,
                    area=peak.area,
                    area_fraction_top=peak.area_fraction_top,
                    hit_time_mean=peak.hit_time_mean,
                    hit_time_mean_global=peak.hit_time_mean + event.start_time,
                    rise_time= - peak.area_decile_from_midpoint[1],
                    range_50p_area=list(peak.range_area_decile)[5],
                    range_80p_area=list(peak.range_area_decile)[8],
                    range_90p_area=list(peak.range_area_decile)[9])

                # Position reconstruction information
                for rp in peak.reconstructed_positions:
                    if rp.algorithm == 'PosRecTopPatternFit':
                         current_peak.update(dict(
                            x_tpf=rp.x,
                            y_tpf=rp.y,
                            goodness_of_fit_tpf=rp.goodness_of_fit))

                    if rp.algorithm == 'PosRecNeuralNet':
                         current_peak.update(dict(
                            x_nn=rp.x,
                            y_nn=rp.y,
                            goodness_of_fit_nn=rp.goodness_of_fit))

                # Something worth checking
                current_peak.update(dict(sum_s1s_before = sum(
                    [p.area for p in peak_pool if p.type == 's1' and p.left < peak.left])))
                current_peak.update(dict(sum_s2s_before = sum(
                    [p.area for p in peak_pool if p.type == 's2' and p.left < peak.left])))

                peak_data_list.append(current_peak)

        return peak_data_list

    def process_event(self, event):
        result = self.extract_data(event)
        if not isinstance(result, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result))

        if len(result) > 0:
            # Add the run and event_number to the result. This is required to make joins succeed later on.
            for i, _ in enumerate(result):
                result[i]['run_number'] = self.run_number
                result[i]['event_number'] = event.event_number
                # global_event_number: rrrrreeeeee
                result[i]['global_event_number'] = self.run_number * 1000000 + event.event_number 
            assert len(result) == 0 or isinstance(result[0], dict)
            self.cache.extend(result)
            self.check_cache(force_empty=False) # Default MultipleRowExtractor force_empty=True


    def event_basic_data_extraction(self, event):
        event_data = dict()

        if not len(event.interactions):
            '''implement lone hit data extraction'''
            return result

        interaction = event.interactions[0]
        s1 = event.peaks[interaction.s1]
        s2 = event.peaks[interaction.s2]

        event_data.update(
            dict(
                s1=s1.area,
                s1_area_fraction_top=s1.area_fraction_top,
                s1_range_50p_area=list(s1.range_area_decile)[5],
                s1_hit_time_mean_global=s1.hit_time_mean + event.start_time,
                s2=s2.area,
                s2_area_fraction_top=s2.area_fraction_top,
                s2_range_50p_area=list(s1.range_area_decile)[5],
                s2_hit_time_mean_global=s2.hit_time_mean + event.start_time,
                x=interaction.x,
                y=interaction.y,
                z=interaction.z,
                drift_time=interaction.drift_time))

        largest_other_indices = get_largest_indices(event.peaks, exclude_indices=(interaction.s1, interaction.s2))
        largest_other_s2_index = largest_other_indices.get('s2', -1)
        if largest_other_s2_index != -1:
            pk = event.peaks[largest_other_s2_index]
            event_data['largest_other_s2']=pk.area

        main_s2_left = peaks[event.interactions[0].s2].left
        event_data['area_before_main_s2'] = sum(
            [pk.area for pk in peaks if p.detector == 'tpc' and p.left<main_s2_left])

        return result


class LargeS2(TreeMaker):
    ''' This tree maker extract S2 peaks large enough to be triggered regardless there is valid S1
        Now we only take out the largest three peaks
    '''
    __version__ = '0.1.3'
    extra_branches = ['peaks.*']
    uses_array = False
    stop_after = np.inf

    peak_cut_list = ['({obj}.type == "s2")',
        '({obj}.detector == "tpc")',
        '({obj}.area > 150)',
]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peaktype = {'unknown':0, 's1':1, 's2':2, 'lone_hit':3}

    def extract_data(self, event):

        if event.event_number == self.stop_after:
            raise hax.paxroot.StopEventLoop()
        
        area_list = [peak.area for peak in event.peaks if eval('&'.join(self.peak_cut_list).format(obj = 'peak'))]
        sort_result = np.argsort(area_list)[-3:][::-1] # Get the index of the largest three peaks
        sort_area = np.array(area_list)[sort_result]
        peak_data_list = [i for i in range(len(sort_area))]
        
        # Loop over peaks
        for ix, peak in enumerate(event.peaks):

            for iy, area in enumerate(sort_area):

                if peak.area == area:

                    # Basics
                    current_peak = dict(
                        area = peak.area,
                        area_fraction_top = peak.area_fraction_top,
                        range_50p_area=list(peak.range_area_decile)[5],
                        hit_time_mean=peak.hit_time_mean,
                        hit_time_mean_global=peak.hit_time_mean + event.start_time,)

                    # Position reconstruction information
                    for rp in peak.reconstructed_positions:
                        if rp.algorithm == 'PosRecNeuralNet':
                            current_peak.update(dict(
                                x_nn=rp.x,
                                y_nn=rp.y))

                    peak_data_list[iy] = current_peak

        return peak_data_list


    def process_event(self, event):
        result = self.extract_data(event)
        if not isinstance(result, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result))

        # Add the run and event_number to the result. This is required to make joins succeed later on.
        for i, _ in enumerate(result):
            result[i]['run_number'] = self.run_number
            result[i]['event_number'] = event.event_number
            # global_event_number: rrrrreeeeee
            result[i]['global_event_number'] = self.run_number * 1000000 + event.event_number 
        assert len(result) == 0 or isinstance(result[0], dict)
        self.cache.extend(result)
        self.check_cache(force_empty=False) # Default MultipleRowExtractor force_empty=True


def get_largest_indices(peaks, exclude_indices=tuple()):
    """Return a dic with the indices in peaks of the largest peak of each type (s1, s2, etc)
    excluding the inices in exclude_peak_indices from consideration
    """
    largest_area_of_type = defaultdict(float)
    largest_indices = dict()
    for i, p in enumerate(peaks):
        if i in exclude_indices:
            continue
        if p.detector == 'tpc':
            peak_type = p.type
        else:
            if p.type == 'lone_hit':
                peak_type = 'lone_hit_%s' % p.detector    # Will not be saved
            else:
                peak_type = p.detector
        if p.area > largest_area_of_type[peak_type]:
            largest_area_of_type[peak_type] = p.area
            largest_indices[peak_type] = i
    return largest_indices