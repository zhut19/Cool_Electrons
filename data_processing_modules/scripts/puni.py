from hax.minitrees import TreeMaker
from hax import runs, cuts
from hax.paxroot import loop_over_dataset, function_results_datasets
from collections import defaultdict

class Saplings(TreeMaker):
    # Litte extra branches for extra selections
    """
    sum_s2s_before_main_s1: Sum up all the s2 peaks before main s1 peak
    s1_hit_time_mean: Hit time mena of main s1 peak
    largest_other_s2_area_fraction_top: Area fraction top of largest other s2
    """
    __version__ = '0.0.2'
    extra_branches = ['peaks.left','peaks.hit_time_mean','peaks.top_hitpattern_spread','peaks.area_fraction_top']

    def extract_data(self, event):
        result = dict()

        if not len(event.interactions):
            return result

        interaction = event.interactions[0]
        s1 = event.peaks[interaction.s1]
        s2 = event.peaks[interaction.s2]
        largest_other_indices = get_largest_indices(event.peaks, exclude_indices=(interaction.s1, interaction.s2))

        result['sum_s2s_before_main_s1'] = sum([p.area for p in event.peaks if p.type == 's2' and p.detector == 'tpc' and p.left < s1.left])
        result['s1_hit_time_mean'] = s1.hit_time_mean

        result['largest_other_s2_area_fraction_top'] = float('nan')

        largest_other_s2_index = largest_other_indices.get('s2', -1)
        if largest_other_s2_index != -1:
            pk = event.peaks[largest_other_s2_index]
            result['largest_other_s2_area_fraction_top'] = pk.area_fraction_top

        return result

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

