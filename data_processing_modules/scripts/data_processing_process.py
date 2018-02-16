#######################################
# Data Processing Processes
# by Tianyu tz2263@columbia.edu, Feb 2018
# Changed from Fax Production Processes
########################################
# Run as __main__ with
# python data_processing_process.py {out_directory} {production_list_file}
# python data_processing_process.py {out_directory} {production_id}
#
# will initiate and run RunAllProcess() and RunAllProcess.process()
########################################

import sys, os
from configparser import ConfigParser
from contextlib import contextmanager
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import hax

class ProductionProcess():
    '''Generic production process class'''

    __version__ = '0.0.1'

    def process(self, in_directory, out_directory, production_id):
        self.in_directory = in_directory
        self.out_directory = out_directory
        self.production_id = production_id
        flag = self.check()
        if not flag: self._process()        

    def _process(self):
        raise NotImplementedError()

    @contextmanager
    def _divert_stdout(self):
        # Dangerous to use but works very nice if all bugs are fixed.
        # Basically works as 1&>${log}
        self.productionlog_path = os.path.join(self.out_directory, 'processing_logs')
        if not os.path.exists(self.productionlog_path): os.makedirs(self.productionlog_path)
        self.productionlog = open('{path}/{name}.log'.format(path = self.productionlog_path, name = self.production_id), 'a+')

        global saved_stdout, saved_stderr
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        sys.stdout = self.productionlog
        sys.stderr = self.productionlog
        print(saved_stdout)
        print(saved_stderr)
        yield
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        self.productionlog.close()

class RunAllProcess(ProductionProcess):
    '''Go through all processes'''

    __version__ = '0.0.1'

    def __init__(self):
        self.process_list = [BuildMiniTree(),
                            ]

    def process(self, in_directory, out_directory, production_id):
        for process in self.process_list:
            process.process(in_directory, out_directory, production_id)

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

class BuildMiniTree(ProductionProcess):
    '''Run hax to create minitree'''

    __version__ = '0.0.1'

    def hax_init(self, force_reload = False,
                 main_data_paths = ['/project2/lgrandi/xenon1t/processed/pax_v6.8.0'],
                 minitree_paths = ['/scratch/midway2/zhut/data/SingleScatter/data/minitrees']):
        
        # Trick learned from Daniel Coderre's DAQVeto lichen
        if (not len(hax.config)) or force_reload:
            # User didn't init hax yet... let's do it now
            hax.init(experiment = 'XENON1T',
                     main_data_paths = main_data_paths,
                     minitree_paths = minitree_paths,
                     version_policy = 'loose',
                     make_minitrees = True,
                     minitree_caching = False,
                     pax_version_policy = '6.8.0',
                     tqdm_on = True,
                    )

    def _process(self):
        self.hax_init(force_reload = False)
        self.event_list = pd.read_pickle(os.path.join(self.in_directory, '{name}.pkl'.format(name = self.production_id)))
        self.event_list = self.event_list.loc[self.event_list.eval('CutPreS2Junk'), 'event_number'].values

        with self._divert_stdout():
            self.df, cut_history = hax.minitrees.load_single_dataset(run_id = self.production_id,
                                                                     treemakers = [Saplings],
                                                                     preselection=None,
                                                                     force_reload=False,
                                                                     event_list = self.event_list
                                                                    )
            self.df.to_pickle(os.path.join(self.out_directory, '{name}.pkl'.format(name = self.production_id)))
            print ('{production_id} minitrees building success :)'.format(production_id = self.production_id))

    def check(self):
        return os.path.isfile \
        ('{outdir}/{production_id}.pkl'.format(outdir = self.out_directory, production_id = self.production_id))

# When called upon, run RunAllProcess.process
if __name__ == '__main__':
    arg = sys.argv
    if not (len (arg) == 4):
        sys.exit(1)

    in_directory = arg[1]
    out_directory = arg[2]
    production_list_file = arg[3]
    
    # Process a list of productions
    if 'txt' in production_list_file:
        errors = []
        production_list = np.array(np.genfromtxt(production_list_file, dtype='<U32'), ndmin=1)
        for prod in production_list:
            try:
                production_process = RunAllProcess()
                production_process.process(in_directory = in_directory, out_directory = out_directory, production_id = prod)
            except Exception as e:
                err = dict(production_id = prod,
                           error_type = e.__class__.__name__,
                           error_message = str(e),
                          )
                print (str(e))
                errors.append(err)

        print ('Enconter %d exceptions' %len(errors))
        if len(errors) != 0:
            errors = pd.DataFrame(errors)
            errors.to_pickle(production_list_file[:-11]+'err_msg.pkl')

    # Process a single prodution
    else:
        prod = production_list_file
        production_process = RunAllProcess()
        production_process.process(in_directory = in_directory, out_directory = out_directory, production_id = prod)