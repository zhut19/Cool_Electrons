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
    '''Generic production process class
       Have build in checking before production
    '''

    __version__ = '0.1.0'

    def process(self, in_directory, out_directory, production_id):
        self.in_directory = in_directory
        self.out_directory = out_directory
        self.production_id = production_id
        flag = self.check()
        if not flag: self._process()

    def _process(self):
        raise NotImplementedError()

    def hax_init(self, force_reload = False,
                 main_data_paths = ['/project2/lgrandi/xenon1t/processed/pax_v6.8.0'],
                 minitree_paths = ['/scratch/midway2/zhut/data/SingleScatter/data/minitrees',
                                   '/project2/lgrandi/xenon1t/minitrees/pax_v6.8.0']):

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
        self.process_list = [PickleEventList(),
                             BuildMiniTree_LargeS2(),
                             BuildMiniTree_PreTriggerPeaks(),
                             FindPreviousLargeS2(),
                            ]

    def process(self, in_directory, out_directory, production_id):
        for process in self.process_list:
            process.process(in_directory, out_directory, production_id)

from lax.lichens import sciencerun1
DAQVeto = sciencerun1.DAQVeto()
S2Ss = sciencerun1.S2SingleScatter()
class PickleEventList(ProductionProcess):
    def process(self, in_directory, out_directory, production_id):
        self.in_directory = in_directory
        self.out_directory = os.path.join(out_directory,'../Elist')
        self.production_id = production_id
        flag = self.check()
        if not flag: self._process()

    def _process(self):
        self.hax_init(force_reload = False)
        df, _ = hax.minitrees.load_single_dataset(self.production_id[:11],
            ['Fundamentals', 'Corrections', 'Basics', 'LargestPeakProperties', 'Proximity', 'FlashIdentification'])

        df = DAQVeto.process(df)
        slist = ['next_', 'nearest_', 'lone_hit', 'unknown', 'range_90p_area', 'index', '_on', '_off', 'flashing',
            '_n_', 'pe_event', 'Cut', 'largest_', 'correction', '_3d_', '_observed','cs1', 'cs2', '_x', '_y',
            '_area', 'hit_time_std','_pax']
        klist = ['flashing_time', 'CutDAQVeto', 'CurS2SingleScatter', 'nearest_busy']
        col_selection = lambda col: all(list(map(lambda s:s not in col, slist))) or any(list(map(lambda s:s in col, klist)))
        sel = list(map(col_selection, df.columns))
        df = df.loc[:, df.columns[sel]]
        df.to_pickle(os.path.join(self.out_directory, '{name}.pkl'.format(name = self.production_id)))

    def check(self):
        return False # Since is cost so little time (2sec) to update, let's do it all the time.
        return os.path.isfile \
        ('{outdir}/{production_id}.pkl'.format(outdir = self.out_directory, production_id = self.production_id))


# Import treemakers
sys.path.append('/home/zhut/Cool_Electrons/data_processing_modules/scripts/')
import puni, peak_extraction
from puni import Saplings, get_largest_indices
from peak_extraction import S2TrailingPeaks, PreTriggerPeaks, LargeS2

class BuildMiniTree_LargeS2(ProductionProcess):
    '''Run hax to create minitree'''

    __version__ = '0.0.2'

    def process(self, in_directory, out_directory, production_id):
        self.in_directory = in_directory
        self.out_directory = os.path.join(out_directory,'../LargeS2')
        self.production_id = production_id
        flag = self.check()
        if not flag: self._process()

    def _process(self):
        self.hax_init(force_reload = False)
        self.event_list = pd.read_pickle(os.path.join(self.in_directory, '{name}.pkl'.format(name = self.production_id)))
        self.event_list = self.event_list.loc[:,'event_number'].values

        with self._divert_stdout():
            self.df, cut_history = hax.minitrees.load_single_dataset(run_id = self.production_id,
                                                                     treemakers = [LargeS2],
                                                                     preselection=None,
                                                                     force_reload=False,
                                                                     event_list = self.event_list
                                                                    )
            self.df.to_pickle(os.path.join(self.out_directory, '{name}.pkl'.format(name = self.production_id)))
            print ('{production_id} minitrees building success :)'.format(production_id = self.production_id))

    def check(self):
        return os.path.isfile \
        ('{outdir}/{production_id}.pkl'.format(outdir = self.out_directory, production_id = self.production_id))


class BuildMiniTree_PreTriggerPeaks(ProductionProcess):
    '''Run hax to create minitree'''

    __version__ = '0.0.1'

    def _process(self):
        self.hax_init(force_reload = False)
        self.event_list = pd.read_pickle(os.path.join(self.in_directory, '{name}.pkl'.format(name = self.production_id)))
        ####
        self.event_list = self.event_list[self.event_list.eval('CutDAQVeto')]
        self.event_list = self.event_list.loc[:,'event_number'].values

        with self._divert_stdout():
            self.df, cut_history = hax.minitrees.load_single_dataset(run_id = self.production_id,
                                                                     treemakers = [PreTriggerPeaks],
                                                                     preselection=None,
                                                                     force_reload=False,
                                                                     event_list = self.event_list
                                                                    )
            self.df.to_pickle(os.path.join(self.out_directory, '{name}.pkl'.format(name = self.production_id)))
            print ('{production_id} minitrees building success :)'.format(production_id = self.production_id))

    def check(self):
        return os.path.isfile \
        ('{outdir}/{production_id}.pkl'.format(outdir = self.out_directory, production_id = self.production_id))


from os.path import isfile, join
from pax.utils import Memoize

class FindPreviousLargeS2(ProductionProcess):
    '''Use LargeS2 to find previous large s2 for each peak in PreTriggerPeaks minitree'''

    __version__ = '0.0.1'

    def process(self, in_directory, out_directory, production_id):
        self.indir = dict(
            larges2 = os.path.join(out_directory,'../LargeS2'),
            peak = os.path.join(out_directory,'../PreTriggerPeaks'),
            elist = os.path.join(out_directory,'../Elist'),)
        
        self.outdir = dict(
            larges2 = os.path.join(out_directory,'../LargeS2'),
            peak = os.path.join(out_directory,'../Peaks_Processed'),
            elist = os.path.join(out_directory,'../Elist_Processed'),)

        self.production_id = production_id
        self._process()

    def _process(self):
        self._process_list = [
            self._check_existance,
            self._read_pickle,
            self._calculations,
            self._merge,
            self._reduce_columns,
            self._write,]

        rolling_kwarg = dict(file = '{name}.pkl'.format(name = self.production_id))

        # Loop over stages of sub-processing
        for _proc in self._process_list:
            rolling_kwarg = (_proc(**rolling_kwarg))
            print(_proc.__name__, rolling_kwarg['flag'])
            if all(rolling_kwarg['flag'].values()):
                break

    def _check_existance(self, file):
        ans, flag = dict(), dict()

        for key in self.outdir.keys():
            if isfile(join(self.outdir[key], file)):
                flag.update({key:True})
                ans.update({key:pd.read_pickle(join(self.outdir[key], file))})
            else:
                flag.update({key:False})

        return dict(file = file, ans = ans, flag = flag)

    def _read_pickle(self, file, ans, flag):
        for key in ['peak', 'elist']:
            if not flag[key]:
                ans.update({key:pd.read_pickle(join(self.indir[key], file))})
        return dict(file = file, ans = ans, flag = flag)

    def _calculations(self, file, ans, flag):
        @Memoize
        def get_previous_s2_area(event_number):
            if event_number == 0: return [0]
            event_number -= 1
            areas = ans['larges2'].loc[ans['larges2'].event_number == event_number].area.values
            if len(areas) > 0: return areas
            else: return get_previous_s2_area(event_number)

        @Memoize
        def get_previous_s2_time(event_number):
            if event_number == 0: return [0]
            event_number -= 1
            times = ans['larges2'].loc[ans['larges2'].event_number == event_number].hit_time_mean_global.values
            if len(times) > 0: return times
            else: return get_previous_s2_time(event_number)

        for key in ['peak', 'elist']:
            if not flag[key]:
                ans[key]['previous_s2_areas'] = [get_previous_s2_area(n) for n in ans[key].event_number.values]
                ans[key]['previous_s2_times'] = [get_previous_s2_time(n) for n in ans[key].event_number.values]
                ans[key]['previous_s2_areas_sum'] = [np.sum(areas) for areas in ans[key].previous_s2_areas.values]
                ans[key]['previous_largest_s2_area'] = [np.amax(areas) for areas in ans[key].previous_s2_areas.values]
                ans[key]['previous_largest_s2_index'] = [np.argmax(areas) for areas in ans[key].previous_s2_areas.values]
                ans[key]['previous_largest_s2_time'] = [ans[key].previous_s2_times.values[ix][iy] for ix, iy in enumerate(ans[key].previous_largest_s2_index)]

        return dict(file = file, ans = ans, flag = flag)

    def _merge(self, file, ans, flag):
        cols_to_merge = ['event_number', 'nearest_busy', 'previous_busy', 
            'previous_event', 'previous_hev', 'previous_muon_veto_trigger',
            'inside_flash', 'nearest_flash']
        ans['peak'] = ans['peak'].merge(ans['elist'].loc[:,cols_to_merge],
            left_on='event_number', right_on='event_number', how='left')
        return dict(file = file, ans = ans, flag = flag)

    def _reduce_columns(self, file, ans, flag):
        remain_columns = dict(elist = [
            'run_number', 'event_number', 'event_time', 'event_duration', 'CutDAQVeto',
            'previous_busy', 'previous_event', 'previous_hev', 'previous_muon_veto_trigger',
            'flashing_time', 'flashing_width', 'inside_flash', 'nearest_flash', 'nearest_busy',
            's1', 's1_center_time', 's2', 's2_center_time', 'drift_time','x_pax', 'y_pax', 'z',
            'previous_largest_s2_area', 'previous_s2_areas', 'previous_s2_times', 'previous_largest_s2_time'],
                             )
        for key in remain_columns.keys():
            if not flag[key]:
                ans[key] = ans[key].loc[:, remain_columns[key]]

        return dict(file = file, ans = ans, flag = flag)

    def _write(self, file, ans, flag):
        for key in ['peak', 'elist']:
            if not flag[key]:
                ans[key].to_pickle(join(self.outdir[key], file))
                flag[key] = True

        return dict(file = file, ans = ans, flag = flag)

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