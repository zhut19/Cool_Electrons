#######################################
# Testing Blocks for Treemakers
# by Tianyu tz2263@columbia.edu, Feb 2018
#######################################
# This class only test if the treemaker works or not
# This should not be used to mass porduce minitrees
########################################
import os, sys, io, time
import numpy as np
from multihist import Histdd, Hist1d
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, minimize
pd.options.mode.chained_assignment = None        # default='warn'
import warnings
warnings.filterwarnings('ignore')

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

from numpy import sqrt, exp, pi, square
from tqdm import tqdm
import hax

class Treemaker_Testing():
    '''A common testing ground for treemakers
       Just use a single run to do it
    '''
    __version__ = '0.1.0'
    
    def __init__(self, treemakers):
        self.hax_init()
        
        if isinstance(treemakers, list):
            self.treemakers = treemakers
        else:
            self.treemakers = [treemakers]
        
        self.run_dict = dict(sciencerun0 = dict(Kr83m = '161114_1039', 
                                                none = '161122_2118', 
                                                AmBe = '161021_1514', 
                                                Rn220 = '161229_1111',),
                             sciencerun1 = dict(Kr83m = '170206_1255', 
                                                none = '170202_1747', 
                                                AmBe = '170316_0834', 
                                                Rn220 = '170313_1817', 
                                                neutron_generator = '170524_1435',
                                                Cs137 = '170404_1815',)
                            )
        
    def start_testing(self, sciencerun = 'sciencerun1', source_type = 'none', event_list_file = False, nevent_test = False):
        self.event_list_file = event_list_file
        
        if isinstance(nevent_test, int):
            tmp = pd.DataFrame(dict(event_number = list(range(nevent_test))))
            tmp.to_pickle(os.path.join(os.getcwd(), 'tmp.pkl'))
            self.event_list_file = os.path.join(os.getcwd(), 'tmp.pkl')
        
        if self.event_list_file:
            self.event_list = self.get_event_list(self.event_list_file)
        else:
            self.event_list = None
            
        self.test_run = self.run_dict[sciencerun][source_type]
        
        df, cut_history = hax.minitrees.load_single_dataset(run_id = self.test_run,
                                                            treemakers = self.treemakers,
                                                            preselection=None,
                                                            force_reload=False,
                                                            event_list = self.event_list,
                                                            )
        if isinstance(nevent_test, int): os.remove(os.path.join(os.getcwd(), 'tmp.pkl'))
        return df
    
    def hax_init(self, force_reload = False,
                 main_data_paths = ['/project2/lgrandi/xenon1t/processed/pax_v6.8.0'],
                 minitree_paths = ['/scratch/midway2/zhut/data/SingleScatter/data/minitrees']):

        if (not len(hax.config)) or force_reload:
            print ('Initiating hax, it takes some time...')
            # Initiate hax if not done already
            hax.init(experiment = 'XENON1T',
                     main_data_paths = main_data_paths,
                     minitree_paths = minitree_paths,
                     version_policy = 'loose',
                     make_minitrees = True,
                     minitree_caching = False, # We don't really use root anymore
                     pax_version_policy = '6.8.0',
                     tqdm_on = False,
                    )
            
    
    def get_event_list(self, event_list_file):
        return pd.read_pickle(event_list_file).event_number.values

#### Main ####
if __name__ == '__main__':
    ''' Using argv
        Example python treemaker_testing.py [path] [module] [treename]
    '''
    if len(sys.argv) > 1:
        argv = sys.argv
    else:
        exit(0)
    
    sys.path.append(argv[1])
    mo = __import__(argv[2])
    t = getattr(mo, argv[3])
    
    tt = Treemaker_Testing(t)
    df = tt.start_testing(sciencerun = 'sciencerun1', 
                          source_type = 'Rn220', 
                          event_list_file = False, 
                          nevent_test = 10)

    print(df.head(5))