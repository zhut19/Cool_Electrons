# Use the event infomation of individual run to get the total live time

import os, sys
import json
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')


from numpy import sqrt, square, exp, log, log10, pi

from tqdm import tqdm
from multiprocessing import Pool
from numba import vectorize, guvectorize, int64, int32

io_dir = dict (
    indir_elist = '/project2/lgrandi/zhut/data/ce/Sciencerun1_None_CE/Elist/', # Elist In
    indir_larges2 = '/project2/lgrandi/zhut/data/ce/Sciencerun1_None_CE/LargeS2/', # Large S2 In
    indir_peak = '/project2/lgrandi/zhut/data/ce/Sciencerun1_None_CE/PreTriggerPeaks/', # Pre Trigger Peak In
    outdir_elist = '/project2/lgrandi/zhut/data/ce/Sciencerun1_None_CE/Elist_Processed/', # Processed Peak Out
    outdir_peak = '/project2/lgrandi/zhut/data/ce/Sciencerun1_None_CE/Peaks_Processed/',) # Processed Elist Out

time_range = [0, 1500]
time_bin_edges = np.linspace(time_range[0]*1e6, time_range[1]*1e6, 1500+1)
time_bin_centers = 0.5*(time_bin_edges[:-1] + time_bin_edges[1:])

area_range = [1e2, 1e8]
area_bin_edges = np.logspace(np.log10(area_range[0]),  np.log10(area_range[1]), 101)
area_bin_centers = sqrt(area_bin_edges[:-1]*area_bin_edges[1:])

# 1D live time calculation, delay from previous s2 as x axis
@vectorize([int64(int64, int64, int64, int64)])
def func_nb(a1, a2, b1, b2):
    # Calculate the over lap between two segments on an axis
    return min(a2, b2) - max(a1, b1)

def BinS2AreaAxis(df, D2lt):
    df['binned_lt'] = list(D2lt.T)
    df['area_index'] = np.digitize(df.previous_largest_s2_area_sum, area_bin_edges[1:])
    df.sort_values(by='area_index', inplace=True)

    uniarea, unindices = np.unique(df.area_index.values, return_index=True)
    # In the following
    # D3lt indicies: (Area, Sub Area, Delay)
    # d2lt indicies: (Sub area, Delay) -> we use it to collapse 'Sub area' index essentially events
    # D2lt indicies: (Area, Delay) -> we need to patch up missing ones

    D3lt = np.split(df.binned_lt.values, indices_or_sections = unindices)[1:-1]
    _D2lt = np.array([np.sum(d2lt, axis=0) for d2lt in D3lt])

    D2lt = np.zeros((len(time_bin_centers), len(area_bin_centers)))
    D2lt[:, uniarea[:-1]] += _D2lt.T
    return D2lt

def GetLiveTimeSingleFile(file):
    # A1: left x-axis bin edges, A2: right x-axis bin edges
    A1 = np.asarray(time_bin_edges[:-1], dtype=int)
    A2 = np.asarray(time_bin_edges[1:], dtype=int)

    df = pd.read_pickle(os.path.join(io_dir['outdir_elist'], file))
    df = df.loc[df.eval('CutDAQVeto')]
    df['previous_largest_s2_area_sum'] = [np.sum(areas) for areas in df.previous_s2_areas.values]

    df['search_window'] = np.clip(df.s1_center_time, 0, 1e6)
    df.loc[~(df.search_window>0), 'search_window'] = 1e6

    B1 = np.asarray(df.event_time-df.previous_largest_s2_time, dtype=int)
    B2 = np.asarray(df.event_time-df.previous_largest_s2_time+df.search_window, dtype=int)

    # Here is where magic happens
    Ag1, Bg1 = np.meshgrid(A1, B1, sparse=False, indexing='ij')
    Ag2, Bg2 = np.meshgrid(A2, B2, sparse=False, indexing='ij')
    # Dim 2 live time with indicies: (Index of delay from previous s2, index of events)
    D2lt = func_nb(Ag1.reshape((-1,)), Ag2.reshape((-1,)), Bg1.reshape((-1,)), Bg2.reshape((-1,)))
    D2lt = D2lt.reshape((A1.shape[0], B1.shape[0]))
    D2lt[D2lt < 0] = 0; D2lt = D2lt/1e9; # change live time unit to s, it is more natual for event rate

    return np.sum(D2lt, axis=1), BinS2AreaAxis(df, D2lt)# collapse event id axis

def GetLiveTime(flist):
    with Pool(processes=5, maxtasksperchild=50) as pool:
        result = list(tqdm(pool.imap(GetLiveTimeSingleFile, flist, 2), total=len(flist)))
        list_of_D1lt, list_of_D2lt = zip(*result)
    return np.sum(np.array(list_of_D1lt), axis=0), np.sum(np.array(list_of_D2lt), axis=0)

if __name__ == '__main__':
    with open('/home/zhut/Cool_Electrons/data/FileList.json', 'r') as infile:
        flist = json.load(infile)

    D1lt, D2lt = GetLiveTime(flist)

    for item in ['D1lt', 'D2lt']:
        path = '/home/zhut/Cool_Electrons/data/{it}.txt'.format(it = item)
        np.savetxt(path, eval(item), fmt='%.18e', delimiter=' ')