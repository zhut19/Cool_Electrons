#######################################
# Data Process Manager Classes
# by Tianyu tz2263@columbia.edu, Feb 2018
#######################################

import os, getpass
import numpy as np
import pandas as pd

from configparser import ConfigParser
from cax.qsub import submit_job
import time

import hax

class Manager():
    '''Common functionality across processing classes'''
    __version__ = '0.0.0'
    def __init__(self, config_file = None):

        # Read the parameters used for processing
        self.config_file = config_file
        self.config = ConfigParser()
        if config_file:
            config_files = [f[:-4] for f in os.listdir(os.path.join(os.getcwd(),'config')) if '.ini' in f]
            if config_file in config_files:
                self.config.read(os.path.join(os.getcwd(),'config','%s.ini'%config_file))
            else:
                self.config.read(config_file)
        else:
            self.config.read(os.path.join(os.getcwd(),'config','test.ini'))

    def __str__(self, section = 'BASICS'):
        print('Using following conditions for %s:\n' % self.name())
        for k, it in self.config.items(section):
              print (' - ',k,':', it)
        print()
        return ''

    def name(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        # Function named after lower case of class name is defined as class call
        return eval('self.{name}(*args, **kwargs)'.format(name = self.name().lower()))

class Update_Runs_From_Database(Manager):
    '''Check the run database by calling hax'''
    __version = '0.0.1'

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
                     tqdm_on = True,
                    )

    def update_runs_from_database(self):
        self.__str__('BASICS')

        self.hax_init()
        self.runs = ['sciencerun0', 'sciencerun1']
        self.datasets = dict()
        self.cbasics = self.config['BASICS']

        for sr in self.runs:
            datasets = hax.runs.datasets
            datasets = hax.runs.tags_selection(datasets, include = [sr], exclude = [])
            datasets_copy = datasets.copy()
            datasets_copy = datasets_copy.loc[datasets_copy.source__type != 'LED']

            sr_sets_dict = {st : datasets_copy.loc[datasets_copy.source__type == st] for st in datasets_copy.source__type.unique()}
            self.datasets[sr] = sr_sets_dict

        if self.cbasics['run_spicification'] in self.runs:
            self.datasets = self.datasets[self.cbasics['run_spicification']]
        else:
            print('Can not combine runs yet'); return 0

        if self.cbasics['source_type'] in self.datasets.keys():
            self.datasets = self.datasets[self.cbasics['source_type']]
        else:
            print('Can not combine source types yet'); return 0
        
        self.datasets = self.datasets[self.datasets.start > np.datetime64(self.cbasics['set_selection_start'])]
        self.datasets = self.datasets[self.datasets.start < np.datetime64(self.cbasics['set_selection_stop'])]
        
        print('\n%d runs found\n' %len(self.datasets))
        self.datasets.to_pickle(os.path.join(os.getcwd(),'pickle','%s.pkl'%self.cbasics['name'].lower()))
        return self.datasets

class Check_n_Batch(Manager):
    '''Compare processed files with pickled datasets info and group unfinished ones'''
    __version__ = '0.0.2'

    def check_n_batch(self):  
        self.__str__('PROCESSING')

        self.head_directory = self.config['PROCESSING']['head_directory']
        self.cbasics = self.config['BASICS']
        self.log = os.path.join(self.head_directory, 'log')
        self.outdir = os.path.join(self.head_directory, self.cbasics['name'])
        if not os.path.exists(self.log): os.makedirs(self.log)
        if not os.path.exists(self.outdir): os.makedirs(self.outdir)

        self.num_group = int(self.config['PROCESSING']['number_group'])
        self.datasets = pd.read_pickle(os.path.join(os.getcwd(),'pickle','%s.pkl'%self.cbasics['name'].lower()))
        self.datasets.sort_values(by = 'number', axis = 'index', inplace = True)
        self.run_list_all = self.datasets.number.values
        self.exist_runs = [f[:-4] for f in os.listdir(self.outdir) if 'pkl' in f]

        self.run_list_process = list(set(self.run_list_all).difference(self.exist_runs))
        self.run_list_process = self.datasets[self.datasets.number.isin(self.run_list_process)].name.values

        print('\nTry update %s to %d runs; adding %d runs to %s\n' %(self.cbasics['name'], len(self.run_list_all), len(self.run_list_process), self.outdir))

        self.run_list_process = np.array_split(self.run_list_process, self.num_group)
        self.group_list = []

        for ix, group in enumerate(self.run_list_process):
            group_name = '{name}_{:02}'.format(ix, name = self.cbasics['name'])
            self.group_list.append(group_name)
            list_file = os.path.join(self.log, '{gn}_id_list.txt'.format(gn=group_name))
            np.savetxt(list_file, np.asarray(group, dtype='U32'),  fmt='%s')

        with open(os.path.join(os.getcwd(), '%s.txt' % 'group_list'), 'w+') as group_list_file:
            _ = [group_list_file.write('%s\n' % g) for g in self.group_list]
            group_list_file.close()

        return self.group_list

class Submit(Manager):
    '''Submit to computing nodes if needed'''
    __version__ = '0.0.2'

    def submit(self):
        self.__str__('MIDWAYSUBMIT')
        self.head_directory = self.config['PROCESSING']['head_directory']
        self.max_num_submit = int(self.config['MIDWAYSUBMIT']['max_num_submit'])
        self.limit = int(self.config['MIDWAYSUBMIT']['total_submission_number_limit'])
        self.id = 0
        self.log = os.path.join(self.head_directory, 'log')
        
        group_list_file = open(os.path.join(os.getcwd(), 'group_list.txt'))
        self.group_list = group_list_file.read().splitlines()[:self.limit]
        
        index = 0
        while (index < len(self.group_list)):
            if (self.working_job() < self.max_num_submit):
                self.submit_single(group_name = self.group_list[index])
                time.sleep(0.1)
                index += 1

        return self.y

    # check my jobs
    def working_job(self):
        cmd='squeue --user={user} | wc -l'.format(user = 'zhut')
        jobNum=int(os.popen(cmd).read())
        return  jobNum -1

    def submit_single(self, group_name):
        cmd = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000
#SBATCH --output={log}/{group_name}.log
#SBATCH --error={err}/{group_name}.log
#SBATCH --account=pi-lgrandi
#SBATCH --partition={partition}
#SBATCH --qos={qos}
export PATH=/project/lgrandi/anaconda3/bin:$PATH
export PROCESSING_DIR={tmp}
cd {cwd}
source activate {conda_env}
python {script} {outdir} {log}/{group_name}_id_list.txt
rm -rf ${{PROCESSING_DIR}}
"""
        y = cmd.format(job_name = 'submit{:02}'.format(self.id),
                       group_name = group_name,
                       log = os.path.join(self.log),
                       err = os.path.join(self.log),
                       tmp = os.path.join(os.getcwd(), 'tmp'),
                       cwd = os.getcwd(),
                       partition = self.config['MIDWAYSUBMIT']['partition'],
                       qos = self.config['MIDWAYSUBMIT']['qos'],
                       conda_env = self.config['BASICS']['conda_env'],
                       outdir = os.path.join(self.head_directory, self.config['BASICS']['name']),
                       script = self.config['MIDWAYSUBMIT']['script'],
                      )

        submit_job(y)
        self.y = y
        self.id +=1

class Combined_Manager(Manager):
    __version__ = '0.0.0'

    def combined_manager(self):

        manager_list = [Update_Runs_From_Database(self.config_file),
                        Check_n_Batch(self.config_file),
                        Submit(self.config_file),
                       ]

        result = [mg() for mg in manager_list]

        return result

import sys
if __name__ == '__main__':
    ''' Using argv
        Example python data_processing_manager.py [config]
        [config] can just be the name or full path
    '''
    if len(sys.argv) > 1:
        argv = sys.argv
    else:
        exit(0)
    
    CM = Combined_Manager(argv[1])
    result = CM()