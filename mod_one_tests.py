#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:19:52 2023

@author: cmwever73

test_script for mod one funcs
"""

import time
import mod_one_sumup as mos
import numpy as np
import random
from scipy import stats
import sys



def tst_corr_plt():
    
    x_lst = [50,55,60,75,80,80,88,90,100,83,79,101,65,99,93,91]
    y_lst = [3, 5, 2, 14, 40,47, 58, 70, 75, 40, 31, 80, 10,82, 71, 72]
    mos.corr_plt((u'Temp (\N{DEGREE SIGN}F)', x_lst), ('Ice Creams Sold', y_lst))

def tst_dnsty_plt():
    x_lst = [3.0, 15.0, 26.0, 27.0, 35.0, 36.0, 37.0, 42.0, 47.0, 48.0, 49.0,
             50.0, 50.0, 52.0, 53.0, 62.0, 63.0, 64.0, 70.0, 74.0, 82.0, 97.0]
    mos.dnsty_plt(x_lst, stat_lns=True)
    mos.dnsty_plt(x_lst, stat_lns=False)

def tst_errs():
    
    obs_tst = [3,-0.5,2,7]
    pred_tst = [2.5, 0.0, 2, 8]
    print(mos.err_ms(obs_tst, pred_tst))
    print(mos.err_rms(obs_tst, pred_tst))
    print(mos.err_r2(obs_tst, pred_tst))

def tst_hist_box_plt():
    inpt_data = random.sample(range(500), 300)
    mos.hist_box_plt(('Randomly Generated Data', inpt_data))

def tst_obs_v_pred_plt():
    
    mos.obs_v_pred([1,2,3,4,5,6,7,9,10], [1,2,3,4,5,6,7,9,10])
    
    rnd_smpl_x = random.sample(range(50), 30)
    rndish_smpl_y = [rnd_vl+random.randint(0,5) if rnd_vl%2 == 0 else rnd_vl-random.randint(0,5) for rnd_vl in rnd_smpl_x]
    mos.obs_v_pred(rnd_smpl_x, rndish_smpl_y)
    
def tst_7210numsum():
    
    inpt_data = [random.randint(0,20) for i in range(60)]
    print('Ground Truth using NP:')
    trth = {'Mean': np.mean(inpt_data),
            'Median': np.median(inpt_data),
            'Mode': stats.mode(inpt_data),
            'Min': np.min(inpt_data),
            'Max': np.max(inpt_data),
            '.25': np.quantile(inpt_data, 0.25),
            '.75': np.quantile(inpt_data, 0.75),
            'Range': np.max(inpt_data) - np.min(inpt_data),
            'Var_sample': np.var(inpt_data, ddof=1),
            'Var_pop': np.var(inpt_data, ddof=0),
            'Stdev_sample': np.std(inpt_data, ddof=1),
            'Stdev_pop': np.std(inpt_data, ddof=0)}
    
    print(trth)
    print('\n%%%%%%%%%%%%TESTINGT%%%%%%%%%%%%\n')
            
    print('Testing a list, ten=False, sample . . .')
    print(mos.svn2tn_num_sum(inpt_data, ten=False, pop=False))

    print('\nTesting a list, ten=True, sample . . .')
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=False))

    print('\nTesting a list, ten=False, pop . . .')
    print(mos.svn2tn_num_sum(inpt_data, ten=False, pop=True))
    
    print('\nTesting a list, ten=True, pop . . .')
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=True))
    
    print('\nTesting a tuple, ten=True, sample . . .')
    print(mos.svn2tn_num_sum(tuple(inpt_data), ten=True, pop=False))
    
    print('\nTesting a set, ten=True, sample . . .')
    print(mos.svn2tn_num_sum(set(inpt_data), ten=True, pop=False))
    
    print('\nTesting a empty set, ten=True, sample . . .')
    print(mos.svn2tn_num_sum({}, ten=True, pop=False))
    
    print('\nTesting a empty list, ten=True, sample . . .')
    print(mos.svn2tn_num_sum([], ten=True, pop=False))
    
    print('\nTesting a string, ten=True, sample . . .')
    print(mos.svn2tn_num_sum('a cloven pine', ten=True, pop=False))

if __name__ == "__main__":
    
    strt_t = time.time()
    
    tst = sys.argv[1]

    if tst == '7numsum':
        tst_7210numsum()
        
    if tst == 'errrs':
        tst_errs()
        
    if tst == 'plts':
        tst_obs_v_pred_plt()
        tst_corr_plt()
        tst_hist_box_plt()
        tst_dnsty_plt()
    
    print(f'Script took {round(time.time() - strt_t, 2)}s to run.')