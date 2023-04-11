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
from scipy import stats

def tst_7210numsum():
    inpt_data = [1,2,3,4,5,6,7,8,8,8,8,8,10,11,13]
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
    inpt_data = [1,2,3,4,5,6,7,8,8,8,8,8,10,11,13]
    print(mos.svn2tn_num_sum(inpt_data, ten=False, pop=False))

    print('\nTesting a list, ten=True, sample . . .')
    inpt_data = [1,2,3,4,5,6,7,8,8,8,8,8,10,11,13]
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=False))

    print('\nTesting a list, ten=False, pop . . .')
    inpt_data = [1,2,3,4,5,6,7,8,8,8,8,8,10,11,13]
    print(mos.svn2tn_num_sum(inpt_data, ten=False, pop=True))
    
    print('\nTesting a list, ten=True, pop . . .')
    inpt_data = [1,2,3,4,5,6,7,8,8,8,8,8,10,11,13]
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=True))
    
    print('\nTesting a tuple, ten=True, sample . . .')
    inpt_data = tuple([1,2,3,4,5,6,7,8,8,8,8,8,10,11,13])
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=False))
    
    print('\nTesting a set, ten=True, sample . . .')
    inpt_data = {1,2,3,4,5,6,7,8,8,8,8,8,10,11,13}
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=False))
    
    print('\nTesting a empty set, ten=True, sample . . .')
    inpt_data = {}
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=False))
    print('\nTesting a empty list, ten=True, sample . . .')
    inpt_data = []
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=False))
    
    print('\nTesting a string, ten=True, sample . . .')
    inpt_data ='cat in the hat'
    print(mos.svn2tn_num_sum(inpt_data, ten=True, pop=False))

if __name__ == "__main__":
    
    strt_t = time.time()
    
    print(f'Script took {round(time.time() - strt_t, 2)}s to run.')