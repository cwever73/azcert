#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:03:31 2023

@author: cmwever73

Script with First Module Takeaways (funcs):
    
%i. 7num sum func -- input list, returns dict
%ii. hist + boxplot graph -- **make in plotly**
%iii. plot density -- but write your own density func and check it, add mean, 
     and 3 stds
%iv. range, var and stdev return funcs -- maybe wuth 7numsum? 
    **Give pop vs sample opts**
%v. scipy stats linregress use func, given an x and y -- return model
%vi. make index percentile func -- make option to assign/save as a lambda?
vii. test func for all of this
%viii. obvserved data vs predicted plot with ploynomial overlay
%ix. err (MSE, RMSE, R2) -- write your own, and verify
%x. scatter plot with line over it (to show correlation)


#for all graphs, try plotly library 


#once done, make classes if that makes sense
"""
from collections import Counter
import math
import numpy as np
from scipy import stats
import sys
import time 

def corr_plt():
    '''
    inpt_x -- numeric data in list form (or something convertible 
                                            to list form)
    inpt_y -- numeric data in list form (or something convertible 
                                            to list form)

    Return: plotly scatter plot with correlation graphed and value in title

    '''
    pass


def dnsty():
    '''
    inpt_data -- numeric data in list form (or something convertible 
                                            to list form)

    Return: probability density func (PDF) as a dict cuz values in dicts can be 
            funcs bc python is neato

    '''
    pass

def dnsty_plt():
    '''
    inpt_data -- numeric data in list form (or something convertible 
                                            to list form)
    
    opt var: 3stdev -- default to False, when True, adds 3 stdev 
                       lines w/ percents
                       
    Return: plotly density plot with mean, median and mode plotted and optioanl
            stddev lines
    '''
    pass

def err_ms():
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: mean-squared error

    '''
    pass

def err_rms():
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: root-mean-squared error

    '''
    pass

def err_r2():
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: r-squared error

    '''
    pass

def hist_box_plt():
    '''
    inpt_data -- numeric data in list form (or something convertible 
                                            to list form)

    Return: Plotly Plot with histogram and boxplot
    '''
    pass


def lnr_reg():
    '''
    inpt_x -- numeric data in list form (or something convertible 
                                            to list form)
    inpt_y -- numeric data in list form (or something convertible 
                                            to list form)

    Return: linear reg model as dict
    '''
    pass

def obs_v_pred():
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: plotly plot with observed vs predicted values and a  ploynomial

    '''
    pass

def prcntl_indx(inpt_data, prcnt):
    '''
    inpt_data -- numeric data in list form (or something convertible 
                                            to list form)
    prcnt -- desired percentile

    Return: index where percenile lands
    '''
    # multiply N by prcnt
    return round(len(inpt_data)*prcnt)

def svn2tn_num_sum(inpt_data, ten=False, pop=False):
    '''
    inpt_data -- numeric data in list form (or something convertible 
                                            to list form)

    
    opt var: ten -- default to False (adds Range, Var and StDev to r
                                      eturned dict)
             pop -- default to False. When true, applies population variance
                    and stdev. When False, applies sample variance and stdev
    
    Return: Dict with Mean, Median, Mode, 0.25q, 0.75q, Min, Max, 
            Ten num sum oadditions: Range, Var, StDev
    '''
    
    if not isinstance(inpt_data, list):
        try:
            inpt_data = list(inpt_data)
            return svn2tn_num_sum(inpt_data, ten=ten, pop=pop)
        except:
            print(f'Current input datatype: {type(inpt_data)}')
            print('Not a valid datatype. Please input data in list form.')
    else:
        if len(inpt_data) > 0:
            #sort for median
            inpt_data.sort()
            #calc mean
            data_avg = sum(inpt_data)/len(inpt_data)
            #calc median
            if len(inpt_data) % 2 == 0:
                data_med = (inpt_data[int(len(inpt_data)/2)] + 
                            inpt_data[int(len(inpt_data)/2) - 1])/2
            else:
                data_med = inpt_data[int(len(inpt_data)/2)]
            #calc mode
            data_md = [tpl[0] for tpl in Counter(inpt_data).most_common() 
                       if tpl[1] == Counter(inpt_data).most_common()[0][1]]
            
            
            svn2tn_sum = {'Mean': data_avg, 
                          '25q': inpt_data[prcntl_indx(inpt_data, 0.25)],
                          'Median': data_med, 
                          '75q': inpt_data[prcntl_indx(inpt_data, 0.75)],
                          'Mode': data_md, 
                          'Min': min(inpt_data), 
                          'Max': max(inpt_data)}
            
            if ten:
                svn2tn_sum['Range'] = svn2tn_sum['Max'] - svn2tn_sum['Min']
                if pop:
                    svn2tn_sum['Variance'] = sum([(x-svn2tn_sum['Mean'])**2 
                                        for x in inpt_data])/len(inpt_data) 
                                        #pop
                    svn2tn_sum['pop|sample'] = 'population'
                if not pop:
                    svn2tn_sum['Variance'] = sum([(x-svn2tn_sum['Mean'])**2 
                                        for x in inpt_data])/(len(inpt_data)-1) 
                                        #sample
                    svn2tn_sum['pop|sample'] = 'sample'
                svn2tn_sum['StDev'] = math.sqrt(svn2tn_sum['Variance'])
            return svn2tn_sum
        else:
            print('Error: length of inputted data is 0')




if __name__ == "__main__":
    
    strt_t = time.time()
    
    opt = sys.argv[1]
    
    if opt == 'test_7munsum':
        
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
        print(svn2tn_num_sum(inpt_data, ten=False, pop=False))
    
        print('\nTesting a list, ten=True, sample . . .')
        inpt_data = [1,2,3,4,5,6,7,8,8,8,8,8,10,11,13]
        print(svn2tn_num_sum(inpt_data, ten=True, pop=False))
    
        print('\nTesting a list, ten=False, pop . . .')
        inpt_data = [1,2,3,4,5,6,7,8,8,8,8,8,10,11,13]
        print(svn2tn_num_sum(inpt_data, ten=False, pop=True))
        
        print('\nTesting a list, ten=True, pop . . .')
        inpt_data = [1,2,3,4,5,6,7,8,8,8,8,8,10,11,13]
        print(svn2tn_num_sum(inpt_data, ten=True, pop=True))
        
        print('\nTesting a tuple, ten=True, sample . . .')
        inpt_data = tuple([1,2,3,4,5,6,7,8,8,8,8,8,10,11,13])
        print(svn2tn_num_sum(inpt_data, ten=True, pop=False))
        
        print('\nTesting a set, ten=True, sample . . .')
        inpt_data = {1,2,3,4,5,6,7,8,8,8,8,8,10,11,13}
        print(svn2tn_num_sum(inpt_data, ten=True, pop=False))
        
        print('\nTesting a empty set, ten=True, sample . . .')
        inpt_data = {}
        print(svn2tn_num_sum(inpt_data, ten=True, pop=False))
        
        print('\nTesting a empty list, ten=True, sample . . .')
        inpt_data = []
        print(svn2tn_num_sum(inpt_data, ten=True, pop=False))
        
        print('\nTesting a string, ten=True, sample . . .')
        inpt_data ='cat in the hat'
        print(svn2tn_num_sum(inpt_data, ten=True, pop=False))
        
        
        
        
    
    print(f'Script took {round(time.time()-strt_t, 2)}s to run.')
