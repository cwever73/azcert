#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:03:31 2023

@author: cmwever73

Script with First Module Takeaways (funcs):
    
i. 7num sum func -- input list, returns dict
ii. hist + boxplot graph -- **make in plotly**
iii. plot density -- but write your own density func and check it, add mean, 
     and 3 stds
iv. range, var and stdev return funcs -- maybe wuth 7numsum? 
    **Give pop vs sample opts**
v. scipy stats linregress use func, given an x and y -- return model
vi. make index percentile func -- make option to assign/save as a lambda
vii. test func for all of this
viii. obvserved data vs predicted func
ix. err (MSE, RMSE, R2) -- write your own, and verify

#for all graphs, try plotly library 
"""
import time 

if __name__ == "__main__":
    
    strt_t = time.time()
    
    
    print(f'Script took {round(time.time()-strt_t, 2)}s to run.')
