#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:03:31 2023

@author: cmwever73

Script with First Module Takeaways (funcs):
    
%%i. 7num sum func -- input list, returns dict
%%ii. hist + boxplot graph -- **make in plotly**
%%iii. plot density -- but write your own density func and check it, add mean, 
     and 3 stds
%%iv. range, var and stdev return funcs -- maybe wuth 7numsum? 
    **Give pop vs sample opts**
%v. scipy stats linregress use func, given an x and y -- return model
%%vi. make index percentile func -- make option to assign/save as a lambda?
%%vii. obvserved data vs predicted plot with ploynomial overlay
%%viii. err (MSE, RMSE, R2) -- write your own, and verify
%%ix. scatter plot with line over it (to show correlation)


#for all graphs, try plotly library 


#once done, make classes if that makes sense
"""
from collections import Counter
import math
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Lasso
from scipy.stats import gaussian_kde
import time 


def corr_plt(inpt_x, inpt_y):
    '''
    inpt_x -- numeric data in tuple format ('Label', [<list of data>])
    
    inpt_y -- numeric data in tuple format ('Label', [<list of data>])

    Return: plotly scatter plot with correlation (pearson method)
            graphed and value in title

    '''
    corr = np.corrcoef(inpt_x[1], inpt_y[1])[0,1]
    fig = px.scatter(x=inpt_x[1], y=inpt_y[1], trendline='ols',
                     trendline_color_override='#F4B25E',
                     labels={"x": inpt_x[0], "y": inpt_y[0]},
                     title = inpt_x[0]+ ' vs '+ inpt_y[0] + 
                     '- correlation: ' + str(corr))
    fig.show(renderer='browser')


def dnsty_plt(inpt_data, stat_lns=True):
    '''
    inpt_data -- numeric data in list form (or something convertible 
                                            to list form)
    
    opt var: stdev_lns -- default to True, when True, adds 3 stdev 
                       lines w/ percents and mean, median and mode lines
                       
    Return: plotly density plot with mean, median and mode plotted and optioanl
            stddev lines, and density function
    '''

    stats_data = svn2tn_num_sum(inpt_data, ten=True, pop=False)
    xrange = list(np.arange(stats_data['Mean'] - stats_data['Range'],
                            stats_data['Mean'] + stats_data['Range'],
                            stats_data['Range']*2/1000))
    fnc = gaussian_kde(inpt_data)
    yvals = list(fnc.pdf(xrange))
    fig = go.Figure(go.Scatter(x=xrange, 
                  y=yvals, 
                  mode='lines',
                  line=dict(width=1.5)))
    
                  
    if stat_lns:
        #within 1 stdev
        fig.add_shape(type='line', 
                      x0=stats_data['Mean'] - stats_data['StDev'], 
                      x1=stats_data['Mean'] + stats_data['StDev'], 
                      y0=fnc.pdf(stats_data['Mean'] - stats_data['StDev'])[0], 
                      y1=fnc.pdf(stats_data['Mean'] - stats_data['StDev'])[0], 
                      line={'width':3, 'color':"#93deca"},
                      label={'text': "68.26%, 1stdev", 'xanchor':"right", 
                             'textposition':"start"})
        #within 2 stdevs
        fig.add_shape(type='line', 
                      x0=stats_data['Mean'] - (stats_data['StDev']*2), 
                      x1=stats_data['Mean'] + (stats_data['StDev']*2), 
                      y0=fnc.pdf(stats_data['Mean'] - (stats_data['StDev']*2))[0], 
                      y1=fnc.pdf(stats_data['Mean'] - (stats_data['StDev']*2))[0], 
                      line={'width':3, 'color':"#3bc29d"},
                      label={'text': "95.45%, 2stdev", 'xanchor':"right", 
                             'textposition':"start"})
        #within 3 stdevs
        fig.add_shape(type='line', 
                      x0=stats_data['Mean'] - (stats_data['StDev']*3), 
                      x1=stats_data['Mean'] + (stats_data['StDev']*3), 
                      y0=fnc.pdf(stats_data['Mean'] - (stats_data['StDev']*3))[0], 
                      y1=fnc.pdf(stats_data['Mean'] - (stats_data['StDev']*3))[0], 
                      line={'width':3, 'color':"#28846b"},
                      label={'text': "99.73%, 3stdev", 'xanchor':"right", 
                             'textposition':"start"})
        #mean
        fig.add_trace(go.Scatter(x=[stats_data['Mean'], stats_data['Mean']],
                                 y = [0,fnc.pdf(stats_data['Mean'])[0]],
                    mode='lines',
                    line={'width':3, 'color':"#a3572e", 'dash': "dot"},
                    marker={'color': "#a3572e", "opacity" :0.6},
                    fillcolor='#da9a77',
                    name='Mean'))
        
        #median
        fig.add_trace(go.Scatter(x=[stats_data['Median'], stats_data['Median']],
                                  y = [0,fnc.pdf(stats_data['Median'])[0]],
                    mode='lines',
                    line={'width':3, 'color':"#da9a77", 'dash': "dot"},
                    marker={'color': "#da9a77", 'opacity' :0.6},
                    fillcolor="#da9a77",
                    name='Median'))
        
        #mode
        for mode in stats_data['Mode']:
            fig.add_trace(go.Scatter(x=[mode, mode],
                                      y = [0,fnc.pdf(mode)[0]],
                        mode='lines',
                        line={'width':3, 'color':"#ebc8b5", 'dash': "dot"},
                        marker={'color': "#ebc8b5", 'opacity' :0.6},
                        name='Mode'))
        
    fig.update_layout(showlegend=False)
    fig.show(renderer='browser')
    

def err_ms(obs_vals, pred_vals):
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: mean-squared error

    '''
    return sum([(obs_vals[i] - pred_vals[i])**2 for i in \
                range(len(obs_vals))])/len(obs_vals)

def err_rms(obs_vals, pred_vals):
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: root-mean-squared error

    '''
    return math.sqrt(err_ms(obs_vals, pred_vals))

def err_r2(obs_vals, pred_vals):
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: r-squared error

    '''
    #r2 = 1- (ss_res/ss_tot)
    #ss_res = sum (obs-pred)^2
    #ss_tot = sum(obs-mean_obs)^2
    
    ss_res = sum([(obs_vals[i]-pred_vals[i])**2 for i in range(len(obs_vals))])
    ss_tot = sum([(obs_vals[i]-(sum(obs_vals)/len(obs_vals)))**2 \
                  for i in range(len(obs_vals))])


    return 1 - (ss_res/ss_tot)

def errr_vrbs(obs_vals, pred_vals):
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: print statement of mse, rmse, and r2 
    '''
    print(f'''
          MSE:  {err_ms(obs_vals, pred_vals)}\n
          RMSE: {err_rms(obs_vals, pred_vals)}\n
          R2:   {err_r2(obs_vals, pred_vals)}
          ''')

def hist_box_plt(inpt_data):
    '''
    inpt_data -- numeric data in tuple format ('Label', [<list of data>])

    Return: Plotly Plot with histogram and boxplot
    '''
    
    fig = make_subplots(rows=2, cols=1)
    
    fig.add_trace(go.Histogram(x=inpt_data[1], marker={'color': '#2CA756', 
                                                        'opacity':0.5}, name='Freq'),
                  row=1, col=1)
    
    fig.add_trace(go.Box(x=inpt_data[1], notched=True, fillcolor='#DE639E', name='',
                         opacity=0.75, line={'color': '#642b47' }), row=2, col=1)
    
    fig.update_layout(height=600, width=900, 
                      title_text= f"Histo and Box Plot for {inpt_data[0]}",
                      showlegend=False)
    
    fig.show(renderer='browser')
    

def lnr_reg(x_trn, y_trn, x_tst, y_tst, lsso_tf=False):
    '''
    x_trn -- numeric data in list form (or something convertible 
                                            to list form)
    y_trn -- numeric data in list form (or something convertible 
                                            to list form)
    x_tst -- numeric data in list form (or something convertible 
                                            to list form)
    y_tst -- numeric data in list form (or something convertible 
                                            to list form)

    Return: linear reg model, actual and pred values as dict
    '''
    mdl = LinearRegression().fit(x_trn, y_trn) if not lsso_tf else Lasso().fit(x_trn, y_trn)
        
    print(f'Model: {mdl}')

    #predict using test set
    preds = mdl.predict(x_tst)
    np.set_printoptions(suppress=True)
    print('First 10 Predicted labels: ', np.round(preds)[:10])
    print('First 10 Actual labels   : ' ,y_tst[:10])
    
    return {'Model': mdl, 'Observed Labels': y_tst, 'Predicted Labels': preds}

def obs_v_pred(obs_vals, pred_vals):
    '''
    obs_vals -- numeric data in list form (or something convertible 
                                            to list form)
    pred_vals -- numeric data in list form (or something convertible 
                                            to list form)

    Returns: plotly plot with observed vs predicted values and a  ploynomial

    '''
    fig = px.scatter(x=obs_vals, y=pred_vals, trendline='ols',
                     trendline_color_override='#F4B25E',
                     labels={"x": 'Observed Values', "y": 'Predicted Values'},
                     title= 'Observed vs Predicted Values')
    fig.show(renderer='browser')

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
    
    
    print(f'Script took {round(time.time()-strt_t, 2)}s to run.')