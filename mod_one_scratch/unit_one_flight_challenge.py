#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 13:11:30 2023

@author: cmwever73

Script walks thru azure training module Flights challenge:
    
The dataset contains observations of US domestic flights in 2013, and consists of the following fields:

Year: The year of the flight (all records are from 2013)
Month: The month of the flight
DayofMonth: The day of the month on which the flight departed
DayOfWeek: The day of the week on which the flight departed - from 1 (Monday) 
    to 7 (Sunday)
Carrier: The two-letter abbreviation for the airline.
OriginAirportID: A unique numeric identifier for the departure aiport
OriginAirportName: The full name of the departure airport
OriginCity: The departure airport city
OriginState: The departure airport state
DestAirportID: A unique numeric identifier for the destination aiport
DestAirportName: The full name of the destination airport
DestCity: The destination airport city
DestState: The destination airport state
CRSDepTime: The scheduled departure time
DepDelay: The number of minutes departure was delayed (flight that left ahead
    of schedule have a negative value)
DelDelay15: A binary indicator that departure was delayed by more than 15 
    minutes (and therefore considered "late")
CRSArrTime: The scheduled arrival time
ArrDelay: The number of minutes arrival was delayed (flight that arrived ahead
    of schedule have a negative value)
ArrDelay15: A binary indicator that arrival was delayed by more than 15 
    minutes (and therefore considered "late")
Cancelled: A binary indicator that the flight was cancelled
Your challenge is to explore the flight data to analyze possible 
factors that affect delays in departure or arrival of a flight.

Start by cleaning the data.
i.   Identify any null or missing data, and impute appropriate replacement values.
ii.  Identify and eliminate any outliers in the DepDelay and ArrDelay columns.
iii. Explore the cleaned data.
iv.  View summary statistics for the numeric fields in the dataset.
v.   Determine the distribution of the DepDelay and ArrDelay columns.
vi.  Use statistics, aggregate functions, and visualizations to answer the 
     following questions:
         vii. What are the average (mean) departure and arrival delays?
         viii.How do the carriers compare in terms of arrival delay performance?
         ix.  Is there a noticable difference in arrival delays for different days 
         of the week?
         x.   Which departure airport has the highest average departure delay?
         xii. Do late departures tend to result in longer arrival delays than on-time departures?*
         xiii.Which route (from origin airport to destination airport) has the most 
         late arrivals?
         xiv. Which route has the highest average arrival delay?   
"""
from bokeh.plotting import figure, show
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import random as rndm

#read in csv -- assumes running inside repo
#######################################################
with open('data/flights.csv') as f:
    flght_data = f.readlines()

    
hdrs = flght_data[0].split(',')
#clean up '\n' too
hdrs = [hdr.replace('\n', '') for hdr in hdrs]
#list of dictionaries holding flights
flghts = [{hdrs[indx]:data.replace('\n', '') for indx, data in enumerate(flght.split(','))} for flght in flght_data[1:]]
#now have list of dictionaries holding flight data

###pandas parallel###
flghts_df = pd.read_csv('data/flights.csv', delimiter=',')


#i.  Identify any null or missing data, and impute appropriate replacement values.
#######################################################
nlls = []
for indx, flght in enumerate(flghts):
    for k,v in flght.items():
        if v == None:
            nlls.append(['None', indx, k , v, ])
        elif v == '':
            nlls.append(['Empty str', indx, k , v, ])
        elif v == 0 and k not in ['Cancelled', 'ArrDel15', 'DelDelay15' ]:
            #zero values that are not binary columns
            nlls.append(['Zero', indx, k , v, ])

#zero and None nulls seem to never occur, only empty strings
clmns_w_nlls = []
for nll_val in nlls:
    print(flghts[nll_val[1]])
    #what columns are typically empty?
    clmns_w_nlls.append(nll_val[2])

#check DepDelay column and set > 15min as true (1), neg numbers== early
for indx, flght in enumerate(flghts):
    if flght['DepDel15'] == '':
        if int(flght['DepDelay']) > 15:
            flght['DepDel15'] = '1'
        else:
            flght['DepDel15'] = '0'
        flghts[indx] == flght
    
#ii.  Identify and eliminate any outliers in the DepDelay and ArrDelay columns.
#######################################################

#going to use percentile route to get rid of anything below the 0.01th 
# percentile and above the 0.99 percentile

depdel = [int(flght['DepDelay']) for flght in flghts]
arrdel = [int(flght['ArrDelay']) for flght in flghts]

def stats(data):
    '''Given a list of numbers, return mean, median and mode of data'''
    if not isinstance(data, list):
        print('Not a valid datatype. Please input data in list form.')
    else:
        #sort for median
        data.sort()
        #calc mean
        data_avg = sum(data)/len(data)
        #calc median
        if len(data) % 2 == 0:
            data_med = (data[int(len(data)/2)] + data[int(len(data)/2) - 1])/2
        else:
            data_med = data[int(len(data)/2)]
        #calc mode
        data_md = [tpl[0] for tpl in Counter(data).most_common() if tpl[1] == Counter(data).most_common()[0][1]]
        
        return data_avg, data_med, data_md, min(data), max(data)

#check DepDelay column
depdel_mn, depdel_med, depdel_md, depdel_min, depdel_max = stats(depdel)
###pandas parallel###
depdel_pd_mn, depdel_pd_med, depdel_pd_md, depdel_pd_min, depdel_pd_max = flghts_df['DepDelay'].mean(), flghts_df['DepDelay'].median(), flghts_df['DepDelay'].mode(), flghts_df['DepDelay'].min(), flghts_df['DepDelay'].max()

#check DepDelay column
arrdel_mn, arrdel_med, arrdel_md, arrdel_min, arrdel_max = stats(arrdel)
###pandas parallel###
arrdel_pd_mn, arrdel_pd_med, arrdel_pd_md, arrdel_pd_min, arrdel_pd_max = flghts_df['ArrDelay'].mean(), flghts_df['ArrDelay'].median(), flghts_df['ArrDelay'].mode(), flghts_df['ArrDelay'].min(), flghts_df['ArrDelay'].max()


#woot. ok. now, want to get rid of outliers by removing anything below 0.01th
#percentile and above the 0.99th percentile. A function to output percentile
# would be nice

def prcntl(data, prcnt):
    ''' given a list of data and a percentage, 
    return index value for that percent -- this could be a lambda func'''

    # multiply N by prcnt
    return round(len(data)*prcnt)

#check this agrees with pandas parallel
depdel.sort()
print('0.01 Percentile agrees? ', depdel[prcntl(depdel, 0.01)] == flghts_df.DepDelay.quantile(0.01))
print(depdel[prcntl(depdel, 0.01)], flghts_df.DepDelay.quantile(0.01))
print('0.99 Percentile agrees? ', depdel[prcntl(depdel, 0.99)] == flghts_df.DepDelay.quantile(0.99))
print(depdel[prcntl(depdel, 0.99)], flghts_df.DepDelay.quantile(0.99))

arrdel.sort()
print('0.01 Percentile agrees? ', arrdel[prcntl(arrdel, 0.01)] == flghts_df.ArrDelay.quantile(0.01))
print(arrdel[prcntl(depdel, 0.01)], flghts_df.ArrDelay.quantile(0.01))
print('0.99 Percentile agrees? ', arrdel[prcntl(arrdel, 0.99)] == flghts_df.ArrDelay.quantile(0.99))
print(arrdel[prcntl(arrdel, 0.99)], flghts_df.ArrDelay.quantile(0.99))

#to shave off 0.001% of data on either side (therby ridding any extremes)
depdel_updt = depdel[prcntl(depdel, 0.001):prcntl(depdel, 0.999)]
arrdel_updt = arrdel[prcntl(arrdel, 0.001):prcntl(arrdel, 0.999)]

#check stats now with outliers gonzo
depdel_mn, depdel_med, depdel_md, depdel_min, depdel_max = stats(depdel)
depdel_mn_u, depdel_med_u, depdel_md_u, depdel_min_u, depdel_max_u = stats(depdel_updt)
arrdel_mn, arrdel_med, arrdel_md, arrdel_min, arrdel_max = stats(arrdel)
arrdel_mn_u, arrdel_med_u, arrdel_md_u, arrdel_min_u, arrdel_max_u = stats(arrdel_updt)

#after exploring data further, outliers lie on the max-end of data, not min
#so, will instead just crop the top
depdel_updt = depdel[:prcntl(depdel, 0.9999)]
arrdel_updt = arrdel[:prcntl(arrdel, 0.9999)]

#now need to actually trim data as a whole, above we 
#just did that for a copy of the ArrDelay and DepDelay data
flghts_updt = sorted(flghts, key=lambda dct: dct['DepDelay'])
#first sort by DepDelay and get rid of top 0.0001%
flghts_updt = flghts_updt[:prcntl(depdel, 0.9999)]
#now check min value of ArrDel that would be in 0.0001%
#so also go thru and remove any flight with >= 686 for Arrdel
flghts_updt = [flght for flght in flghts_updt if int(flght['ArrDelay']) < arrdel[prcntl(arrdel, 0.9999)]]
#this ensures that any flights in the top 0.0001% of Departure Delays are removed
#and that the top 0.0001% of Arrival Delays are also removed 
#this still leaves us with 271886 data points -- 99% of the data.
#so we should be ok
print('Percentage of Kept flights: ', len(flghts_updt)/len(flghts))


#iii. Explore the cleaned data.
#######################################################

explr = []
cnt = 12 #look at 12 random flights
while cnt > 0:
    explr.append(flghts_updt[rndm.randint(0, len(flghts_updt))])
    cnt -= 1

for flght in explr:
    print('**********************')
    for k,v in flght.items():
        print(f'{k}:  {v}')
        
print(set([f['Year'] for f in flghts_updt]))
#all for year 2013

print('----------Origin Airport-----------------')
for arprt in Counter([f['OriginAirportName'] for f in flghts_updt]).most_common():
    print(arprt)

print('----------Destination Airport------------')

for arprt in Counter([f['DestAirportName'] for f in flghts_updt]).most_common():
    print(arprt)

    
#iv.  View summary statistics for the numeric fields in the dataset.
#######################################################
#well I'll be .... nice we already have a func for that. yeehaw.


#can use any element to check numeric fields since data has been cleaned
nmrc_flds = [] 
for fld in flghts_updt[0]:
    try:
        int(flghts_updt[0][fld])
        nmrc_flds.append(fld)
    except:
        continue
    
#before moving forward. Should be noted that the binary and ID columns should
#not show anything of value in summary stats -- other than confirming that
#all values are 0/1 for the binary columns and that the ID columns have truly
#unique numbers. Time fields will also have interesting characteristics.

for fld in nmrc_flds:
    print('^^^^^^^^^^^^', fld, '^^^^^^^^^^^^')
    fld_data = [int(flght[fld]) for flght in flghts_updt]
    if 'ID' in fld:
        #check that id is unique
        print(f'For {fld} field -- are IDs unique?')
        tst = {}
        for flght in flghts_updt:
            tst.setdefault(flght[fld.replace('AirportID', 'AirportName')], [])
            if flght[fld] not in tst[flght[fld.replace('AirportID', 'AirportName')]]:
                 tst[flght[fld.replace('AirportID', 'AirportName')]].append(flght[fld])
        # print(tst)
        tst_st = [len(set(v)) for k,v in tst.items()]
        unq_tf = True if set(tst_st) == {1} else False
        print(f'Is the {fld} Column unique to each airport? {unq_tf}')
        
    elif fld in ['ArrDel15', 'DepDel15', 'Cancelled']:
        #are all values binary?
        print(f'List of values that are not binary: {[val for val in fld_data if val not in [0,1]]}')
        print(f'Min of biinary column: {min(fld_data)}')
        print(f'Max of biinary column: {max(fld_data)}')
    else:
        #perform summary stats on the rest
        mn, med, md, data_min, data_max = stats(fld_data)
        q25 = fld_data[prcntl(fld_data, 0.25)]
        q75 = fld_data[prcntl(fld_data, 0.75)]
        print(f'''Mean: {mn}
                 Median: {med}
                 Mode: {md}
                 Min: {data_min}
                 Max: {data_max}
                 25th Percentile: {q25}
                 75th Percentile: {q75}''')
                 
#v. Determine the distribution of the DepDelay and ArrDelay columns.                 
#######################################################

def plt_dst(fld_data):
    mn, med, md, data_min, data_max = stats(fld_data)

    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    #histogram   
    ax[0].hist(fld_data)
    ax[0].set_ylabel('Frequency')

    #stat lines
    ax[0].axvline(x=data_min, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mn, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=md, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=data_max, color = 'gray', linestyle='dashed', linewidth = 2)

    #boxplot   
    ax[1].boxplot(fld_data, vert=False)
    ax[1].set_xlabel('Value')

    fig.suptitle('Data Distribution')
    fig.show()
    
    
plt_dst(depdel_updt)
plt_dst(depdel)
plt_dst(arrdel_updt)
plt_dst(arrdel)

#woah -- definitely could have cropped more of the data
#for comparison:
depdel_updt95 = depdel[:prcntl(depdel, 0.95)]
plt_dst(depdel_updt95)
arrdel_updt95 = arrdel[:prcntl(arrdel, 0.95)]
plt_dst(arrdel_updt95)

depdel_updt90 = depdel[:prcntl(depdel, 0.90)]
plt_dst(depdel_updt90)
arrdel_updt90 = arrdel[:prcntl(arrdel, 0.90)]
plt_dst(arrdel_updt90)
 #.... so what does that mean? What constitues an outlier? I could crop down
 #to the 90% and get close to a normal dist - but should I? dont know. 
 
 
 
#vi.  Use statistics, aggregate functions, and visualizations to answer the 
#     following questions:
#######################################################  
  
# vii. What are the average (mean) departure and arrival delays?
print(stats(depdel_updt90))
print(stats(arrdel_updt90))
# ANS(using 90th percentile cut): Dep_mu = +1.35, Arr_mu = -3.06

# viii. How do the carriers compare in terms of arrival delay performance?

#gotta redo whole data trim for 90% cut (will be greater than 90,
#when accounting for Arr and Dep Delay outliers)
flghts_updt90 = sorted(flghts, key=lambda dct: dct['DepDelay'])
#first sort by DepDelay and get rid of top 0.10%
flghts_updt90 = flghts_updt90[:prcntl(depdel, 0.90)]
#now check min value of ArrDel that would be in 0.10%
flghts_updt90 = [flght for flght in flghts_updt90 if int(flght['ArrDelay']) < arrdel[prcntl(arrdel, 0.90)]]
#this ensures that any flights in the top 0.1% of Departure Delays are removed
#and that the top 0.0001% of Arrival Delays are also removed 
#this only leaves us with 84% of the data.
print('Percentage of Kept flights: ', len(flghts_updt90)/len(flghts))

crrr_data = {}
for flght in flghts_updt90:
    #set keys
    crrr_data.setdefault(flght['Carrier'], {})
    crrr_data[flght['Carrier']].setdefault('ArrDelay', [])
    crrr_data[flght['Carrier']].setdefault('DepDelay', [])
    crrr_data[flght['Carrier']].setdefault('Ovr15del_all', 0)
    #add vals
    crrr_data[flght['Carrier']]['ArrDelay'].append(int(flght['ArrDelay']))
    crrr_data[flght['Carrier']]['DepDelay'].append(int(flght['DepDelay']))
    crrr_data[flght['Carrier']]['Ovr15del_all'] += (int(flght['ArrDel15'])+int(flght['DepDel15']))

for crrr in crrr_data:
    arr_stats = stats(crrr_data[crrr]['ArrDelay'])
    crrr_data[crrr]['ArrStats'] = {'Mean': arr_stats[0], 'Median': arr_stats[1],
                                   'Mode': arr_stats[2], 'Min': arr_stats[3],
                                   'Max': arr_stats[4]}
    dep_stats = stats(crrr_data[crrr]['DepDelay'])
    crrr_data[crrr]['DepStats'] = {'Mean': dep_stats[0], 'Median': dep_stats[1],
                                   'Mode': dep_stats[2], 'Min': dep_stats[3],
                                   'Max': dep_stats[4]}
    
    print(crrr, '####################')
    print('--------------------------')
    print((crrr_data[crrr]['ArrStats']['Mean'] + crrr_data[crrr]['DepStats']['Mean'])/2)
    print(max(crrr_data[crrr]['ArrStats']['Max'], crrr_data[crrr]['DepStats']['Max']))
    print('--------------------------')
    
    

#plot with bokeh bar chart
p = figure(x_range=list(crrr_data.keys()), height=350, title="Delay >15min Counts by Carrier",
           toolbar_location=None, tools="")
p.vbar(x=list(crrr_data.keys()), top=[crrr_data[crrr]['Ovr15del_all'] for crrr in crrr_data], 
       width=0.9, color='#e1deff', alpha=0.6)
p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)

#WN is the worst.

###Pandas Parallel from jupyternotebook add-in for the sake of knowing
###  how to do box and whisker plots on same plot with pandas
#gotta trim outliers for ArrDelay based on 1% and 90% percentiles
ArrDelay_01pcntile = flghts_df.ArrDelay.quantile(0.01)
ArrDelay_90pcntile = flghts_df.ArrDelay.quantile(0.90)
flghts_df = flghts_df[flghts_df.ArrDelay < ArrDelay_90pcntile]
flghts_df = flghts_df[flghts_df.ArrDelay > ArrDelay_01pcntile]

# Trim outliers for DepDelay based on 1% and 90% percentiles
DepDelay_01pcntile = flghts_df.DepDelay.quantile(0.01)
DepDelay_90pcntile = flghts_df.DepDelay.quantile(0.90)
flghts_df = flghts_df[flghts_df.DepDelay < DepDelay_90pcntile]
flghts_df = flghts_df[flghts_df.DepDelay > DepDelay_01pcntile]
flghts_df.boxplot(column='DepDelay', by='Carrier', figsize=(8,8))

# ix.  Is there a noticable difference in arrival delays for different days 
# of the week?

week_data = {}
for flght in flghts_updt90:
    #set keys
    week_data.setdefault(flght['DayOfWeek'], {})
    week_data[flght['DayOfWeek']].setdefault('ArrDelay', [])
    week_data[flght['DayOfWeek']].setdefault('DepDelay', [])
    week_data[flght['DayOfWeek']].setdefault('Ovr15del_all', 0)
    #add vals
    week_data[flght['DayOfWeek']]['ArrDelay'].append(int(flght['ArrDelay']))
    week_data[flght['DayOfWeek']]['DepDelay'].append(int(flght['DepDelay']))
    week_data[flght['DayOfWeek']]['Ovr15del_all'] += (int(flght['ArrDel15'])+int(flght['DepDel15']))
    
#plot with bokeh bar chart
p = figure(x_range=list(week_data.keys()), height=350, title="Delay >15min Counts by Day",
           toolbar_location=None, tools="")
p.vbar(x=list(week_data.keys()), top=[week_data[day]['Ovr15del_all'] for day in week_data], 
       width=0.9, color='#83937c', alpha=0.6)
p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)


    
# x.   Which departure airport has the highest average departure delay?
arprt_data = {}
for flght in flghts_updt90:
    #set keys
    arprt_data.setdefault(flght['OriginAirportName'], {})
    arprt_data[flght['OriginAirportName']].setdefault('DepDelay', [])
    #add vals
    arprt_data[flght['OriginAirportName']]['DepDelay'].append(int(flght['DepDelay']))

for indx, arprt in enumerate(arprt_data):
    if indx == 0:
        hghst_arprt = ''
        curr_avg = 0
    arprt_stats = stats(arprt_data[arprt]['DepDelay'])
    arprt_data[arprt]['DepStats'] = {'Mean': arprt_stats[0], 'Median': arprt_stats[1],
                                   'Mode': arprt_stats[2], 'Min': arprt_stats[3],
                                   'Max': arprt_stats[4]}
    
    print(arprt, '####################')
    print('--------------------------')
    print(arprt_data[arprt]['DepStats']['Mean'])
    
    if arprt_data[arprt]['DepStats']['Mean'] > curr_avg:
        hghst_arprt = arprt
        curr_avg = arprt_data[arprt]['DepStats']['Mean']

print('Airport with highest avg dep del: ', hghst_arprt)
    
    
# xii. Do late departures tend to result in longer arrival delays than on-time departures?

p = figure(width=550, height=550)
p.circle(depdel_updt90,arrdel_updt90, size=15, color="navy", alpha=0.5)
p.xaxis.axis_label = 'Departure Delays'
p.yaxis.axis_label = 'Arrival Delays'
show(p)

#ANS: Definitely late departures tend to make later arrival delays


# xiii.Which route (from origin airport to destination airport) has the most 
# late arrivals?
arprt_data = {}
for flght in flghts_updt90:
    #set keys
    arprt_data.setdefault((flght['OriginAirportName'],flght['DestAirportName'] ), 0)
    #add vals
    if int(flght['ArrDelay']) > 0:
        arprt_data[(flght['OriginAirportName'],flght['DestAirportName'] )] += 1
print(arprt_data)
#let's look at this in graph form
arrvl_dels = sorted([(k,v)for k,v in arprt_data.items()], key=lambda x: x[1])
arrvl_dels = arrvl_dels[-10:]
p = figure(x_range=[str(i[0]) for i in arrvl_dels], height=700, width=1400,
            title="To-From Airport Arrival Delay Frequency", 
            toolbar_location=None, tools="")
p.vbar(x=[str(i[0]) for i in arrvl_dels], top=[i[1]for i in arrvl_dels], 
       width=0.9, color='#F397D1', alpha=0.4)
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.xaxis.major_label_orientation = "vertical"
show(p)

#ANS: Sanfran to LA has most late arrivals -- numbers vary more than
#in jynb bc I only took off top 10% of data, didnt trim bottom part

# xiv. Which route has the highest average arrival delay?
arprt_data = {}
for flght in flghts_updt90:
    #set keys
    arprt_data.setdefault((flght['OriginAirportName'],flght['DestAirportName'] ),[])
    #add vals
    arprt_data[(flght['OriginAirportName'],flght['DestAirportName'] )].append(int(flght['ArrDelay']))

avg_arr_dels = []
for trek in arprt_data:
    avg_arr_dels.append((trek, stats(arprt_data[trek])[0]))
    
avg_arr_dels_ordr = sorted(avg_arr_dels, key= lambda x:x[1], reverse=True)
print(avg_arr_dels_ordr[0])
#ANS - New Orleans Louis Armstrong to DCA