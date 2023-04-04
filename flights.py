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
from collections import Counter
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

    


