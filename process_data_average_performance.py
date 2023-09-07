import numpy as np
import pandas as pd
from scipy.stats import sem, ttest_ind

import argparse
import sys
import pickle

sys.path.append('./')
from utils.plotting_tools import *

parser = argparse.ArgumentParser(description='Process raw psiturk data')
parser.add_argument('--input-file', type=str,
                    default='./processed_data_anonymous.csv',
                    help='the processed input csv file containing the behavioral data')
parser.add_argument('--output-folder', type=str,
                    default='./output/',
                    help='where to store figures and things')

args = parser.parse_args()

dfs = {}
T = 71. / 120  # p < .05 binomial test

df = pd.read_csv(args.input_file, header=0)
df.columns = ['workerid', 'condition','stim_present','stimulus', 'response', 'correct', 'rt']
df['stim_present'].replace({1000: '1000msecs', 2000: '2000msecs', 0: 'UntilResponse'}, inplace=True)

df_mean = pd.pivot_table(df,
                         values=['correct'],
                         index=['workerid'],
                         aggfunc=np.mean)
wlist_above_chance = df_mean[df_mean.iloc[:, 0] > T].index
df = df[df.workerid.isin(wlist_above_chance)]

df_mean = pd.pivot_table(df,
                         values=['correct'],
                         index=['workerid', 'condition', 'stim_present'],
                         aggfunc=np.mean)


stim_present_levels = ['1000msecs', '2000msecs', 'UntilResponse']
for c in range(2):
    for t_pres in range(3):
        curr_stim_present = stim_present_levels[t_pres]
        a = np.array(df_mean.query('condition == @c and stim_present == [@curr_stim_present]')).flatten()
        print('Condition: ' + str(c) + ' T: ' + stim_present_levels[t_pres] + ' Count ' + str(a.shape[0]))


"""
PRINT OUT SIGNIFICANT DIFFERENTCES; 
Supplementary Figure 3
U 1SEC TO 2SECS
U 1SEC TO UR
U 2SECS TO UR
O 1SEC TO 2SECS
O 1SEC TO UR
O 2SECS TO UR
"""
stim_present_levels = ['1000msecs', '2000msecs', 'UntilResponse']

for c in range(2):
    for t_short in range(2):
        for t_long in range(t_short+1, 3):
            curr_stim_present = stim_present_levels[t_short]
            a = np.array(df_mean.query('condition == @c and stim_present == [@curr_stim_present]')).flatten()
            curr_stim_present = stim_present_levels[t_long]
            b = np.array(df_mean.query('condition == @c and stim_present == [@curr_stim_present]')).flatten()
            print('Condition: ' + str(c) + ' T1: ' + stim_present_levels[t_short] + ' T2: ' + stim_present_levels[t_long] + ' t-test: ') 
            minN = min(a.shape, b.shape)[0]
            print(minN)
            print(ttest_ind(a[:minN],b[:minN]))


# OVERALL AVERAGE PERFORMANCE ANALYSIS
df_counts = pd.pivot_table(df_mean,
                           values=['correct'],
                           index=['condition', 'stim_present'],
                           aggfunc=np.shape)

df_mean_overall = pd.pivot_table(df_mean, values=['correct'], 
                                 index=['condition', 'stim_present'], 
                                 aggfunc=[np.mean, np.std])

vals = np.reshape(df_mean_overall.values, [2, 3, 2])

colors = np.array([(161,218,180), (166,206,227), (67,162,202)])/255. #updated

PT = PlottingTools()
PT.multiple_barplots(np.moveaxis(vals, 0, 1), ['1sec', '2secs', 'Unl.'], 1, 2, 'overall-averages', 3., 2.5, colors=colors, verbose=False, pdf=False, summary_data=True, set_title=['Un.','Occ.'])


for t_short in range(2):
    for t_long in range(t_short+1, 3):
        curr_stim_present = stim_present_levels[t_short]
        a = np.array(df_mean.query('stim_present == [@curr_stim_present]')).flatten()
        curr_stim_present = stim_present_levels[t_long]
        b = np.array(df_mean.query('stim_present == [@curr_stim_present]')).flatten()
        print('T1: ' + stim_present_levels[t_short] + ' T2: ' + stim_present_levels[t_long] + ' t-test: ') 
        minN = min(a.shape, b.shape)[0]
        print(minN)
        print(ttest_ind(a[:minN],b[:minN]))

# MAIN EFFECT of TIME on PERFORMANCE
df_mean_overall_overall = pd.pivot_table(df_mean, values=['correct'], 
                                 index=['stim_present'], 
                                 aggfunc=[np.mean, np.std])

vals_overall = np.reshape(df_mean_overall_overall.values, [3, 1, 2])


PT = PlottingTools()
PT.multiple_barplots(vals_overall, ['1sec', '2secs', 'Unl.'], 1, 1, 'overall-overall-averages', 3., 1.5, colors=colors, verbose=False, pdf=False, summary_data=True)



# SEPARATE ANALYSIS OF SAME CATEGORY AND DIFFERENT CATEGORY DISTRACTOR TRIALS
same_indices = []
diff_indices = []
for i in range(20):
    for j in range(6):
        if i % 2 == 0:
            same_indices.append(i * 6 + j)
        else:
            diff_indices.append(i * 6 + j)

# SAME CATEGORY DISTRACTOR
df_mean_same = pd.pivot_table(df[df.stimulus.isin(same_indices)], values=[
                              'correct'], index=['workerid', 'condition', 'stim_present'], aggfunc=np.mean)
df_mean_same = pd.pivot_table(df_mean_same, values=['correct'], index=['condition', 'stim_present'], 
                              aggfunc=[np.mean, np.std])

vals_same = np.reshape(df_mean_same.values, [2, 3, 2])
PT.multiple_barplots(np.moveaxis(vals_same, 0, 1), ['1sec', '2secs', 'Unl.'], 1, 2, 'same-distractor-averages', 3., 2.5, colors=colors, verbose=False, pdf=False, summary_data=True, set_title=['Un.', 'Occ.'])

# DIFFERENT CATEGORY DISTRACTOR
df_mean_different = pd.pivot_table(df[df.stimulus.isin(diff_indices)], values=[
                                   'correct'], index=['workerid', 'condition', 'stim_present'], aggfunc=np.mean)
df_mean_different = pd.pivot_table(
    df_mean_different, values=['correct'], index=[
        'condition', 'stim_present'], aggfunc=[
            np.mean, np.std])
vals_different = np.reshape(df_mean_different.values, [2, 3, 2])
PT.multiple_barplots(np.moveaxis(vals_different, 0, 1), ['1sec', '2secs', 'Unl.'], 1, 2, 'different-distractor-averages', 3., 2.5, colors=colors, verbose=False, pdf=False, summary_data=True, set_title=['Un.', 'Occ.'])


# 1000sec [OVERALL, SAME, DIFFERENT]
vals_1000msec = np.zeros((3, 2, 2))
vals_1000msec[0, 0, :] = vals[0, 0, :]  # U
vals_1000msec[0, 1, :] = vals[1, 0, :]  # O

vals_1000msec[1, 0, :] = vals_different[0, 0, :]  # U
vals_1000msec[1, 1, :] = vals_different[1, 0, :]  # O

vals_1000msec[2, 0, :] = vals_same[0, 0, :]  # U
vals_1000msec[2, 1, :] = vals_same[1, 0, :]  # O

PT.multiple_barplots(np.moveaxis(vals_1000msec[1:, :, :], 0, 1), ['Un.', 'Occ.'], 1, 2, '1000msec-averages', 4., 2.5, colors=np.array([colors[0]]*3), verbose=False, pdf=False, summary_data=True, set_title=['Different\n Category', 'Same\n Category'])

# 2000sec [OVERALL, SAME, DIFFERENT]
vals_2000msec = np.zeros((3, 2, 2))
vals_2000msec[0, 0, :] = vals[0, 1, :]  # U
vals_2000msec[0, 1, :] = vals[1, 1, :]  # O

vals_2000msec[1, 0, :] = vals_different[0, 1, :]  # U
vals_2000msec[1, 1, :] = vals_different[1, 1, :]  # O

vals_2000msec[2, 0, :] = vals_same[0, 1, :]  # U
vals_2000msec[2, 1, :] = vals_same[1, 1, :]  # O

PT.multiple_barplots(np.moveaxis(vals_2000msec[1:, :, :], 0, 1), ['Un.', 'Occ.'], 1, 2, '2000msec-averages', 4., 2.5, colors=np.array([colors[1]]*3), verbose=False, pdf=False, summary_data=True, set_title=['Different\n Category', 'Same\n Category'])


# UNTIL RESPONSE PERFORMANCE [OVERALL, SAME, DIFFERENT]
vals_ur = np.zeros((3, 2, 2))
vals_ur[0, 0, :] = vals[0, 2, :]  # U
vals_ur[0, 1, :] = vals[1, 2, :]  # O

vals_ur[1, 0, :] = vals_different[0, 2, :]  # U
vals_ur[1, 1, :] = vals_different[1, 2, :]  # O

vals_ur[2, 0, :] = vals_same[0, 2, :]  # U
vals_ur[2, 1, :] = vals_same[1, 2, :]  # O

PT.multiple_barplots(np.moveaxis(vals_ur[1:, :, :], 0, 1), ['Un.', 'Occ.'], 1, 2, 'until-response-averages', 4., 2.5, colors=np.array([colors[2]]*3), verbose=False, pdf=False, summary_data=True, set_title=['Different\n Category', 'Same\n Category'])

behavioral_averages = {}
behavioral_averages['1000msec'] = np.squeeze(vals_1000msec[1:,:,0])
behavioral_averages['2000msec'] = np.squeeze(vals_2000msec[1:,:,0])
behavioral_averages['UR'] = np.squeeze(vals_ur[1:,:,0])
pickle.dump(behavioral_averages, open('behavioral_averages_per_conditions.pkl', 'wb'))



### For each presentation time, compare the performance across the unoccluded and occluded groups just for the harder same-category trials
same_indices = []
diff_indices = []
for i in range(20):
    for j in range(6):
        if i % 2 == 0:
            same_indices.append(i * 6 + j)
        else:
            diff_indices.append(i * 6 + j)


def trial_type(row):
    if row['stimulus'] in same_indices:
        return 'same'
    else:
        return 'different'

df['trial_type'] = df.apply(lambda row: trial_type(row), axis=1)



# compare performance between unoccluded and occluded under harder trials
df_mean_type = pd.pivot_table(df,
                         values=['correct'],
                              index=['workerid', 'condition', 'stim_present', 'trial_type'],
                         aggfunc=np.mean)
"""
PRINT OUT SIGNIFICANT DIFFERENTCES; 
1SEC  Un to Occ
2SEC
Unlimited
"""
stim_present_levels = ['1000msecs', '2000msecs', 'UntilResponse']

for t in range(3):
    curr_stim_present = stim_present_levels[t]
    tsame = 'same'
    tdiff = 'different'
    a = np.array(df_mean_type.query('condition == 0 and stim_present == [@curr_stim_present] and trial_type == [@tsame]')).flatten()
    b = np.array(df_mean_type.query('condition == 1 and stim_present == [@curr_stim_present] and trial_type == [@tsame]')).flatten()
    print(' T: ' + stim_present_levels[t] + ' Average of Unocc: ' + str(np.mean(a)) + ' Average of Occ ' + str(np.mean(b)) + 't-test: ') 
    minN = min(a.shape, b.shape)[0]
    print(minN)
    print(ttest_ind(a[:minN],b[:minN]))


print('t-test of the main effect of occlusion condition')
a = np.array(df_mean.query('condition == 0')).flatten()
b = np.array(df_mean.query('condition == 1')).flatten()
minN = min(a.shape, b.shape)[0]
print(minN)
print(ttest_ind(a[:minN],b[:minN]))

# following bootstrap procedure takes a while to run
#quit()

### BOOTSTRAP For distance summary plot in fig 4
S = 10000
behavioral_averages_bootstrap = np.zeros((3, S, 4))

for s in range(S):
    if s % 10 == 0:
        print(s)

    for t in range(3):
        out = np.zeros(4)
        curr_stim_present = stim_present_levels[t]

        df_curr = df.query('condition == 0 and stim_present == [@curr_stim_present]')
        curr_wlist = np.unique(np.array(df_curr.workerid))
        wlist_sampled = np.random.choice(curr_wlist, len(curr_wlist))
        df_sampled = df_curr[df_curr.workerid == wlist_sampled[0]]
        for w in wlist_sampled[1:]:
            df_sampled = df_sampled.append(df_curr[df_curr.workerid == w])
        df_overall = pd.pivot_table(df_sampled, values = ['correct'], index = ['trial_type'], aggfunc = np.mean)
        out[[0, 2]] = np.array(df_overall).flatten()


        df_curr = df.query('condition == 1 and stim_present == [@curr_stim_present]')
        curr_wlist = np.unique(np.array(df_curr.workerid))
        wlist_sampled = np.random.choice(curr_wlist, len(curr_wlist))
        df_sampled = df_curr[df_curr.workerid == wlist_sampled[0]]
        for w in wlist_sampled[1:]:
            df_sampled = df_sampled.append(df_curr[df_curr.workerid == w])
        df_overall = pd.pivot_table(df_sampled, values = ['correct'], index = ['trial_type'], aggfunc = np.mean)
        out[[1, 3]] = np.array(df_overall).flatten()

            
        behavioral_averages_bootstrap[t, s, :] = out


np.save('behavioral_averages_per_conditions_bootstrap.npy', behavioral_averages_bootstrap)
