import numpy as np
import pandas as pd
from scipy.stats import sem
import pickle

import argparse
import sys

sys.path.append('./')
from utils.plotting_tools import *

parser = argparse.ArgumentParser(description='Process raw psiturk data')
parser.add_argument('--input-file',  type = str,
                    default='./processed_data_anonymous.csv',
                    help='the processed input csv file containing the behavioral data')
parser.add_argument('--output-folder', type = str,
                    default = './output/',
                    help='where to store figures and things')

args = parser.parse_args()

dfs = {}
T = 71./120 # 0.; p<0.05

df = pd.read_csv(args.input_file, header=0)
df.columns = ['workerid', 'condition', 'stim_present', 'stimulus', 'response', 'correct', 'rt']
df['stim_present'].replace({1000:'1000msecs', 2000:'2000msecs', 0:'UntilResponse'}, inplace=True)

df_mean = pd.pivot_table(df, values = ['correct'], index = ['workerid'], aggfunc = np.mean)
wlist_above_chance = df_mean[df_mean.iloc[:, 0] > T].index
df = df[df.workerid.isin(wlist_above_chance)]


df_overall = pd.pivot_table(df, values = ['correct'], index = ['stimulus', 'condition', 'stim_present'], aggfunc = np.mean)

behavior = {}
behavior['U'] = np.vstack([np.array(df_overall.query('condition == 0 and stim_present == ["1000msecs"]')).flatten(),
                             np.array(df_overall.query('condition == 0 and stim_present == ["2000msecs"]')).flatten(),
                             np.array(df_overall.query('condition == 0 and stim_present == ["UntilResponse"]')).flatten()])
behavior['O'] = np.vstack([np.array(df_overall.query('condition == 1 and stim_present == ["1000msecs"]')).flatten(),
                             np.array(df_overall.query('condition == 1 and stim_present == ["2000msecs"]')).flatten(),
                             np.array(df_overall.query('condition == 1 and stim_present == ["UntilResponse"]')).flatten()])

pickle.dump(behavior, open('behavioral_data_per_stimuli.pkl', 'wb'))




#### BOOTSTRAP samples -- sample subjects with replacement
S = 5000
behavior_bootstrap = {}
behavior_bootstrap['U'] = {}
behavior_bootstrap['O'] = {}

stim_present_levels = ['1000msecs', '2000msecs', 'UntilResponse']
df_dict = {}
df_dict[0] = {}
df_dict[1] = {}
for c in range(2):
    for t in range(3):
        curr_stim_present = stim_present_levels[t]
        df_dict[c][t] = df.query('condition == @c and stim_present == [@curr_stim_present]')


for s in range(S):
    if s % 10 == 0:
        print(s)

    for c in range(2):
        out = []
        if c == 0:
            curr_condition = 'U'
        else:
            curr_condition = 'O'

        for t in range(3):
            curr_stim_present = stim_present_levels[t]
            df_curr = df_dict[c][t]
            curr_wlist = np.unique(np.array(df_curr.workerid))
            wlist_sampled = np.random.choice(curr_wlist, len(curr_wlist))
            df_sampled = df_curr[df_curr.workerid == wlist_sampled[0]]
            for w in wlist_sampled[1:]:
                df_sampled = df_sampled.append(df_curr[df_curr.workerid == w])
            df_overall = pd.pivot_table(df_sampled, values = ['correct'], index = ['stimulus'], aggfunc = np.mean)
            if len(out) == 0:
                out = np.array(df_overall).flatten()
            else:
                out = np.vstack((out, np.array(df_overall).flatten()))
            
        behavior_bootstrap[curr_condition][s] = out

pickle.dump(behavior_bootstrap, open('behavioral_data_per_stimuli_bootstrap_subjects.pkl', 'wb'))



#### SPLIT-HALF CORRELATIONS AT GRANULATIRY 
same_indices = []
diff_indices = []
for i in range(20):
    for j in range(6):
        if i % 2 == 0:
            same_indices.append(i * 6 + j)
        else:
            diff_indices.append(i * 6 + j)


split_half_correlation_granular = np.zeros((2, 2500))
for c in range(2):
    if c == 0:
        curr_condition = 'U'
    else:
        curr_condition = 'O'

    for s in range(2500):

        behavior1 = behavior_bootstrap[curr_condition][s][2].flatten()[same_indices]
        behavior2 = behavior_bootstrap[curr_condition][2500 + s][2].flatten()[same_indices]
        split_half_correlation_granular[c, s] = np.corrcoef(behavior1, behavior2)[0,1]

print(split_half_correlation_granular.mean(axis=1))




'''
### ESTIMATE SPLIT-HALF correlation in the data
S = 100
behavior_first = {}
behavior_second = {}
split_half_correlations = {}
split_half_correlations = np.zeros((2, 3, S))
for s in range(S):
    indices = np.random.permutation(list(range(len(wlist))))
    df_sampled = df[df.workerid == wlist[indices[0]]]
    half = int(len(wlist)/2)
    for i in range(1, half):
        df_sampled = df_sampled.append(df[df.workerid == wlist[indices[i]]])

    df_overall = pd.pivot_table(df_sampled, values = ['correct'], index = ['stimulus', 'condition', 'stim_present'], aggfunc = np.mean)
    behavior_first['U'] = np.vstack([np.array(df_overall.query('condition == 0 and stim_present == ["1000msecs"]')).flatten(),
                                       np.array(df_overall.query('condition == 0 and stim_present == ["2000msecs"]')).flatten(),
                                       np.array(df_overall.query('condition == 0 and stim_present == ["UntilResponse"]')).flatten()])
    behavior_first['O']= np.vstack([np.array(df_overall.query('condition == 1 and stim_present == ["1000msecs"]')).flatten(),
                                      np.array(df_overall.query('condition == 1 and stim_present == ["2000msecs"]')).flatten(),
                                      np.array(df_overall.query('condition == 1 and stim_present == ["UntilResponse"]')).flatten()])

    df_sampled = df[df.workerid == wlist[indices[half]]]
    for i in range(half+1, len(wlist)):
        df_sampled = df_sampled.append(df[df.workerid == wlist[indices[i]]])

    df_overall = pd.pivot_table(df_sampled, values = ['correct'], index = ['stimulus', 'condition', 'stim_present'], aggfunc = np.mean)
    behavior_second['U'] = np.vstack([np.array(df_overall.query('condition == 0 and stim_present == ["1000msecs"]')).flatten(),
                                       np.array(df_overall.query('condition == 0 and stim_present == ["2000msecs"]')).flatten(),
                                       np.array(df_overall.query('condition == 0 and stim_present == ["UntilResponse"]')).flatten()])
    behavior_second['O']= np.vstack([np.array(df_overall.query('condition == 1 and stim_present == ["1000msecs"]')).flatten(),
                                      np.array(df_overall.query('condition == 1 and stim_present == ["2000msecs"]')).flatten(),
                                      np.array(df_overall.query('condition == 1 and stim_present == ["UntilResponse"]')).flatten()])
        
    for k, c in enumerate(['U', 'O']):
        for time in range(3):
            split_half_correlations[k, time, s] = np.corrcoef(behavior_first[c][time], behavior_second[c][time])[0,1]
'''
