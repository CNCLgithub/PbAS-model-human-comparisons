import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy import stats
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
T = 71./120 # p < .05

df = pd.read_csv(args.input_file, header=0)
df.columns = ['workerid', 'condition', 'stim_present', 'stimulus', 'response', 'correct', 'rt']
df['stim_present'].replace({1000:'1000msecs', 2000:'2000msecs', 0:'UntilResponse'}, inplace=True)

df_mean = pd.pivot_table(df, values = ['correct'], index = ['workerid'], aggfunc = np.mean)
wlist_above_chance = df_mean[df_mean.iloc[:, 0] > T].index
df = df[df.workerid.isin(wlist_above_chance)]


behavior = {}
behavior['U'] = np.zeros((3, 120))
behavior['O'] = np.zeros((3, 120))
condition_list = ['U', 'O']

for condition in range(2):
    for k, stim_present in enumerate(["1000msecs", "2000msecs", "UntilResponse"]):
        df_cur = df.query('condition == @condition and stim_present == [@stim_present]')
        df_cur = df_cur[np.abs(stats.zscore(df_cur.rt)) < 3]
        df_cur = pd.pivot_table(df_cur, values = ['rt'], index = ['stimulus', 'condition', 'stim_present'], aggfunc = np.mean)
        behavior[condition_list[condition]][k] = np.array(df_cur).flatten()


pickle.dump(behavior, open('behavioral_rt_data_per_stimuli.pkl', 'wb'))


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

    for c, curr_condition in enumerate(condition_list):
        out = []

        for t in range(3):
            curr_stim_present = stim_present_levels[t]
            df_curr = df_dict[c][t]
            curr_wlist = np.unique(np.array(df_curr.workerid))
            wlist_sampled = np.random.choice(curr_wlist, len(curr_wlist))
            df_sampled = df_curr[df_curr.workerid == wlist_sampled[0]]
            for w in wlist_sampled[1:]:
                df_sampled = df_sampled.append(df_curr[df_curr.workerid == w])

            df_sampled = df_sampled[np.abs(stats.zscore(df_sampled.rt)) < 3]
            df_overall = pd.pivot_table(df_sampled, values = ['rt'], index = ['stimulus'], aggfunc = np.mean)
            if len(out) == 0:
                out = np.array(df_overall).flatten()
            else:
                out = np.vstack((out, np.array(df_overall).flatten()))
            
        behavior_bootstrap[curr_condition][s] = out

pickle.dump(behavior_bootstrap, open('behavioral_data_per_stimuli_bootstrap_subjects_for_rt.pkl', 'wb'))

