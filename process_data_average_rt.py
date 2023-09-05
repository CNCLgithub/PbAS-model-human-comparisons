import numpy as np
import pandas as pd
from scipy import stats
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
T = 71. / 120 # p < .05

df = pd.read_csv(args.input_file, header=0)
df.columns = ['workerid', 'condition','stim_present','stimulus', 'response', 'correct', 'rt']
df['stim_present'].replace({1000: '1000msecs', 2000: '2000msecs', 0: 'UntilResponse'}, inplace=True)

df_mean = pd.pivot_table(df,
                         values=['correct'],
                         index=['workerid'],
                         aggfunc=np.mean)
wlist_above_chance = df_mean[df_mean.iloc[:, 0] > T].index
df = df[df.workerid.isin(wlist_above_chance)]

# OVERALL AVERAGE RT ANALYSIS


"""
PRINT OUT SIGNIFICANT DIFFERENTCES; 
UUU 1SEC TO 2SECS
UUU 1SEC TO UR
UUU 2SECS TO UR
UUO 1SEC TO 2SECS
UUO 1SEC TO UR
UUO 2SECS TO UR
"""
stim_present_levels = ['1000msecs', '2000msecs', 'UntilResponse']

for c in range(2):
    for t_short in range(2):
        for t_long in range(t_short+1, 3):
            stim_present = stim_present_levels[t_short]
            df_cur = df.query('condition == @c and stim_present == [@stim_present]')
            df_cur = df_cur[np.abs(stats.zscore(df_cur.rt)) < 3]
            df_cur = pd.pivot_table(df_cur, values = ['rt'], index = ['workerid', 'condition', 'stim_present'], aggfunc = np.mean)
            a = np.array(df_cur).flatten()

            stim_present = stim_present_levels[t_long]
            df_cur = df.query('condition == @c and stim_present == [@stim_present]')
            df_cur = df_cur[np.abs(stats.zscore(df_cur.rt)) < 3]
            df_cur = pd.pivot_table(df_cur, values = ['rt'], index = ['workerid', 'condition', 'stim_present'], aggfunc = np.mean)
            b = np.array(df_cur).flatten()
            print('Condition: ' + str(c) + ' T1: ' + stim_present_levels[t_short] + ' T2: ' + stim_present_levels[t_long] + ' t-test: ') 
            minN = min(a.shape, b.shape)[0]
            print('Average RTs')
            print(a.mean())
            print(b.mean())
            print(minN)
            print(ttest_ind(a[:minN],b[:minN]))


colors = np.array([(161,218,180), (166,206,227), (67,162,202)])/255.

vals = np.zeros((3, 2, 24))

for s, stim_present in enumerate(["1000msecs", "2000msecs", "UntilResponse"]):
    for c in range(2):
        df_cur = df.query('condition == @c and stim_present == [@stim_present]')
        df_cur = df_cur[np.abs(stats.zscore(df_cur.rt)) < 3]
        df_cur = pd.pivot_table(df_cur, values = ['rt'], index = ['workerid', 'condition', 'stim_present'], aggfunc = np.mean)
        vals[s, c] = np.array(df_cur.rt)[:24]

PT = PlottingTools()
PT.multiple_violins(vals, ['1sec', '2secs', 'Unl.'], 1, 2, 'overall-averages-rt', 3., 2.5, colors=colors, verbose=False, pdf=False, summary_data=True, set_title=['UUU', 'UUO'], ylimits=((0, 5000)))


# MAIN EFFECT OF PRESENTATION TIME CONDITION ON RTs
for t_short in range(2):
    for t_long in range(t_short+1, 3):
        stim_present = stim_present_levels[t_short]
        df_cur = df.query('stim_present == [@stim_present]')
        df_cur = df_cur[np.abs(stats.zscore(df_cur.rt)) < 3]
        df_cur = pd.pivot_table(df_cur, values = ['rt'], index = ['workerid', 'condition', 'stim_present'], aggfunc = np.mean)
        a = np.array(df_cur).flatten()

        stim_present = stim_present_levels[t_long]
        df_cur = df.query('stim_present == [@stim_present]')
        df_cur = df_cur[np.abs(stats.zscore(df_cur.rt)) < 3]
        df_cur = pd.pivot_table(df_cur, values = ['rt'], index = ['workerid', 'condition', 'stim_present'], aggfunc = np.mean)
        b = np.array(df_cur).flatten()

        minN = min(a.shape, b.shape)[0]
        print(minN)
        print('T1: ' + stim_present_levels[t_short] + ' T2: ' + stim_present_levels[t_long] + ' t-test: ') 
        print(ttest_ind(a[:minN],b[:minN]))

vals_overall = np.reshape(vals, (3, 1, 48))

PT = PlottingTools()
PT.multiple_violins(vals_overall, ['1sec', '2secs', 'Unl.'], 1, 1, 'overall-overall-averages-rt', 3., 1.5, colors=colors, verbose=False, pdf=False, summary_data=True, ylimits=((0, 5000)))

