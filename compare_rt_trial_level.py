import numpy as np

import pickle

import sys
sys.path.append('./')
from utils.plotting_tools import *


same_indices = []
diff_indices = []
for i in range(20):
    for j in range(6):
        if i % 2 == 0:
            same_indices.append(i * 6 + j)
        else:
            diff_indices.append(i * 6 + j)

S = 120

mask = np.zeros(S * 2, dtype='int')
mask[S:2*S] = 1
submask = np.zeros(S, dtype='int')
submask[same_indices] = 1

N = 200

correlation_to_behavior = np.zeros((2, N)) 
behavior_conditions = ['U', 'O']
behavior_data = pickle.load(open('behavioral_data_per_stimuli.pkl', 'rb'))
rt_data = pickle.load(open('behavioral_rt_data_per_stimuli.pkl', 'rb'))

TIMECOND = 2

rt = np.hstack([rt_data['U'][TIMECOND], rt_data['O'][TIMECOND]])

PT = PlottingTools()
colors = np.array([(161,218,180), (65,182,196), (34,94,168)])/255.
marker_colors_for_pixels = np.array([(210, 204, 227), (188,189,220), (117,107,177)])/255
marker_colors_for_no_uncertainty = np.array([(254,224,210), (252,146,114), (222,45,38)])/255.

def makedecision(model, behavior):
    ind = len(model) - 1
    for i in range(len(model)):
        val = np.mean(model[i])
        if val >= behavior:
            ind = i
            break
    return ind


iteration_counts = np.zeros((2, 120))
perf_result = np.zeros((2, 120))
iteration_counts_pixels = np.zeros((2, 120))
perf_result_pixels = np.zeros((2, 120))
iteration_counts_ft = np.zeros((2, 120))
perf_result_ft = np.zeros((2, 120))

counter = 0

indices = range(120)

pbas_anp = np.load('pbas-full-model-anp.npy')
no_encoding_anp = np.load('td-no-image-encoding-anp.npy')

anp_network = pickle.load(open('fine-tuned-anp.pkl', 'rb'))

for i in indices:
    for c, condition in enumerate(['unoccluded', 'occluded']):
        model = np.mean(pbas_anp[c, :, i, :], axis=0).flatten()
        model_no_encoding = np.mean(no_encoding_anp[c, :, i, :], axis=0).flatten()
        model_ft = np.mean(anp_network['finetune'][condition]['overallPerformancePerBORun'][:, i, :], axis=0).flatten()
        behavior_mean = np.mean(behavior_data[behavior_conditions[c]][TIMECOND])
        iteration_counts[c, i] = makedecision(model, behavior_mean)
        perf_result[c, i] = model[int(iteration_counts[c, i])]
        iteration_counts_pixels[c, i] = makedecision(model_no_encoding, behavior_mean)
        perf_result_pixels[c, i] = model_no_encoding[int(iteration_counts_pixels[c, i])]
        iteration_counts_ft[c, i] = makedecision(model_ft, behavior_mean)
        perf_result_ft[c, i] = model_ft[int(iteration_counts_ft[c, i])]


iteration_counts += 1
iteration_counts = np.log(iteration_counts)
iteration_counts_pixels += 1
iteration_counts_pixels = np.log(iteration_counts_pixels)
iteration_counts_ft += 1
iteration_counts_ft = np.log(iteration_counts_ft)

### BOOTSTRAP SUBJECTS FOR CORRELATIONS

S = 5000
behavior_data_bootstrap = pickle.load(open('behavioral_data_per_stimuli_bootstrap_subjects_for_rt.pkl', 'rb'))

correlation_to_behavior_triplets = np.zeros((2, S)) 
correlation_to_behavior_triplets_pixels = np.zeros((2, S)) 
correlation_to_behavior_triplets_ft = np.zeros((2, S)) 

correlation_to_behavior_triplets_sc = np.zeros((2, S)) 
correlation_to_behavior_triplets_pixels_sc = np.zeros((2, S)) 
correlation_to_behavior_triplets_ft_sc = np.zeros((2, S)) 



for c, condition in enumerate(['U', 'O']):
    model = iteration_counts[c, :]
    model_no_encoding = iteration_counts_pixels[c, :]
    model_ft = iteration_counts_ft[c, :]
    for s in range(S):
        behavior = behavior_data_bootstrap[condition][s][TIMECOND]
        behavior = behavior.flatten()
        correlation_to_behavior_triplets[c, s] = np.corrcoef(model, behavior)[0,1]
        correlation_to_behavior_triplets_pixels[c, s] = np.corrcoef(model_no_encoding, behavior)[0,1]
        correlation_to_behavior_triplets_ft[c, s] = np.corrcoef(model_ft, behavior)[0,1]

        correlation_to_behavior_triplets_sc[c, s] = np.corrcoef(model[same_indices], behavior[same_indices])[0,1]
        correlation_to_behavior_triplets_pixels_sc[c, s] = np.corrcoef(model_no_encoding[same_indices], behavior[same_indices])[0,1]
        correlation_to_behavior_triplets_ft_sc[c, s] = np.corrcoef(model_ft[same_indices], behavior[same_indices])[0,1]



### PRINT SAME CATEGORY CORRELATION PERCENTILES
print("PbAS - mean")
print([np.mean(x) for x in correlation_to_behavior_triplets])
print("PbAS - 97.5")
print([np.percentile(x, 97.5) for x in correlation_to_behavior_triplets])
print("PbAS - 2.5")
print([np.percentile(x, 2.5) for x in correlation_to_behavior_triplets])
print("No enc - mean")
print([np.mean(x) for x in correlation_to_behavior_triplets_pixels])
print("No Enc. - 97.5")
print([np.percentile(x, 97.5) for x in correlation_to_behavior_triplets_pixels])
print("No Enc - 2.5")
print([np.percentile(x, 2.5) for x in correlation_to_behavior_triplets_pixels])
print("FT - mean")
print([np.mean(x) for x in correlation_to_behavior_triplets_ft])
print("FT - 97.5")
print([np.percentile(x, 97.5) for x in correlation_to_behavior_triplets_ft])
print("FT - 2.5")
print([np.percentile(x, 2.5) for x in correlation_to_behavior_triplets_ft])


unlimited = np.zeros((2, 2, S))
unlimited[:, 0, :] = correlation_to_behavior_triplets
unlimited[:, 1, :] = correlation_to_behavior_triplets_pixels

corr_summary_colors = np.zeros((3, 2, 3))
corr_summary_colors[0] = np.array([colors[2], marker_colors_for_pixels[2]])
corr_summary_colors[1] = np.array([colors[2], marker_colors_for_pixels[2]])

# Direct bootstrap hypothesis testing on the correlations
print("PbAS to No Encoding")
print(np.sum(unlimited[0, 0, :] - unlimited[0, 1, :] > 0))
print(np.sum(unlimited[1, 0, :] - unlimited[1, 1, :] > 0))

print("PbAS to FT")
print(np.sum(unlimited[0, 0, :] - correlation_to_behavior_triplets_ft[0, :] > 0))
print(np.sum(unlimited[1, 0, :] - correlation_to_behavior_triplets_ft[1, :] > 0))

unlimited = np.moveaxis(unlimited, 0, 1)
PT.multiple_barplots(unlimited, ['PbAS', 'No-Enc.'], 1, 2, 'trial-level-rt-bootstrap', 5., 3., colors=corr_summary_colors, verbose=False, pdf=False, summary_data=False, set_title=['Unoccluded', 'Occluded'])

mean_rt = {}
mean_rt['unoccluded'] = np.zeros((2, 120))
mean_rt['occluded'] = np.zeros((2, 120))
mean_rt['data_by_condition'] = np.zeros((2,240))
mean_rt['unoccluded'] = np.vstack([iteration_counts[0], iteration_counts_pixels[0]])
mean_rt['occluded'] = np.vstack([iteration_counts[1], iteration_counts_pixels[1]])
mean_rt['data_by_condition'][0] = np.log(np.tile(rt[:120], 2))
mean_rt['data_by_condition'][1] = np.log(np.tile(rt[120:], 2))

scatter_mask = np.zeros(240, dtype='int')
scatter_mask[120:240] = 1

PT.scatter_plot_multipane(mean_rt['unoccluded'].flatten(), mean_rt['data_by_condition'][0].flatten(), 'scatter-unlimited-unoccluded-rt', 2.5, 5., mask=scatter_mask, colors=corr_summary_colors[0], verbose=True, pdf=False, submask=None, regline=True)

PT.scatter_plot_multipane(mean_rt['occluded'].flatten(), mean_rt['data_by_condition'][1].flatten(), 'scatter-unlimited-occluded-rt', 2.5, 5., mask=scatter_mask, colors=corr_summary_colors[0], verbose=True, pdf=False, submask=None, regline=True)


PT.scatter_plot_multipane(iteration_counts_ft[0].flatten(),  mean_rt['data_by_condition'][0][:120].flatten(), 'scatter-unlimited-unoccluded-FT-rt', 2.5, 2.5, mask=np.zeros(120, dtype='int'), colors=['black'], verbose=True, pdf=False, submask=None, regline=True)

PT.scatter_plot_multipane(iteration_counts_ft[1].flatten(),  mean_rt['data_by_condition'][1][:120].flatten(), 'scatter-unlimited-occluded-FT-rt', 2.5, 2.5, mask=np.zeros(120, dtype='int'), colors=['black'], verbose=True, pdf=False, submask=None, regline=True)



