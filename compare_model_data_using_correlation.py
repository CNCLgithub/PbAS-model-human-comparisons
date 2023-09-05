import numpy as np
from sklearn.preprocessing import MinMaxScaler
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


R = 32
N = 200

triplets = [48, 80, 110] # PbAS
triplets_pixels = [84, 116, 140] # no image encoding

mask = np.zeros(240, dtype='int')
mask[120:240] = 1
submask = np.zeros(120, dtype='int')
submask[same_indices] = 1


behavior_conditions = ['U', 'O']
mean_accuracy = {}
mean_accuracy['PbAS'] = np.zeros((2, 3, 120))
mean_accuracy['BU'] = np.zeros((2, 3, 120))
mean_accuracy['FT'] = np.zeros((2, 3, 120))
mean_accuracy['no-encoding'] = np.zeros((2, 3, 120))
mean_accuracy['data'] = np.zeros((2, 3, 120))

behavior_data = pickle.load(open('behavioral_data_per_stimuli.pkl', 'rb'))
pbas_anp = np.load('pbas-full-model-anp.npy')
no_encoding_anp = np.load('td-no-image-encoding-anp.npy')


PT = PlottingTools()
colors = np.array([(161,218,180), (65,182,196), (34,94,168)])/255. # orig defunct
marker_colors = np.array([(161,218,180), (166,206,227), (67,162,202)])/255. #updated


S = 5000
behavior_data_bootstrap = pickle.load(open('behavioral_data_per_stimuli_bootstrap_subjects.pkl', 'rb'))

correlation_to_behavior_triplets = np.zeros((2, 3, S)) 
correlation_to_behavior_triplets_pixels = np.zeros((2, 3, S)) 

correlation_to_behavior_triplets_sc = np.zeros((2, 3, S)) 
correlation_to_behavior_triplets_pixels_sc = np.zeros((2, 3, S)) 

for time in range(3):
    for c, condition in enumerate(behavior_conditions):
        model = np.mean(pbas_anp[c, :, :, triplets[time]], axis=0).flatten()
        model_no_encoding = np.mean(no_encoding_anp[c, :, :, triplets_pixels[time]], axis=0).flatten()
        mean_accuracy['PbAS'][c, time, :] = model
        mean_accuracy['no-encoding'][c, time, :] = model_no_encoding
        mean_accuracy['data'][c, time, :] = behavior_data[condition][time]
        for s in range(S):
            behavior = behavior_data_bootstrap[condition][s][time]
            behavior = behavior.flatten()
            correlation_to_behavior_triplets[c, time, s] = np.corrcoef(model, behavior)[0,1]
            correlation_to_behavior_triplets_pixels[c, time, s] = np.corrcoef(model_no_encoding, behavior)[0,1]
            correlation_to_behavior_triplets_sc[c, time, s] = np.corrcoef(model[same_indices], behavior[same_indices])[0,1]
            correlation_to_behavior_triplets_pixels_sc[c, time, s] = np.corrcoef(model_no_encoding[same_indices], behavior[same_indices])[0,1]


#### compare pretrained and model on the Unlimited data

anp_network = pickle.load(open('./fine-tuned-anp.pkl', 'rb'))

anp_network_pr = pickle.load(open('./pretrained-anp.pkl', 'rb'))

finetuned_network_correlation_to_behavior = np.zeros((2, 3, S)) 
pretrained_network_correlation_to_behavior = np.zeros((2, 3, S)) 
network_condition_labels = ['unoccluded', 'occluded']

finetuned_network_correlation_to_behavior_sc = np.zeros((2, 3, S)) 
pretrained_network_correlation_to_behavior_sc = np.zeros((2, 3, S)) 

for time in range(3):
    
    for c, condition in enumerate(behavior_conditions):
        network_condition_label = network_condition_labels[c]
        model = anp_network['finetune'][network_condition_label]['overall'].flatten() # using average performance (instead of correlations b/w embeddings) results in negative correlation
        mean_accuracy['FT'][c, time, :] = model
        for s in range(S):
            behavior = behavior_data_bootstrap[condition][s][time]
            behavior = behavior.flatten()
            finetuned_network_correlation_to_behavior[c, time, s] = np.corrcoef(model, behavior)[0,1]
            finetuned_network_correlation_to_behavior_sc[c, time, s] = np.corrcoef(model[same_indices], behavior[same_indices])[0,1]

    for c, condition in enumerate(behavior_conditions):
        network_condition_label = network_condition_labels[c]
        d1 = np.zeros(120)
        d2 = np.zeros(120)
        for k in range(120):
            targetGtEmbed = anp_network_pr[network_condition_label]['boRuns'][0]['targetGtEmbed'][k]
            targetDisEmbed = anp_network_pr[network_condition_label]['boRuns'][0]['targetDisEmbed'][k]
            studyEmbed = anp_network_pr[network_condition_label]['boRuns'][0]['studyEmbed'][k]
            d1[k] = np.corrcoef(studyEmbed, targetGtEmbed)[0,1]
            d2[k] = np.corrcoef(studyEmbed, targetDisEmbed)[0, 1]
        model = d1 / (d1 + d2 + 1e-4)
        model = model.flatten()
        mean_accuracy['BU'][c, time, :] = model
        for s in range(S):
            behavior = behavior_data_bootstrap[condition][s][time]
            behavior = behavior.flatten()
            pretrained_network_correlation_to_behavior[c, time, s] = np.corrcoef(model, behavior)[0,1]
            pretrained_network_correlation_to_behavior_sc[c, time, s] = np.corrcoef(model[same_indices], behavior[same_indices])[0,1]



### PRINT SAME CATEGORY CORRELATION PERCENTILES
for time in range(3):
    print("Presentation time: " + str(time))
    print("PbAS - mean")
    print([np.mean(x) for x in correlation_to_behavior_triplets_sc[:, time, :]])
    print("PbAS - 97.5")
    print([np.percentile(x, 97.5) for x in correlation_to_behavior_triplets_sc[:, time, :]])
    print("PbAS - 2.5")
    print([np.percentile(x, 2.5) for x in correlation_to_behavior_triplets_sc[:, time, :]])
    print("No enc - mean")
    print([np.mean(x) for x in correlation_to_behavior_triplets_pixels_sc[:, time, :]])
    print("No Enc. - 97.5")
    print([np.percentile(x, 97.5) for x in correlation_to_behavior_triplets_pixels_sc[:, time, :]])
    print("No Enc - 2.5")
    print([np.percentile(x, 2.5) for x in correlation_to_behavior_triplets_pixels_sc[:, time, :]])
    print("Bottom-up - mean")
    print([np.mean(x) for x in pretrained_network_correlation_to_behavior_sc[:, time, :]])
    print("Bottom-up - 97.5")
    print([np.percentile(x, 97.5) for x in pretrained_network_correlation_to_behavior_sc[:, time, :]])
    print("Bottom-up - 2.5")
    print([np.percentile(x, 2.5) for x in pretrained_network_correlation_to_behavior_sc[:, time, :]])
    print("Finetuned - mean")
    print([np.mean(x) for x in finetuned_network_correlation_to_behavior_sc[:, time, :]])
    print("Finetuned - 97.5")
    print([np.percentile(x, 97.5) for x in finetuned_network_correlation_to_behavior_sc[:, time, :]])
    print("Finetued - 2.5")
    print([np.percentile(x, 2.5) for x in finetuned_network_correlation_to_behavior_sc[:, time, :]])


unlimited = np.zeros((2, 4, S))
unlimited[:, 0, :] = correlation_to_behavior_triplets[:, 2, :]
unlimited[:, 1, :] = pretrained_network_correlation_to_behavior[:, 2, :]
unlimited[:, 2, :] = finetuned_network_correlation_to_behavior[:, 2, :]
unlimited[:, 3, :] = correlation_to_behavior_triplets_pixels[:, 2, :]


# Direct bootstrap hypothesis testing on the unlimited correlations
print("PbAS to FT - Unoccluded")
print(np.sum(unlimited[0, 0, :] - unlimited[0, 2, :] > 0))

print("PbAS to PreTr - Unoccluded")
print(np.sum(unlimited[0, 0, :] - unlimited[0, 1, :] > 0))

print("PbAS to Pixels - Unoccluded")
print(np.sum(unlimited[0, 0, :] - unlimited[0, 3, :] > 0))


print("PbAS to FT - Occluded")
print(np.sum(unlimited[1, 0, :] - unlimited[1, 2, :] > 0))

print("PbAS to PreTr - Occluded")
print(np.sum(unlimited[1, 0, :] - unlimited[1, 1, :] > 0))

print("PbAS to Pixels - Occluded")
print(np.sum(unlimited[1, 0, :] - unlimited[1, 3, :] > 0))


marker_colors_for_pixels = np.array([(210, 204, 227), (188,189,220), (117,107,177)])/255.
marker_colors_for_no_uncertainty = np.array([(254,224,210), (252,146,114), (222,45,38)])/255.

corr_summary_colors = np.zeros((3, 4, 3))
corr_summary_colors[0] = np.array([colors[2], (0.7, 0.7, 0.7), (0.5, 0.5, 0.5), marker_colors_for_pixels[2]])
corr_summary_colors[1] = np.array([colors[2], (0.7, 0.7, 0.7), (0.5, 0.5, 0.5), marker_colors_for_pixels[2]])


distance_summary_colors = np.zeros((3, 4, 3))
distance_summary_colors[0] = np.array([marker_colors[0], (.7, .7, .7), (.5, .5, .5), marker_colors[0]])
distance_summary_colors[1] = np.array([marker_colors[1], (.7, .7, .7), (.5, .5, .5), marker_colors[1]])
distance_summary_colors[2] = np.array([marker_colors[2], (.7, .7, .7), (.5, .5, .5), marker_colors[2]])



unlimited = np.moveaxis(unlimited, 0, 1)
PT.multiple_barplots(unlimited, ['PbAS', 'Bottom-up', 'Finetuned', 'Pixel-PbAS'], 1, 2, 'trial-level-unlimited-models', 4., 4, colors=distance_summary_colors[2].squeeze(), hatches=['///', '', '', 'o'], verbose=False, pdf=False, summary_data=False, set_title=['Unoccluded', 'Occluded'])


# scatter plot

#scatter_colors = np.array([(0.,0.,1), (0., 0., 1.), (0.,0.,1)])
scatter_colors = np.array(["purple", "purple", "purple"])

scatter_mask = np.zeros(240, dtype='int')
scatter_mask[60:120] = 1
scatter_mask[120:180] = 2
scatter_mask[180:240] = 3

scaler = MinMaxScaler()
mean_accuracy['unoccluded'] = np.zeros((4, 60))
mean_accuracy['occluded'] = np.zeros((4, 60))
mean_accuracy['data_by_condition'] = np.zeros((2, 240))
mean_accuracy['unoccluded'][0, :] = scaler.fit_transform(mean_accuracy['PbAS'][0, 2, same_indices].reshape(-1,1)).flatten()
mean_accuracy['unoccluded'][1, :] = scaler.fit_transform(mean_accuracy['BU'][0, 2, same_indices].reshape(-1,1)).flatten()
mean_accuracy['unoccluded'][2, :] = scaler.fit_transform(mean_accuracy['FT'][0, 2, same_indices].reshape(-1,1)).flatten()
mean_accuracy['unoccluded'][3, :] = scaler.fit_transform(mean_accuracy['no-encoding'][0, 2, same_indices].reshape(-1,1)).flatten()

mean_accuracy['occluded'][0, :] = scaler.fit_transform(mean_accuracy['PbAS'][1, 2, same_indices].reshape(-1,1)).flatten()
mean_accuracy['occluded'][1, :] = scaler.fit_transform(mean_accuracy['BU'][1, 2, same_indices].reshape(-1,1)).flatten()
mean_accuracy['occluded'][2, :] = scaler.fit_transform(mean_accuracy['FT'][1, 2, same_indices].reshape(-1,1)).flatten()
mean_accuracy['occluded'][3, :] = scaler.fit_transform(mean_accuracy['no-encoding'][1, 2, same_indices].reshape(-1,1)).flatten()

mean_accuracy['data_by_condition'][0] = np.tile(mean_accuracy['data'][0, 2, same_indices], 4)
mean_accuracy['data_by_condition'][1] = np.tile(mean_accuracy['data'][1, 2, same_indices], 4)

#PT.scatter_plot_multipane(mean_accuracy['unoccluded'].flatten(), np.squeeze(mean_accuracy['data_by_condition'][0]).flatten(), 'scatter-unlimited-unoccluded', 2.5, 10.0, mask=scatter_mask, colors=corr_summary_colors[0], verbose=True, pdf=False, submask=None, regline=True)

#PT.scatter_plot_multipane(mean_accuracy['occluded'].flatten(), np.squeeze(mean_accuracy['data_by_condition'][1]).flatten(), 'scatter-unlimited-occluded', 2.5, 10., mask=scatter_mask, colors=corr_summary_colors[0], verbose=True, pdf=False, submask=None, regline=True)


twosec = np.zeros((2, 4, S))
twosec[:, 0, :] = correlation_to_behavior_triplets[:, 1, :]
twosec[:, 1, :] = pretrained_network_correlation_to_behavior[:, 1, :]
twosec[:, 2, :] = finetuned_network_correlation_to_behavior[:, 1, :]
twosec[:, 3, :] = correlation_to_behavior_triplets_pixels[:, 1, :]


# Direct bootstrap hypothesis testing on the unlimited correlations
print("PbAS to FT - Unoccluded")
print(np.sum(twosec[0, 0, :] - twosec[0, 2, :] > 0))

print("PbAS to PreTr - Unoccluded")
print(np.sum(twosec[0, 0, :] - twosec[0, 1, :] > 0))

print("PbAS to Pixels - Unoccluded")
print(np.sum(twosec[0, 0, :] - twosec[0, 3, :] > 0))


print("PbAS to FT - Occluded")
print(np.sum(twosec[1, 0, :] - twosec[1, 2, :] > 0))

print("PbAS to PreTr - Occluded")
print(np.sum(twosec[1, 0, :] - twosec[1, 1, :] > 0))

print("PbAS to Pixels - Occluded")
print(np.sum(twosec[1, 0, :] - twosec[1, 3, :] > 0))


corr_summary_colors[0] = np.array([colors[1], (0.7, 0.7, 0.7), (0.5, 0.5, 0.5), marker_colors_for_pixels[1]])
corr_summary_colors[1] = np.array([colors[1], (0.7, 0.7, 0.7), (0.5, 0.5, 0.5), marker_colors_for_pixels[1]])

twosec = np.moveaxis(twosec, 0, 1)
PT.multiple_barplots(twosec, ['PbAS', 'Bottom-up', 'Finetuned', 'Pixel-PbAS'],1, 2, 'trial-level-twosec-models', 4., 4., colors=distance_summary_colors[1].squeeze(), hatches=['///', '', '', 'o'], verbose=False, pdf=False, summary_data=False, set_title=['Unoccluded', 'Occluded'])


onesec = np.zeros((2, 4, S))
onesec[:, 0, :] = correlation_to_behavior_triplets[:, 0, :]
onesec[:, 1, :] = pretrained_network_correlation_to_behavior[:, 0, :]
onesec[:, 2, :] = finetuned_network_correlation_to_behavior[:, 0, :]
onesec[:, 3, :] = correlation_to_behavior_triplets_pixels[:, 0, :]

# Direct bootstrap hypothesis testing on the unlimited correlations
print("PbAS to FT - Unoccluded")
print(np.sum(onesec[0, 0, :] - onesec[0, 2, :] > 0))

print("PbAS to PreTr - Unoccluded")
print(np.sum(onesec[0, 0, :] - onesec[0, 1, :] > 0))

print("PbAS to Pixels - Unoccluded")
print(np.sum(onesec[0, 0, :] - onesec[0, 3, :] > 0))


print("PbAS to FT - Occluded")
print(np.sum(onesec[1, 0, :] - onesec[1, 2, :] > 0))

print("PbAS to PreTr - Occluded")
print(np.sum(onesec[1, 0, :] - onesec[1, 1, :] > 0))

print("PbAS to Pixels - Occluded")
print(np.sum(onesec[1, 0, :] - onesec[1, 3, :] > 0))


corr_summary_colors[0] = np.array([colors[0], (0.7, 0.7, 0.7), (0.5, 0.5, 0.5), marker_colors_for_pixels[0]])
corr_summary_colors[1] = np.array([colors[0], (0.7, 0.7, 0.7), (0.5, 0.5, 0.5), marker_colors_for_pixels[0]])

onesec = np.moveaxis(onesec, 0, 1)
PT.multiple_barplots(onesec, ['PbAS', 'Bottom-up', 'Finetuned', 'Pixel-PbAS'], 1, 2, 'trial-level-onesec-models', 4., 4., colors=distance_summary_colors[0].squeeze(), hatches=['///', '', '', 'o'], verbose=False, pdf=False, summary_data=False, set_title=['Unoccluded', 'Occluded'])



