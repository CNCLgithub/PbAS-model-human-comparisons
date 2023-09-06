import numpy as np

import pickle

from scipy.stats import ttest_ind

import sys
sys.path.append('./')
from utils.plotting_tools import *

# indices for the same category trials
same_indices = []
# indices for the different category trials
diff_indices = []
for i in range(20):
    for j in range(6):
        if i % 2 == 0:
            same_indices.append(i * 6 + j)
        else:
            diff_indices.append(i * 6 + j)

# number of PbAS runs
R = 32
# number of trials
N = 200

average_accuracy = np.zeros((N, 2, 2, 2)) #dimensions: N trials; diff vs same; Unoccluded vs Ooccluded; mean vs std
average_accuracy_no_encoding = np.zeros((N, 2, 2, 2))
average_accuracy_no_uncertainty = np.zeros((N, 2, 2, 2))
pretrained_average_accuracy = np.zeros((2, 2, 2))
finetuned_average_accuracy = np.zeros((2, 2, 2))


# load PbAS results
anp = np.load('pbas-full-model-anp.npy') # dim: np.zeros((2, 32, 120, 200)) -- # of conditions x # of runs of the model x # of trials x #r of iterations
no_encoding_anp = np.load('td-no-image-encoding-anp.npy')

for i in range(N):
    for c in range(2):

        # pbas
        accuracy = np.zeros(R)
        accuracy_diff_indices = np.zeros(R)
        accuracy_same_indices = np.zeros(R)

        for j in range(R):
            pred = anp[c, j, :, i] 
            accuracy[j] = np.mean(pred)
            accuracy_diff_indices[j] = np.mean(pred[diff_indices])
            accuracy_same_indices[j] = np.mean(pred[same_indices])

        average_accuracy[i, 0, c, 0] = np.mean(accuracy_diff_indices)
        average_accuracy[i, 0, c, 1] = np.std(accuracy_diff_indices)
        average_accuracy[i, 1, c, 0] = np.mean(accuracy_same_indices)
        average_accuracy[i, 1, c, 1] = np.std(accuracy_same_indices)

        # the no-encoding model (pixel-pbas)
        accuracy = np.zeros(R)
        accuracy_diff_indices = np.zeros(R)
        accuracy_same_indices = np.zeros(R)

        for j in range(R):
            pred = no_encoding_anp[c, j, :, i] 
            accuracy[j] = np.mean(pred)
            accuracy_diff_indices[j] = np.mean(pred[diff_indices])
            accuracy_same_indices[j] = np.mean(pred[same_indices])

        average_accuracy_no_encoding[i, 0, c, 0] = np.mean(accuracy_diff_indices)
        average_accuracy_no_encoding[i, 0, c, 1] = np.std(accuracy_diff_indices)
        average_accuracy_no_encoding[i, 1, c, 0] = np.mean(accuracy_same_indices)
        average_accuracy_no_encoding[i, 1, c, 1] = np.std(accuracy_same_indices)



# bottom-up network and the fine-tuned network  with 32 runs
anp_network = pickle.load(open('./fine-tuned-anp.pkl', 'rb'))

for c, condition in enumerate(['unoccluded', 'occluded']):
    pretrained = anp_network[condition]['overall'].flatten()
    pretrained_average_accuracy[0, c, 0] = np.mean(pretrained[diff_indices])
    pretrained_average_accuracy[0, c, 1] = 0 #np.std(pretrained[diff_indices])
    pretrained_average_accuracy[1, c, 0] = np.mean(pretrained[same_indices])
    pretrained_average_accuracy[1, c, 1] = 0 #np.std(pretrained[same_indices])

    accuracy = np.zeros(R)
    accuracy_diff_indices = np.zeros(R)
    accuracy_same_indices = np.zeros(R)

    for j in range(R):
        pred = anp_network['finetune'][condition]['overallPerformancePerBORun'][j]
        accuracy[j] = np.mean(pred)
        accuracy_diff_indices[j] = np.mean(pred[diff_indices])
        accuracy_same_indices[j] = np.mean(pred[same_indices])

    finetuned_average_accuracy[0, c, 0] = np.mean(accuracy_diff_indices)
    finetuned_average_accuracy[0, c, 1] = np.std(accuracy_diff_indices)
    finetuned_average_accuracy[1, c, 0] = np.mean(accuracy_same_indices)
    finetuned_average_accuracy[1, c, 1] = np.std(accuracy_same_indices)


# load behavioral data; the pkl file is generated using ./process_data_average_performance.py
behavioral_averages = pickle.load(open('behavioral_averages_per_conditions.pkl', 'rb'))
average_based_distances = np.zeros((3, N)) # 3 time conditions
average_based_distances_no_encoding = np.zeros((3, N)) # 3 time conditions
pretrained_distances = np.zeros(3)
finetuned_distances = np.zeros(3)

for t, time in enumerate(['1000msec', '2000msec', 'UR']):
    for i in range(N):
        average_based_distances[t, i] = np.linalg.norm(average_accuracy[i, :, :, 0].flatten() - behavioral_averages[time][:, :2].flatten())
        average_based_distances_no_encoding[t, i] = np.linalg.norm(average_accuracy_no_encoding[i, :, :, 0].flatten() - behavioral_averages[time][:, :2].flatten())


# the iterations that fit each presentation time condition  the best
triplet = [np.argmin(average_based_distances[0, :]), np.argmin(average_based_distances[1, :]), np.argmin(average_based_distances[2, :])]
print(triplet)
triplet_no_encoding = [np.argmin(average_based_distances_no_encoding[0, :]), np.argmin(average_based_distances_no_encoding[1, :]), np.argmin(average_based_distances_no_encoding[2, :])] #no image encoding
print(triplet_no_encoding)


PT = PlottingTools()
marker_colors = np.array([(161,218,180), (166,206,227), (67,162,202)])/255. #updated
marker_colors_for_pixels = np.array([(210, 204, 227), (188,189,220), (117,107,177)])/255.
colors = marker_colors

PT.multiple_lines(average_based_distances, average_based_distances*0, 'average-based-distances-across-iterations', 4.0, 5.0, horizontal_lines=None, verbose=False, y_limits=(0, 0.6), colors=marker_colors, pdf=False, start=0, end=N-1, stepsize=20, set_markers=triplet, marker_colors=marker_colors)

PT.multiple_lines(average_based_distances_no_encoding, average_based_distances*0, 'average-based-distances-no-encoding-across-iterations', 4.0, 5.0, horizontal_lines=None, verbose=False, y_limits=(0, 0.6), colors=marker_colors, pdf=False, start=0, end=N-1, stepsize=20, set_markers=triplet_no_encoding, marker_colors=marker_colors)


### DISTANCE SUMMARY BOOTSTRAP ###

vals = np.zeros((3, 4, 2)) # times, models, mean vs std
vals[:, 0, 0] = [average_based_distances[x] for x in zip(range(3), triplet)]

## Fig. 4
vals[:, 1, 0] = pretrained_distances
vals[:, 2, 0] = finetuned_distances
vals[:, 3, 0] = [average_based_distances_no_encoding[x] for x in zip(range(3), triplet_no_encoding)]
vals = np.moveaxis(vals, 0, 1)

# for plotting
distance_summary_colors = np.zeros((3, 4, 3))
distance_summary_colors[0] = np.array([marker_colors[0], (.7, .7, .7), (.5, .5, .5), marker_colors[0]])
distance_summary_colors[1] = np.array([marker_colors[1], (.7, .7, .7), (.5, .5, .5), marker_colors[1]])
distance_summary_colors[2] = np.array([marker_colors[2], (.7, .7, .7), (.5, .5, .5), marker_colors[2]])

# bootstrapped data generated using ./process_data_average_performance.py
behavioral_averages_bootstrap = np.load('behavioral_averages_per_conditions_bootstrap.npy')

S = 5000
all_models_average_based_distances = np.zeros((3, 4, S)) # 3 time conditions x 4 models x {mean, std}

for s in range(S):
    for t, time in enumerate(['1000msec', '2000msec', 'UR']):
        behavior_average_sample = behavioral_averages_bootstrap[t][s].flatten()
        for models in range(4):
            if models == 0:
                model_accuracy = average_accuracy[triplet[t], :, :, 0].flatten()
            elif models == 1:
                model_accuracy = pretrained_average_accuracy[:, :, 0].flatten()
            elif models == 2:
                model_accuracy = finetuned_average_accuracy[:, :, 0].flatten()
            elif models == 3:
                model_accuracy = average_accuracy_no_encoding[triplet_no_encoding[t], :, :, 0].flatten()

            all_models_average_based_distances[t, models, s] = np.linalg.norm(model_accuracy - behavior_average_sample)


## Direct bootstrap comparisons for Fig. 4F
for t, time in enumerate(['1000msec', '2000msec', 'UR']):
    print(time)
    for model in range(3):
        print('PbAS vs. model '+str(model+1))
        print(np.sum(all_models_average_based_distances[t, 0, :] - all_models_average_based_distances[t, model+1, :] > 0))

PT.multiple_barplots(np.moveaxis(all_models_average_based_distances, 0, 1), ['PbAS', 'BU', 'FT', 'Pixel-\n PbAS'], 1, 3, 'distances-summary', 5, 6, colors=distance_summary_colors, hatches=['///', '', '', 'o'], verbose=False, pdf=False, summary_data=False, set_title=['1 sec', '2 secs', 'Unlimited'], ylimits=(0, 0.6))


# PbAS: plot accuracy plots per iteration
triplet = [50, 80, 110] # for presentation in the paper for the PbAS model
colors = marker_colors
hatches = ['', '///', 'o']

for c, t in enumerate(triplet):
    vals = average_accuracy[t]
    vals = np.moveaxis(vals, 0, 1)
    if c > 2:
        color_index = 2
    else:
        color_index = c
    PT.multiple_barplots(vals, ['Un.', 'Occ.'], 1, 2, 'BOiteration PbAS'+str(c), 4., 2.5, colors=np.array(colors[color_index]), hatches=[hatches[1]]*2, verbose=False, pdf=False, summary_data=True, set_title=['Different\n Category', 'Same\n Category'])

# PbAS-Pixels: plot accuracy plots per iteration
triplet_no_encoding = [80, 110, 140]
colors = marker_colors
hatches = ['', '///', 'o']

for c, t in enumerate(triplet_no_encoding):
    vals = average_accuracy_no_encoding[t]
    vals = np.moveaxis(vals, 0, 1)
    if c > 2:
        color_index = 2
    else:
        color_index = c
    PT.multiple_barplots(vals, ['Un.', 'Occ.'], 1, 2, 'BOiteration PbAS-Pixels'+str(c), 4., 2.5, colors=np.array(colors[color_index]), hatches=[hatches[1]]*2, verbose=False, pdf=False, summary_data=True, set_title=['Different\n Category', 'Same\n Category'])

vals = pretrained_average_accuracy
vals = np.moveaxis(vals, 0, 1)
PT.multiple_barplots(vals, ['Un.', 'Occ.'], 1, 2, 'Pretrained', 4., 2.5, colors=np.array([(0.7, 0.7, 0.7)]*3), verbose=False, pdf=False, summary_data=True, set_title=['Different\n Category', 'Same\n Category'])

vals = finetuned_average_accuracy
vals = np.moveaxis(vals, 0, 1)
PT.multiple_barplots(vals, ['Un.', 'Occ.'], 1, 2, 'Finetuned', 4., 2.5, colors=np.array([(0.5, 0.5, 0.5)]*3), verbose=False, pdf=False, summary_data=True, set_title=['Different\n Category', 'Same\n Category'])



### BOOTSTRAP OF THE BEST FITTING ITERATION
S = 5000
average_based_distances = np.zeros((3, N)) # 3 time conditions
best_fitting_iterations = np.zeros((3, S))
for s in range(S):
    for t, time in enumerate(['1000msec', '2000msec', 'UR']):
        behavior_average_sample = behavioral_averages_bootstrap[t][s].flatten()
        for i in range(N):
            average_based_distances[t, i] = np.linalg.norm(average_accuracy[i, :, :, 0].flatten() - behavior_average_sample)
        best_fitting_iterations[t, s] = np.argmin(average_based_distances[t, :])
        
 
print('Comparing best fitting iterations across presentation time conditions')
print('average and intervals of best fitting iterations')
print('1 sec, 2 sec, UR mean' + str(np.mean(best_fitting_iterations, axis=1)))
print('1 sec, 2 sec, UR lower end' + str(np.percentile(best_fitting_iterations, 2.5, axis=1)))
print('1 sec, 2 sec, UR higher end' + str(np.percentile(best_fitting_iterations, 97.5, axis=1)))

print('1 sec vs. 2 sec')
print(np.sum(best_fitting_iterations[0, :] - best_fitting_iterations[1, :] > 0))
print('2 sec vs. UR sec')
print(np.sum(best_fitting_iterations[1, :] - best_fitting_iterations[2, :] > 0))



#### STATISTICAL COMPARISONS OF OCC vs. UNOCC PERFORMANCE FOR THE SAME CATEGORY TRIALS
### For PbAS and PbAS-Pixels

accuracy_diff_indices = np.zeros((2, R))
accuracy_same_indices = np.zeros((2, R))

for i in range(3):
    for c in range(2):
        accuracy = np.zeros(R)
        iteration = triplet[i]
        for j in range(R):
            pred = anp[c, j, :, iteration] 
            accuracy[j] = np.mean(pred)
            accuracy_diff_indices[c, j] = np.mean(pred[diff_indices])
            accuracy_same_indices[c, j] = np.mean(pred[same_indices])


    print('PbAS model -- Condition: ' + condition + ' ttest: ')
    print(accuracy_same_indices[0, :].shape)
    print(ttest_ind(accuracy_same_indices[0, :], accuracy_same_indices[1, :]))

    for c in range(2):
        accuracy = np.zeros(R)
        iteration = triplet_no_encoding[i]
        for j in range(R):
            pred = no_encoding_anp[c, j, :, iteration] 
            accuracy[j] = np.mean(pred)
            accuracy_diff_indices[c, j] = np.mean(pred[diff_indices])
            accuracy_same_indices[c, j] = np.mean(pred[same_indices])
    
    print('No encoding model -- Condition: ' + condition + ' ttest: ')
    print(accuracy_same_indices[0, :].shape)
    print(ttest_ind(accuracy_same_indices[0, :], accuracy_same_indices[1, :]))
