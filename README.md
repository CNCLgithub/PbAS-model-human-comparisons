# PbAS-human-comparisons
Behavioral data and analysis scripts underlying the model-human comparisons reported in "Perception of 3D shape integrates intuitive physics and analysis by synthesis"

## Setup

```
python -m venv venv
source venv/bin/activate
```

Install requirements
```
python3 -m pip install -r requirements.txt
```


Download the `.pkl` files with the pretrained and fine-tuned network results.

```
# pretrained
gdown 1wqYoMp8ExtGFU1HgJjvxwtYmM0BgvTs7
# finetuned
gdown 1cKxRi21HdcdJrDi-l33sHJeVOImk7VOU
```

## Pipeline

### Step 1: Process behavioral data, include quantifying effects of experimental design parameters (occlusion setting and presentation time condition) on behavior 

We separate scripts based on whether we are processing 2AFC accuracy or response time. These scripts also generate the `.pkl` files that are than used in the model-behavior comparisons scripts. 

First accuracy: 

```
# average accuracy levels (average at condition level)
python3 process_data_average_performance.py

# trial level accuracy
python3 process_data_by_stimuli.py
```

Second response times:

```
# average rt across conditions
python3 process_data_average_rt.py

# trial level response times
python3 process_data_by_stimuli_for_rt.py
```

(NOTE: the scripts `process_data_by_stimuli.py` and  `process_data_by_stimuli_for_rt.py` will take a while to complete as they generate thousands of bootstrap samples for later analysis.)

### Step 2: Model-behavior comparisons

We again separate the scripts for accuracy and response data

First accuracy:

```
# average accuracy levels
python3 compare_model_data_using_average_performance.py

# trial-level accuracy correlations
python3 compare_model_data_using_correlation.py
```

Second response times

```
# trial-level rt comparisons
python3 compare_rt_trial_level.py
```

Figures are generated in the `./output` folder, alongside various `.pkl` files that are stored at the top level directory. 
