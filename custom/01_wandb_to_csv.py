import pandas as pd
import re
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("joelavond/FedDecay")

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })
print('final runs:', runs_df.shape)


## Recover all metrics of interest from wandb runs
# All metrics start with "Results_"
# Keep test and validation metric

# get columns from summary dict
metrics = runs_df['summary'].apply(pd.Series)

# keep validation metric
validation_metric = 'Results_avg/val_acc'

# all metrics start with Results_
# only keep test metrics other than validation columns
metrics = metrics[[name for name in metrics.columns if re.search('^Results', name)]]
metrics = metrics[[
    name for name in metrics.columns
    if re.search('/test', name)  # keep all test metrics
       or name == validation_metric
]]


# Further reduce to test metrics of interest
# Remove metrics that are unnecessary

# subset to metrics of interest
metrics_of_interest = ['acc', 'avg_loss', 'f1']

metrics = metrics[[
    name for name
    in metrics.columns
    if (
            any([bool(re.search(metric, name)) for metric in metrics_of_interest])
            and not(bool(re.search('top', name)))  # not interested in metrics containing "top"
    )
]]


# Remove redundant information from metric columns
# Meaningfully order metrics

# remove excess information from column names
# similarly change holder for validation metric
metrics.columns = [re.sub('_avg', '', re.sub('_fairness', '', name)) for name in metrics.columns]
validation_metric = re.sub('_avg', '', re.sub('_fairness', '', validation_metric))
print('new validation metric key:', validation_metric)

# order metrics
metrics = metrics[sorted(metrics.columns)]


# Use run name to extract relevant hyperparameters

# extract hyperparameter name and value
# force convert values to numeric
hyperparameters = runs_df.name.str.split('--').apply(
    lambda x: {
        re.sub('[0-9.]*', '', obj):re.sub('[a-z_]*', '', obj)
        for i, obj in enumerate(x)
        if re.search('[0-9]', obj)
           and i > 0
    }
).apply(pd.Series).apply(pd.to_numeric, errors='coerce')

# identify method used
methods = ['exact', 'fedavg', 'pfedme', 'fedbn', 'ditto', 'fedem']
method = runs_df.name.apply(
    lambda x: [method for method in methods if re.search(method, x)].pop(0)
)
dataset = runs_df.name.apply(lambda x: re.sub('-.*', '', x))
finetune = runs_df.name.apply(lambda x: bool(re.search('finetune', x))).astype(int)


# Combine all extracted information

# combine with metrics
df = pd.concat({'dataset':dataset, 'method':method, 'finetune':finetune}, axis=1)
df = df.join(hyperparameters)
df = df.join(metrics)


# All Fed runs can be thought of as special cases of decay
# Create copy of rows to treat as additional decay runs

# Copy fedsgd runs to exact decay
fedsgd_runs = df.loc[(df.method == 'fedavg')].copy()
fedsgd_runs.method = 'exact'

# Identify SGD runs as beta 0, otherwise beta 1
fedsgd_runs.beta = 1.0
fedsgd_runs.loc[(df.n_epochs == 1), 'beta'] = 0.0

# combine with previous data
df = pd.concat([df, fedsgd_runs])
df.to_csv('final_run_metrics.csv')

