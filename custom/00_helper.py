""" Packages """
import os
import re
import pandas as pd


""" Helper functions """
def load_data(names):

    # read each data set
    df = []
    print('input datasets:')
    for project_name in names:
        temp = pd.read_csv(f'decay--{project_name}.csv')
        print('\t', temp.shape)
        df.append(temp)

    df = pd.concat(df, axis=0, ignore_index=True)
    df = df.drop([name for name in df.columns if re.search('unname', name.lower())], axis=1)
    print('all runs:', df.shape)

    # remove any values from list if still inside one
    # convert to numeric
    metrics_to_reformat = [
        name for name, type in zip(df.columns, df.dtypes)
        if re.search('Results', name)
           and type == 'object'
    ]
    df[metrics_to_reformat] = df[metrics_to_reformat].applymap(lambda x: re.sub('[\[\]]', '', x) if isinstance(x, str) else x).astype('float')

    return df


# Return run summaries based on metric performance
def process_run_metrics(filtered_df, method='exact'):

    ## Process metrics and get top runs for each
    # are large or small metric values are desirable?
    descending_metrics = [name for name in filtered_df.columns if re.search('Results.*test', name)]
    ascending_metrics = [
        descending_metrics.pop(descending_metrics.index(name))
        for name in descending_metrics
        if re.search('std', name)
    ]
    metrics = descending_metrics + ascending_metrics

    # all non-metrics columns are used to identify the experimental run
    filtered_runs = filtered_df[[
        name for name in filtered_df.columns
        if name not in metrics
    ]]

    # rank the metrics
    ranked_descending = filtered_df.groupby(['dataset'])[descending_metrics].rank(
        method='min',
        ascending=False
    )
    ranked_ascending = filtered_df.groupby(['dataset'])[ascending_metrics].rank(
        method='min',
        ascending=True
    )

    # combine and sort the ranked_metrics
    ranked_metrics = pd.concat([ranked_descending, ranked_ascending], axis=1)
    ranked_metrics = ranked_metrics[sorted(ranked_metrics.columns)]
    filtered_ranks = filtered_runs.join(ranked_metrics)


    ## Return table of top ranks for each dataset
    filtered_top_ranks = filtered_ranks.copy()
    filtered_top_ranks[metrics] = filtered_top_ranks[metrics].applymap(lambda x: x if x <= 3 else pd.NA)


    ## Manipulate rank data to be summarized by runs and metrics
    # convert to long format
    # filter to top ranks
    long_filtered_ranks = pd.melt(filtered_ranks, id_vars=filtered_runs.columns, var_name='metric')
    top_filtered_metrics = long_filtered_ranks.loc[long_filtered_ranks.value <= 3].copy()  # top 3 runs

    # compute rank summaries to understand what runs are top overall
    top_filtered_metrics['rank_one_ind'] = (top_filtered_metrics.value == 1)
    top_filtered_metrics['rank_two_ind'] = (top_filtered_metrics.value == 2)
    top_filtered_metrics['rank_three_ind'] = (top_filtered_metrics.value == 3)
    top_filtered_metrics.replace(False, pd.NA, inplace=True)

    # summarized metric ranks for run type
    rank_summary_columns = ['rank_one_ind', 'rank_two_ind', 'rank_three_ind', 'value']
    id_columns = ['method', 'finetune']
    run_summary = top_filtered_metrics.groupby(id_columns)[rank_summary_columns].count()

    # summarize metric ranks for metric choice
    metric_summary = top_filtered_metrics.loc[top_filtered_metrics.method == method]
    metric_summary = metric_summary.sort_values(by='metric').groupby('metric')[rank_summary_columns].count()


    return (
        run_summary,
        metric_summary,
        filtered_top_ranks
    )


# convert runs to latex table
def runs_to_latex(
    filtered_df,
    file_suffix,
    id_columns=['method', 'finetune'],
    remove_columns=['alpha'],
    out_path='output'
):

    # store output
    tuning_objects = []
    metric_objects = []

    filtered_df = filtered_df[[
        name for name in filtered_df.columns if not name in remove_columns
    ]]

    metrics = [name for name in filtered_df.columns if re.search('Results.*test', name)]
    non_metrics = [
        name for name in filtered_df.columns
        if name not in metrics
           and name != 'dataset'
    ]
    non_metrics = [
        'method',
        'n_epochs', 'batch_size', 'lr',
        'regular_weight', 'K', 'beta'
    ]

    # iterate each dataset analyzed
    for dataset in filtered_df.dataset.unique():


        ## Return table of best hyper-parameters for each run
        temp_df = filtered_df.loc[filtered_df.dataset == dataset]
        temp_tuning = temp_df[non_metrics]
        #temp_tuning = temp_tuning.sort_values(by=id_columns)
        temp_tuning = temp_tuning.sort_values(by='method')
        temp_tuning.columns = [
            re.sub('/', '_', re.sub('Results[_/]*', '', name)) for name in temp_tuning.columns
        ]

        with open(os.path.join(out_path, f'{dataset}--hyperparameters--{file_suffix}.txt'), 'w') as h:
            string_temp_tuning = temp_tuning.to_string(
                header=True,
                index=False,
                index_names=False
            )

            # remove leading space after newlines
            string_temp_tuning = re.sub('\n[\s]+', '\n', string_temp_tuning)
            # replace white space between words with table column skip
            string_temp_tuning = re.sub('[ \t]+', ' & ', string_temp_tuning)

            # add latex newline to end of each line
            string_temp_tuning = string_temp_tuning.replace('_', ' ')
            string_temp_tuning = string_temp_tuning.replace('\n', ' \\\\\n')

            # write to file
            h.writelines(string_temp_tuning + ' \\\\')

        ## Return table of best metrics for each run
        temp_metric = temp_df[id_columns + metrics].round(4)
        temp_metric = temp_metric.sort_values(by=id_columns)
        temp_metric.columns = [
            re.sub('/', '_', re.sub('Results[_/]*', '', name)) for name in temp_metric.columns
        ]

        with open(os.path.join(out_path, f'{dataset}--metrics--{file_suffix}.txt'), 'w') as m:
            string_temp_metric = temp_metric.to_string(
                header=True,
                index=False,
                index_names=False
            )

            # remove leading space after newlines
            # replace white space between words with table column skip
            string_temp_metric = re.sub('\n[\s]+', '\n', string_temp_metric)
            string_temp_metric = re.sub('[ \t]+', ' & ', string_temp_metric)

            # add latex newline to end of each line
            string_temp_metric = string_temp_metric.replace('_', ' ')
            string_temp_metric = string_temp_metric.replace('\n', ' \\\\\n')

            # write to file
            m.writelines(string_temp_metric + ' \\\\')

    return None


