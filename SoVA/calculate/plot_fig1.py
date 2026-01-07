import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
import json

input_file_path = r'.\virtues_NEW1.csv'
output_dir = r'.\dile_result'

config_path = '.\config.json'
with open(config_path, 'r') as config_file:
    format_for_metric = json.load(config_file)

metrics = ['virtue']

def plot_barh_two_subplots(dicts, labels, bar_width, output_file_path, 
                           positive_keys, positive_new_keys, 
                           negative_keys, negative_new_keys):
    num_models = len(labels)
    if num_models == 0:
        raise ValueError("Error: Model list is empty. Please check CSV file reading!")

    colors = ['#E56F5E', '#F19685', '#FFB77F', '#F6C957', '#DCE9F4', '#ABD0F1', '#9EC4BE', '#43978F']
    hatches = ['//////', '\\\\\\', '**', 'xx', '|', '', '+', 'o']

    if num_models == 1:
        bar_width = 0.8
    else:
        bar_width = min(0.8 / num_models, 0.4)

    group_spacing = 0.05
    num_dicts = len(dicts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=100, sharex=False)
    sns.set(style="whitegrid")

    pos_base_positions = np.arange(len(positive_keys)) * (1 + group_spacing)
    pos_r = [pos_base_positions + i * bar_width for i in range(num_dicts)]

    ax1.grid(axis='x', linestyle='-.', color="#514F4F", linewidth=0.8, alpha=0.6)

    for i, d in enumerate(dicts):
        values = [d.get(key, 0) for key in positive_keys]
        ax1.barh(
            pos_r[i], values,
            height=bar_width,
            edgecolor='#585252',
            color=colors[i],
            label=labels[i],
            hatch=hatches[i % len(hatches)],
            linewidth=1.0
        )

    ax1.set_yticks(pos_r[0] + bar_width * (num_dicts - 1) / 2, positive_new_keys, fontsize=28)

    x1_min = -0.5
    x1_max = 1.5
    new_ticks1 = np.arange(x1_min, x1_max + 0.5, 0.5)
    ax1.set_xlim(x1_min, x1_max)
    ax1.set_xticks(new_ticks1, [f'{tick:.1f}%' for tick in new_ticks1], fontsize=28)

    neg_base_positions = np.arange(len(negative_keys)) * (1 + group_spacing)
    neg_r = [neg_base_positions + i * bar_width for i in range(num_dicts)]

    ax2.grid(axis='x', linestyle='-.', color="#514F4F", linewidth=0.8, alpha=0.6)

    for i, d in enumerate(dicts):
        values = [d.get(key, 0) for key in negative_keys]
        ax2.barh(
            neg_r[i], values,
            height=bar_width,
            edgecolor='#585252',
            color=colors[i],
            label=labels[i],
            hatch=hatches[i % len(hatches)],
            linewidth=1.0
        )

    ax2.set_yticks(neg_r[0] + bar_width * (num_dicts - 1) / 2, negative_new_keys, fontsize=28)

    handles, labels_legend = ax2.get_legend_handles_labels()
    leg2 = ax2.legend(
        handles[::-1],
        labels_legend[::-1],
        fontsize=24,
        ncol=1,
        loc='lower right',
        bbox_to_anchor=(1.0, 0.02)
    )
    frame1 = leg2.get_frame()
    frame1.set_linewidth(1)
    frame1.set_edgecolor("black")

    x2_min = -1.0
    x2_max = 1.0
    new_ticks2 = np.arange(x2_min, x2_max + 0.5, 0.5)
    ax2.set_xlim(x2_min, x2_max)
    ax2.set_xticks(new_ticks2, [f'{tick:.1f}%' for tick in new_ticks2], fontsize=28)

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor("black")

    plt.subplots_adjust(wspace=0.2, right=0.9)
    plt.tight_layout()
    plt.savefig(output_file_path, format='pdf', bbox_inches='tight')
    plt.close()

analysis_df = pd.read_csv(input_file_path, sep=None, engine='python')
print("CSV columns read:", analysis_df.columns.tolist())

first_column_name = analysis_df.columns[0]
models = analysis_df.columns[1:].tolist()
print("Extracted model list:", models)
if not models:
    raise ValueError("Error: No model columns extracted. Please check CSV file format!")

total_counter_list = []
for model in models:
    model_dict = dict(zip(analysis_df[first_column_name], analysis_df[model]))
    total_counter_list.append(model_dict)

for metric in tqdm(metrics):
    metric_config = format_for_metric[metric]
    output_file = f'{output_dir}/percentage_diff_prop_{metric}_1226.pdf'

    positive_keys = ["ethical", "empathy", "gratitude", "understanding"]
    negative_keys = ["dishonesty", "deception", "betrayal", "injustice"]

    positive_new_keys = [metric_config['new_labels'][metric_config['labels'].index(k)] for k in positive_keys]
    negative_new_keys = [metric_config['new_labels'][metric_config['labels'].index(k)] for k in negative_keys]

    plot_barh_two_subplots(
        total_counter_list, models, 0.07, output_file,
        positive_keys, positive_new_keys,
        negative_keys, negative_new_keys
    )