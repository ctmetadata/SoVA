import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

dilemma_csv_path = "./input.csv"
virtues_csv_path = "./virtues.csv"
output_fig_dir = "./dile_result"
output_data_dir = "./dile_result"

MODEL_COLUMNS = ["model_resp_llama3_baseline", 
  "model_resp_Meta-Llama-3-70b-sft_mic_qa_1024_action", 
  "model_resp_graphrag_llama3_70b_virtue_1114",
  "model_resp_llama3_maslow_CoT_1002_2",
  "model_resp_llama3_maslow_PS_1002_2",
  "model_resp_llama3_maslow_MP_1002_2",
  "model_resp_rag_virtue_1119",
    "model_resp_SteerLM_sft_virtue_lr_1e6_1212_action",
    "model_resp_SteerLM_sft_virtue_lr_1e7_1212_action",
    "model_resp_sft_virtue_lr_1e6_1216_action",
    "model_resp_sft_virtue_lr_1e7_1216_action"
  ]
MODEL_DISPLAY_NAMES = {"model_resp_llama3_baseline": "baseline", 
  "model_resp_Meta-Llama-3-70b-sft_mic_qa_1024_action": "sft", 
  "model_resp_graphrag_llama3_70b_virtue_1114": "graphrag_llama3_70b_virtue",
  "model_resp_llama3_maslow_CoT_1002_2": "llama3_maslow_CoT_1002_2",
  "model_resp_llama3_maslow_PS_1002_2": "llama3_maslow_PS_1002_2",
  "model_resp_llama3_maslow_MP_1002_2": "llama3_maslow_MP_1002_2",
  "model_resp_rag_virtue_1119": "rag_llama3_virtue",
    "model_resp_SteerLM_sft_virtue_lr_1e6_1212_action": "SteerLM_1e6",
    "model_resp_SteerLM_sft_virtue_lr_1e7_1212_action": "SteerLM_1e7",
    "model_resp_sft_virtue_lr_1e6_1216_action": "sft_1e6",
    "model_resp_sft_virtue_lr_1e7_1216_action": "sft_1e7"
  }

TOP_N_VIRTUES = 9
FIGURE_SIZE = (20, 10)
DPI = 300
COLORS = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#7c3aed', '#db2777', '#0ea5e9']
MARKERS = ['o', 's', '^', 'D', 'p', '*', 'X']

def ensure_directories():
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

def parse_virtues(val_str, all_virtues):
    if pd.isna(val_str):
        return []
    try:
        cleaned_str = val_str.strip("[]").replace("'", "").replace('"', "").strip()
        if not cleaned_str:
            return []
        parsed_vals = [v.strip().lower() for v in cleaned_str.split(", ") if v.strip()]
        matched_vals = [val for val in parsed_vals if val in all_virtues]
        return matched_vals
    except Exception as e:
        print(f"Failed to parse virtues (first 50 chars: {str(val_str)[:50]}): {str(e)}")
        return []

def get_virtue_stats(all_appeared, total):
    if total == 0:
        return pd.DataFrame(columns=["virtue", "frequency", "proportion(%)"])
    counter = Counter(all_appeared)
    stats = pd.DataFrame({
        "virtue": list(counter.keys()),
        "frequency": list(counter.values())
    }).sort_values("frequency", ascending=False)
    stats["proportion(%)"] = round((stats["frequency"] / total) * 100, 2)
    return stats

def load_and_process_data():
    try:
        virtues_df = pd.read_csv(virtues_csv_path, encoding='utf-8', usecols=["virtue"])
        all_virtues = virtues_df["virtue"].str.strip().str.lower().dropna().unique().tolist()
        print(f"Loaded standard virtue list: {len(all_virtues)} total")
    except Exception as e:
        print(f"Failed to load virtues.csv: {str(e)}")
        exit()

    try:
        dilemma_df = pd.read_csv(dilemma_csv_path, encoding='utf-8')
        print(f"Loaded dilemma data: {len(dilemma_df)} records, {len(dilemma_df.columns)} fields")
        missing_cols = [col for col in MODEL_COLUMNS if col not in dilemma_df.columns]
        if missing_cols:
            print(f"Missing model columns: {', '.join(missing_cols)}")
            exit()
        print(f"All model column names matched successfully")
    except Exception as e:
        print(f"Failed to load dilemma data: {str(e)}")
        exit()

    model_results = {}
    for model_col in MODEL_COLUMNS:
        model_name = MODEL_DISPLAY_NAMES[model_col]
        print(f"\n=== Processing model: {model_name} ===")
        
        grouped = dilemma_df.groupby("idx")
        selected_virtues_list = []
        neglected_virtues_list = []
        no_choice_count = 0
        no_neglected_count = 0

        for idx, group in grouped:
            selected_rows = group[group[model_col].astype(str).str.strip() == "√"]
            if len(selected_rows) >= 1:
                selected_str = selected_rows.iloc[0]["virtue"]
                selected_virtues = parse_virtues(selected_str, all_virtues)
                selected_visited = []
                selected_virtues = [v for v in selected_virtues if v not in selected_visited and not selected_visited.append(v)]
                selected_virtues_list.append(selected_virtues)
            else:
                selected_virtues_list.append([])
                no_choice_count += 1

            unselected_rows = group[group[model_col].astype(str).str.strip() != "√"]
            if len(unselected_rows) >= 1:
                neglected_str = unselected_rows.iloc[0]["virtue"]
                neglected_virtues = parse_virtues(neglected_str, all_virtues)
                neglected_visited = []
                neglected_virtues = [v for v in neglected_virtues if v not in neglected_visited and not neglected_visited.append(v)]
                neglected_virtues_list.append(neglected_virtues)
            else:
                neglected_virtues_list.append([])
                no_neglected_count += 1

        print(f"Selected virtue records: {len(selected_virtues_list)} (no choice: {no_choice_count} dilemmas)")
        print(f"Neglected virtue records: {len(neglected_virtues_list)} (no neglected: {no_neglected_count} dilemmas)")

        all_selected = []
        for vs in selected_virtues_list:
            all_selected.extend(vs)
        selected_total = len(all_selected)
        selected_stats = get_virtue_stats(all_selected, selected_total)

        all_neglected = []
        for vs in neglected_virtues_list:
            all_neglected.extend(vs)
        neglected_total = len(all_neglected)
        neglected_stats = get_virtue_stats(all_neglected, neglected_total)

        selected_counter = Counter(all_selected)
        neglected_counter = Counter(all_neglected)
        all_virtues_in_stats = set(selected_stats["virtue"]).union(set(neglected_stats["virtue"]))
        total_stats_data = []
        for virtue in all_virtues_in_stats:
            selected_freq = selected_counter.get(virtue, 0)
            neglected_freq = neglected_counter.get(virtue, 0)
            diff_freq = selected_freq - neglected_freq
            selected_prop = round((selected_freq / selected_total) * 100, 2) if selected_total > 0 else 0.0
            neglected_prop = round((neglected_freq / neglected_total) * 100, 2) if neglected_total > 0 else 0.0

            total_freq = selected_freq + neglected_freq
            if total_freq == 0:
                freq_bias_percent = 0.0
            else:
                freq_bias_percent = round((diff_freq / total_freq) * 100, 2)

            total_stats_data.append({
                "virtue": virtue,
                "selected_frequency": selected_freq,
                "selected_proportion(%)": selected_prop,
                "neglected_frequency": neglected_freq,
                "neglected_proportion(%)": neglected_prop,
                "diff_frequency(selected-neglected)": diff_freq,
                "freq_bias_percent(%)": freq_bias_percent
            })
        total_stats = pd.DataFrame(total_stats_data).sort_values("diff_frequency(selected-neglected)", ascending=False)

        model_results[model_col] = {
            "selected": {"stats": selected_stats, "total": selected_total},
            "neglected": {"stats": neglected_stats, "total": neglected_total},
            "total": {"stats": total_stats},
            "model_name": model_name
        }

    return model_results, all_virtues

def plot_proportion_comparison(model_results):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    virtue_total_count = Counter()
    for model_col in model_results.keys():
        total_stats = model_results[model_col]["total"]["stats"]
        for _, row in total_stats.iterrows():
            virtue_total_count[row["virtue"]] += row["diff_frequency(selected-neglected)"]
    all_top_virtues = [val for val, _ in virtue_total_count.most_common(TOP_N_VIRTUES)]
    print(f"\nDisplaying top {len(all_top_virtues)} high-frequency virtues")

    for i, model_col in enumerate(model_results.keys()):
        model_name = MODEL_DISPLAY_NAMES[model_col]
        selected_stats = model_results[model_col]["selected"]["stats"]
        neglected_stats = model_results[model_col]["neglected"]["stats"]
        selected_dict = dict(zip(selected_stats["virtue"], selected_stats["proportion(%)"]))
        neglected_dict = dict(zip(neglected_stats["virtue"], neglected_stats["proportion(%)"]))

        selected_props = [selected_dict.get(v, 0.0) for v in all_top_virtues]
        neglected_props = [neglected_dict.get(v, 0.0) for v in all_top_virtues]

        ax.plot(all_top_virtues, selected_props, label=f"{model_name} (Selected)",
                color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], linewidth=2.2, markersize=8)
        ax.plot(all_top_virtues, neglected_props, label=f"{model_name} (Neglected)",
                color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], linewidth=2.2, markersize=8, linestyle="--")

        for j, (v, p) in enumerate(zip(all_top_virtues, selected_props)):
            if p >= 1.0:
                ax.text(j, p-0.2, f"{p:.1f}%", ha="center", va="top", fontsize=7.5, color=COLORS[i % len(COLORS)])

    ax.set_xlabel("Virtues", fontsize=13, fontweight='bold')
    ax.set_ylabel("Proportion (%)", fontsize=13, fontweight='bold')
    ax.set_title(f"Virtue Proportion Comparison (Selected vs Neglected)", fontsize=16, fontweight='bold', pad=25)
    ax.set_xticks(range(len(all_top_virtues)))
    ax.set_xticklabels(all_top_virtues, rotation=45, ha="right", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(output_fig_dir, "all_selected_vs_neglected_virtue_1216.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {output_path}")

def export_combined_comparison(model_results, writer):
    all_virtues = set()
    for model_col in MODEL_COLUMNS:
        virtues = model_results[model_col]["total"]["stats"]["virtue"].tolist()
        all_virtues.update(virtues)
    all_virtues = sorted(list(all_virtues))

    combined_df = pd.DataFrame({"virtue": all_virtues})

    for model_col in MODEL_COLUMNS:
        model_name = model_results[model_col]["model_name"]
        stats = model_results[model_col]["total"]["stats"].copy()
        
        stats = stats.rename(columns={
            "selected_frequency": f"{model_name}_selected_freq",
            "selected_proportion(%)": f"{model_name}_selected_prop(%)",
            "neglected_frequency": f"{model_name}_neglected_freq",
            "neglected_proportion(%)": f"{model_name}_neglected_prop(%)",
            "diff_frequency(selected-neglected)": f"{model_name}_diff_freq",
            "freq_bias_percent(%)": f"{model_name}_freq_bias(%)"
        })
        
        combined_df = combined_df.merge(stats[["virtue"] + list(stats.columns[1:])], on="virtue", how="left")

    combined_df = combined_df.fillna(0)
    combined_df.to_excel(writer, sheet_name="All Models Virtue Comparison", index=False)
    print("Virtue summary comparison table generated successfully")

def export_results(model_results):
    output_path = os.path.join(output_data_dir, "all_virtue_selected_neglected_analysis_1216.xlsx")
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for model_col in model_results.keys():
                model_name = MODEL_DISPLAY_NAMES[model_col]
                selected_stats = model_results[model_col]["selected"]["stats"]
                selected_stats.to_excel(writer, sheet_name=f"{model_name} - Selected", index=False)
                neglected_stats = model_results[model_col]["neglected"]["stats"]
                neglected_stats.to_excel(writer, sheet_name=f"{model_name} - Neglected", index=False)
                total_stats = model_results[model_col]["total"]["stats"]
                total_stats.to_excel(writer, sheet_name=f"{model_name} - Selected vs Neglected", index=False)
            
            export_combined_comparison(model_results, writer)
        
        print(f"Excel export successful (including summary sheet): {output_path}")
    except Exception as e:
        print(f"Failed to export Excel: {str(e)}")
        print("Install openpyxl: pip install openpyxl")

def prepare_boxplot_data(model_results, analysis_dimension="model_bias"):
    if analysis_dimension == "model_bias":
        data_list = []
        for model_col in model_results.keys():
            model_name = MODEL_DISPLAY_NAMES[model_col]
            total_stats = model_results[model_col]["total"]["stats"]
            for _, row in total_stats.iterrows():
                data_list.append({
                    "model": model_name,
                    "virtue": row["virtue"],
                    "freq_bias_percent": row["freq_bias_percent(%)"]
                })
        boxplot_df = pd.DataFrame(data_list)
        
    elif analysis_dimension == "virtue_bias":
        data_list = []
        virtue_total_count = Counter()
        for model_col in model_results.keys():
            total_stats = model_results[model_col]["total"]["stats"]
            for _, row in total_stats.iterrows():
                virtue_total_count[row["virtue"]] += row["diff_frequency(selected-neglected)"]
        top_virtues = [val for val, _ in virtue_total_count.most_common(TOP_N_VIRTUES)]
        
        for model_col in model_results.keys():
            model_name = MODEL_DISPLAY_NAMES[model_col]
            total_stats = model_results[model_col]["total"]["stats"]
            virtue_bias_dict = dict(zip(total_stats["virtue"], total_stats["freq_bias_percent(%)"]))
            for virtue in top_virtues:
                data_list.append({
                    "virtue": virtue,
                    "model": model_name,
                    "freq_bias_percent": virtue_bias_dict.get(virtue, 0.0)
                })
        boxplot_df = pd.DataFrame(data_list)
    
    boxplot_df = boxplot_df[(boxplot_df["freq_bias_percent"] >= -100) & (boxplot_df["freq_bias_percent"] <= 100)]
    return boxplot_df

def plot_virtue_boxplot(model_results):
    boxplot_df_model = prepare_boxplot_data(model_results, analysis_dimension="model_bias")
    fig, ax = plt.subplots(figsize=(25, 12))
    
    model_names = [MODEL_DISPLAY_NAMES[col] for col in MODEL_COLUMNS]
    box_data = [boxplot_df_model[boxplot_df_model["model"] == m]["freq_bias_percent"].values for m in model_names]
    
    bp = ax.boxplot(box_data, 
                    labels=model_names,
                    patch_artist=True,
                    showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "red", "markersize": 4},
                    medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"color": "#666666", "linewidth": 1.5},
                    capprops={"color": "#666666", "linewidth": 1.5},
                    flierprops={"marker": "x", "markerfacecolor": "#999999", "markersize": 3})
    
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    
    ax.set_xlabel("Models", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency Bias Percent (%)", fontsize=14, fontweight="bold")
    ax.set_title("Distribution of Virtue Frequency Bias Percent by Model", fontsize=18, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    output_path1 = os.path.join(output_fig_dir, "boxplot_model_virtue_bias.png")
    plt.tight_layout()
    plt.savefig(output_path1, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Model-virtue bias boxplot saved: {output_path1}")

    boxplot_df_virtue = prepare_boxplot_data(model_results, analysis_dimension="virtue_bias")
    fig, ax = plt.subplots(figsize=(18, 10))
    
    virtue_total_count = Counter()
    for model_col in model_results.keys():
        total_stats = model_results[model_col]["total"]["stats"]
        for _, row in total_stats.iterrows():
            virtue_total_count[row["virtue"]] += row["diff_frequency(selected-neglected)"]
    top_virtues = [val for val, _ in virtue_total_count.most_common(TOP_N_VIRTUES)]
    
    box_data = [boxplot_df_virtue[boxplot_df_virtue["virtue"] == v]["freq_bias_percent"].values for v in top_virtues]
    
    bp = ax.boxplot(box_data,
                    labels=top_virtues,
                    patch_artist=True,
                    showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "red", "markersize": 4},
                    medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"color": "#666666", "linewidth": 1.5},
                    capprops={"color": "#666666", "linewidth": 1.5},
                    flierprops={"marker": "x", "markerfacecolor": "#999999", "markersize": 3})
    
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    
    ax.set_xlabel("Top Virtues", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency Bias Percent (%)", fontsize=14, fontweight="bold")
    ax.set_title("Distribution of Frequency Bias Percent by Top Virtue (Across Models)", fontsize=18, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    output_path2 = os.path.join(output_fig_dir, "boxplot_virtue_model_bias.png")
    plt.tight_layout()
    plt.savefig(output_path2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Virtue-model bias boxplot saved: {output_path2}")

def get_all_virtues_from_csv():
    try:
        virtues_df = pd.read_csv(virtues_csv_path, encoding='utf-8', usecols=["virtue"])
        all_virtues = virtues_df["virtue"].str.strip().str.lower().dropna().unique().tolist()
        print(f"Read {len(all_virtues)} virtues from virtues.csv: {all_virtues}")
        return all_virtues
    except Exception as e:
        print(f"Failed to read virtues.csv: {str(e)}")
        return []

def plot_single_virtue_horizontal_boxplot(model_results, target_virtue="honesty"):
    virtue_bias_data = []
    model_names_list = []
    for model_col in MODEL_COLUMNS:
        model_name = MODEL_DISPLAY_NAMES[model_col]
        total_stats = model_results[model_col]["total"]["stats"]
        virtue_row = total_stats[total_stats["virtue"] == target_virtue]
        bias_value = virtue_row.iloc[0]["freq_bias_percent(%)"] if not virtue_row.empty else 0.0
        virtue_bias_data.append(bias_value)
        model_names_list.append(model_name)
    
    if all(bias == 0 for bias in virtue_bias_data):
        print(f"Virtue [{target_virtue}] has no valid data, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    box_data = [virtue_bias_data]
    bp = ax.boxplot(box_data,
                    vert=False,
                    labels=[f"Virtue: {target_virtue}"],
                    patch_artist=True,
                    showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "red", "markersize": 8, "label": "Mean"},
                    medianprops={"color": "black", "linewidth": 2, "label": "Median"},
                    whiskerprops={"color": "#666666", "linewidth": 1.5},
                    capprops={"color": "#666666", "linewidth": 1.5},
                    flierprops={"marker": "x", "markerfacecolor": "#999999", "markersize": 6})
    
    bp["boxes"][0].set_facecolor("#2563eb")
    bp["boxes"][0].set_alpha(0.7)
    
    y_pos = np.arange(1, len(model_names_list)+1)
    ax.scatter(virtue_bias_data, y_pos, 
               color="#dc2626", marker="s", s=60, alpha=0.8, label="Individual Model")
    
    for i, (model, val) in enumerate(zip(model_names_list, virtue_bias_data)):
        ax.text(val + 0.5, y_pos[i],
                f"{model} ({val:.1f}%)", 
                ha="left", va="center", fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax.set_xlabel(f"Frequency Bias Percent (%) for {target_virtue}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Models", fontsize=14, fontweight="bold")
    ax.set_title(f"Model Bias Distribution for Virtue: {target_virtue}", fontsize=18, fontweight="bold", pad=20)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names_list, fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=12)
    
    output_path = os.path.join(output_fig_dir, f"horizontal_boxplot_single_virtue_{target_virtue}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Single virtue [{target_virtue}] horizontal boxplot saved: {output_path}")

def plot_all_virtues_horizontal_boxplot(model_results, all_virtues):
    box_data = []
    valid_virtues = []
    for virtue in all_virtues:
        virtue_bias = []
        for model_col in MODEL_COLUMNS:
            total_stats = model_results[model_col]["total"]["stats"]
            virtue_row = total_stats[total_stats["virtue"] == virtue]
            bias_val = virtue_row.iloc[0]["freq_bias_percent(%)"] if not virtue_row.empty else 0.0
            virtue_bias.append(bias_val)
        if not all(b == 0 for b in virtue_bias):
            box_data.append(virtue_bias)
            valid_virtues.append(virtue)
    
    if not valid_virtues:
        print("No valid virtue data, skipping all virtues integrated boxplot")
        return
    
    fig_height = max(12, len(valid_virtues) * 0.8)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    
    bp = ax.boxplot(box_data,
                    vert=False,
                    labels=valid_virtues,
                    patch_artist=True,
                    showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "red", "markersize": 4, "label": "Mean"},
                    medianprops={"color": "black", "linewidth": 2, "label": "Median"},
                    whiskerprops={"color": "#666666", "linewidth": 1.2},
                    capprops={"color": "#666666", "linewidth": 1.2},
                    flierprops={"marker": "x", "markerfacecolor": "#999999", "markersize": 3})
    
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    
    ax.set_xlabel("Frequency Bias Percent (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Virtues", fontsize=14, fontweight="bold")
    ax.set_title("Model Bias Distribution Across All Virtues", fontsize=20, fontweight="bold", pad=25)
    
    ytick_fontsize = 10 if len(valid_virtues) <= 15 else 8
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.tick_params(axis="x", labelsize=12)
    
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=12)
    
    output_path = os.path.join(output_fig_dir, "horizontal_boxplot_all_virtues.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"All virtues integrated horizontal boxplot saved: {output_path}")

def main():
    ensure_directories()
    print("="*50)
    print("Starting model virtue analysis (selected + neglected)")
    print("="*50)
    
    model_results, _ = load_and_process_data()
    all_virtues = get_all_virtues_from_csv()
    
    if model_results:
        plot_proportion_comparison(model_results)
        plot_virtue_boxplot(model_results)
        
        if all_virtues:
            print("\n=== Starting single virtue horizontal boxplot ===")
            for virtue in all_virtues:
                plot_single_virtue_horizontal_boxplot(model_results, target_virtue=virtue)
        
        if all_virtues:
            print("\n=== Starting all virtues integrated horizontal boxplot ===")
            plot_all_virtues_horizontal_boxplot(model_results, all_virtues)
        
        export_results(model_results)
    else:
        print("No valid model data")
    
    print("\n" + "="*50)
    print("Analysis completed!")
    print("="*50)

if __name__ == "__main__":
    main()