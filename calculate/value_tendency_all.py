import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

dilemma_csv_path = ""
values_csv_path = ""
output_fig_dir = ""
output_data_dir = ""

MODEL_COLUMNS = ["model_resp_llama3_baseline", 
  "model_resp_Meta-Llama-3-70b-sft_mic_qa_1024_action", 
  "model_resp_graphrag_llama3_70b_virtue_1114",
  "model_resp_llama3_maslow_CoT_1002_2",
  "model_resp_llama3_maslow_PS_1002_2",
  "model_resp_llama3_maslow_MP_1002_2",
  "model_resp_rag_virtue_1119",
  "model_resp_SteerLM_sft_virtue_lr_1e7_1212_action",
  "model_resp_sft_virtue_lr_1e6_1216_action"]

MODEL_DISPLAY_NAMES = {"model_resp_llama3_baseline": "baseline", 
  "model_resp_Meta-Llama-3-70b-sft_mic_qa_1024_action": "sft", 
  "model_resp_graphrag_llama3_70b_virtue_1114": "graphrag_llama3_70b_virtue",
  "model_resp_llama3_maslow_CoT_1002_2": "llama3_maslow_CoT_1002_2",
  "model_resp_llama3_maslow_PS_1002_2": "llama3_maslow_PS_1002_2",
  "model_resp_llama3_maslow_MP_1002_2": "llama3_maslow_MP_1002_2",
  "model_resp_rag_virtue_1119": "rag_llama3_virtue",
  "model_resp_SteerLM_sft_virtue_lr_1e7_1212_action": "SteerLM",
  "model_resp_sft_virtue_lr_1e6_1216_action": "sft"}

TOP_N_VALUES = 12
FIGURE_SIZE = (20, 10)
DPI = 300
COLORS = {
    "baseline_selected": "#2563eb",
    "baseline_neglected": "#94a3b8",
    "sft_selected": "#dc2626",
    "sft_neglected": "#fecaca"
}
MARKERS = {
    "selected": "o",
    "neglected": "x"
}


def ensure_directories():
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

def parse_values(val_str, all_values):
    if pd.isna(val_str):
        return []
    try:
        cleaned_str = val_str.strip("[]").replace("'", "").replace('"', "").strip()
        if not cleaned_str:
            return []
        parsed_vals = [v.strip().lower() for v in cleaned_str.split(", ") if v.strip()]
        matched_vals = [val for val in parsed_vals if val in all_values]
        return matched_vals
    except Exception as e:
        print(f"⚠️ Failed to parse values (first 50 chars: {str(val_str)[:50]}): {str(e)}")
        return []

def get_value_stats(all_appeared, total):
    if total == 0:
        return pd.DataFrame(columns=["value", "frequency", "proportion(%)"])
    counter = Counter(all_appeared)
    stats = pd.DataFrame({
        "value": list(counter.keys()),
        "frequency": list(counter.values())
    }).sort_values("frequency", ascending=False)
    stats["proportion(%)"] = round((stats["frequency"] / total) * 100, 2)
    return stats

def load_and_process_data():
    try:
        values_df = pd.read_csv(values_csv_path, encoding='utf-8', usecols=["value"])
        all_values = values_df["value"].str.strip().str.lower().dropna().unique().tolist()
        print(f"✅ Loaded standard value list: {len(all_values)} types")
    except Exception as e:
        print(f"❌ Failed to load values.csv: {str(e)}")
        exit()

    try:
        dilemma_df = pd.read_csv(dilemma_csv_path, encoding='utf-8')
        print(f"✅ Loaded dilemma data: {len(dilemma_df)} records, {len(dilemma_df.columns)} fields")
        missing_cols = [col for col in MODEL_COLUMNS if col not in dilemma_df.columns]
        if missing_cols:
            print(f"❌ Missing model columns: {', '.join(missing_cols)}")
            exit()
        print(f"✅ All model column names matched successfully")
    except Exception as e:
        print(f"❌ Failed to load dilemma data: {str(e)}")
        exit()

    model_results = {}
    for model_col in MODEL_COLUMNS:
        model_name = MODEL_DISPLAY_NAMES[model_col]
        print(f"\n=== Processing model: {model_name} ===")
        
        grouped = dilemma_df.groupby("idx")
        selected_values_list = []
        neglected_values_list = []
        no_choice_count = 0
        no_neglected_count = 0

        for idx, group in grouped:
            selected_rows = group[group[model_col].astype(str).str.strip() == "√"]
            if len(selected_rows) >= 1:
                selected_str = selected_rows.iloc[0]["values_names"]
                selected_values = parse_values(selected_str, all_values)
                selected_visited = []
                selected_values = [v for v in selected_values if v not in selected_visited and not selected_visited.append(v)]
                selected_values_list.append(selected_values)
            else:
                selected_values_list.append([])
                no_choice_count += 1

            unselected_rows = group[group[model_col].astype(str).str.strip() != "√"]
            if len(unselected_rows) >= 1:
                neglected_str = unselected_rows.iloc[0]["values_names"]
                neglected_values = parse_values(neglected_str, all_values)
                neglected_visited = []
                neglected_values = [v for v in neglected_values if v not in neglected_visited and not neglected_visited.append(v)]
                neglected_values_list.append(neglected_values)
            else:
                neglected_values_list.append([])
                no_neglected_count += 1

        print(f"✅ Selected value records: {len(selected_values_list)} (no choice: {no_choice_count} dilemmas)")
        print(f"✅ Neglected value records: {len(neglected_values_list)} (no neglected: {no_neglected_count} dilemmas)")

        all_selected = []
        for vs in selected_values_list:
            all_selected.extend(vs)
        selected_total = len(all_selected)
        selected_stats = get_value_stats(all_selected, selected_total)

        all_neglected = []
        for vs in neglected_values_list:
            all_neglected.extend(vs)
        neglected_total = len(all_neglected)
        neglected_stats = get_value_stats(all_neglected, neglected_total)

        selected_counter = Counter(all_selected)
        neglected_counter = Counter(all_neglected)
        all_values_in_stats = set(selected_stats["value"]).union(set(neglected_stats["value"]))
        total_stats_data = []
        for value in all_values_in_stats:
            selected_freq = selected_counter.get(value, 0)
            neglected_freq = neglected_counter.get(value, 0)
            total_freq = selected_freq - neglected_freq
            selected_prop = round((selected_freq / selected_total) * 100, 2) if selected_total > 0 else 0.0
            neglected_prop = round((neglected_freq / neglected_total) * 100, 2) if neglected_total > 0 else 0.0
            total_stats_data.append({
                "value": value,
                "selected_frequency": selected_freq,
                "selected_proportion(%)": selected_prop,
                "neglected_frequency": neglected_freq,
                "neglected_proportion(%)": neglected_prop,
                "diff_frequency(selected-neglected)": total_freq,
                "diff_proportion(%)": round(selected_prop - neglected_prop, 2)
            })
        total_stats = pd.DataFrame(total_stats_data).sort_values("diff_frequency(selected-neglected)", ascending=False)

        model_results[model_col] = {
            "selected": {"stats": selected_stats, "total": selected_total},
            "neglected": {"stats": neglected_stats, "total": neglected_total},
            "total": {"stats": total_stats},
            "model_name": model_name
        }

    return model_results, all_values

def plot_proportion_comparison(model_results):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    value_total_count = Counter()
    for model_col in MODEL_COLUMNS:
        total_stats = model_results[model_col]["total"]["stats"]
        for _, row in total_stats.iterrows():
            value_total_count[row["value"]] += row["diff_frequency(selected-neglected)"]
    top_values = [val for val, _ in value_total_count.most_common(TOP_N_VALUES)]
    print(f"\n✅ Displaying top {len(top_values)} high-frequency values: {top_values}")

    all_props = []
    for model_col in MODEL_COLUMNS:
        model_name = MODEL_DISPLAY_NAMES[model_col]
        selected_stats = model_results[model_col]["selected"]["stats"]
        neglected_stats = model_results[model_col]["neglected"]["stats"]
        
        selected_dict = dict(zip(selected_stats["value"].tolist(), selected_stats["proportion(%)"].tolist()))
        neglected_dict = dict(zip(neglected_stats["value"].tolist(), neglected_stats["proportion(%)"].tolist()))

        selected_props = [float(selected_dict.get(v, 0.0)) for v in top_values]
        neglected_props = [float(neglected_dict.get(v, 0.0)) for v in top_values]

        all_props.extend(selected_props + neglected_props)

        if model_name == "baseline":
            selected_color = COLORS["baseline_selected"]
            neglected_color = COLORS["baseline_neglected"]
        else:
            selected_color = COLORS["sft_selected"]
            neglected_color = COLORS["sft_neglected"]

        ax.plot(
            top_values, selected_props, 
            label=f"{model_name} - Selected",
            color=selected_color, marker=MARKERS["selected"], 
            linewidth=3, markersize=10, alpha=0.8
        )
        ax.plot(
            top_values, neglected_props, 
            label=f"{model_name} - Neglected",
            color=neglected_color, marker=MARKERS["neglected"], 
            linewidth=3, markersize=10, linestyle="--", alpha=0.8
        )

        for j, (v, p) in enumerate(zip(top_values, selected_props)):
            if p >= 1.0:
                ax.text(
                    j, p + 0.3, f"{p:.1f}%", 
                    ha="center", va="bottom", fontsize=9, fontweight="bold", 
                    color=selected_color
                )
        for j, (v, p) in enumerate(zip(top_values, neglected_props)):
            if p >= 1.0:
                ax.text(
                    j, p - 0.5, f"{p:.1f}%", 
                    ha="center", va="top", fontsize=9, fontweight="bold", 
                    color=neglected_color
                )

    if all_props:
        y_max = max(all_props) * 1.1
        y_min = 0
        ax.set_ylim(y_min, y_max if y_max > 5 else 5)
    else:
        ax.set_ylim(0, 5)

    ax.set_xlabel("Values", fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel("Proportion (%)", fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title(
        f"Value Proportion Comparison (Top {len(top_values)} Values)", 
        fontsize=18, fontweight='bold', pad=30
    )
    ax.set_xticks(range(len(top_values)))
    ax.set_xticklabels(top_values, rotation=45, ha="right", fontsize=11, fontweight="medium")
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=1)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()

    output_path = os.path.join(output_fig_dir, "all_selected_neglected_comparison_1223.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ Two-model comparison chart saved: {output_path}")

def export_combined_comparison(model_results, writer):
    all_values = set()
    for model_col in MODEL_COLUMNS:
        values = model_results[model_col]["total"]["stats"]["value"].tolist()
        all_values.update(values)
    all_values = sorted(list(all_values))

    combined_df = pd.DataFrame({"value": all_values})

    for model_col in MODEL_COLUMNS:
        model_name = model_results[model_col]["model_name"]
        stats = model_results[model_col]["total"]["stats"].copy()
        
        stats = stats.rename(columns={
            "selected_frequency": f"{model_name}_selected_freq",
            "selected_proportion(%)": f"{model_name}_selected_prop(%)",
            "neglected_frequency": f"{model_name}_neglected_freq",
            "neglected_proportion(%)": f"{model_name}_neglected_prop(%)",
            "diff_frequency(selected-neglected)": f"{model_name}_diff_freq",
            "diff_proportion(%)": f"{model_name}_diff_prop(%)"
        })
        
        combined_df = combined_df.merge(stats, on="value", how="left")

    combined_df = combined_df.fillna(0)
    combined_df.to_excel(writer, sheet_name="All Models Comparison", index=False)
    print("✅ Summary comparison table generated successfully")

def export_results(model_results):
    output_path = os.path.join(output_data_dir, "all_value_selected_neglected_analysis_1223.xlsx")
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for model_col in MODEL_COLUMNS:
                model_name = MODEL_DISPLAY_NAMES[model_col]
                selected_stats = model_results[model_col]["selected"]["stats"]
                selected_stats.to_excel(writer, sheet_name=f"{model_name} - Selected", index=False)
                neglected_stats = model_results[model_col]["neglected"]["stats"]
                neglected_stats.to_excel(writer, sheet_name=f"{model_name} - Neglected", index=False)
                total_stats = model_results[model_col]["total"]["stats"]
                total_stats.to_excel(writer, sheet_name=f"{model_name} - Selected vs Neglected", index=False)
            
            export_combined_comparison(model_results, writer)
        
        print(f"✅ Excel export successful (including summary table): {output_path}")
    except Exception as e:
        print(f"❌ Failed to export Excel: {str(e)}")
        print("💡 Install openpyxl: pip install openpyxl")

def main():
    ensure_directories()
    print("="*50)
    print("Starting model value analysis")
    print("="*50)
    
    model_results, all_values = load_and_process_data()
    
    if len(model_results) == len(MODEL_COLUMNS):
        plot_proportion_comparison(model_results)
        export_results(model_results)
    else:
        print(f"❌ Model count mismatch! Expected {len(MODEL_COLUMNS)}, got {len(model_results)}")
    
    print("\n" + "="*50)
    print("Model analysis completed!")
    print("="*50)

if __name__ == "__main__":
    main()