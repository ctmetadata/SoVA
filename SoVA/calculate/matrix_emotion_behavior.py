import pandas as pd
import os
import ast
from typing import List, Dict

def get_emotion_groups() -> List[str]:
    return [
        'Fear, Terror',
        'Anger, Rage',
        'Joy, Ecstasy',
        'Sadness, Grief',
        'Acceptance, Trust',
        'Disgust, Loathing',
        'Expectancy, Anticipation',
        'Surprise, Astonishment'
    ]

def get_merged_behaviors() -> List[str]:
    return [
        'Withdrawing/Escaping',
        'Attacking/Biting',
        'Mating/Possessing',
        'Crying for Help',
        'Pair Bonding/Grooming',
        'Vomiting/Defecating',
        'Examining/Mapping',
        'Stopping/Freezing'
    ]

def get_emotion_to_group_mapping() -> Dict[str, str]:
    return {
        'fear': 'Fear, Terror',
        'terror': 'Fear, Terror',
        'anger': 'Anger, Rage',
        'rage': 'Anger, Rage',
        'joy': 'Joy, Ecstasy',
        'ecstasy': 'Joy, Ecstasy',
        'sadness': 'Sadness, Grief',
        'grief': 'Sadness, Grief',
        'acceptance': 'Acceptance, Trust',
        'trust': 'Acceptance, Trust',
        'disgust': 'Disgust, Loathing',
        'loathing': 'Disgust, Loathing',
        'expectancy': 'Expectancy, Anticipation',
        'anticipation': 'Expectancy, Anticipation',
        'surprise': 'Surprise, Astonishment',
        'astonishment': 'Surprise, Astonishment'
    }

def get_behavior_to_merged_mapping() -> Dict[str, str]:
    return {
        'withdrawing': 'Withdrawing/Escaping',
        'escaping': 'Withdrawing/Escaping',
        'attacking': 'Attacking/Biting',
        'biting': 'Attacking/Biting',
        'mating': 'Mating/Possessing',
        'possessing': 'Mating/Possessing',
        'crying for help': 'Crying for Help',
        'pair bonding': 'Pair Bonding/Grooming',
        'cryingforhelp': 'Crying for Help',
        'pairbonding': 'Pair Bonding/Grooming',
        'grooming': 'Pair Bonding/Grooming',
        'vomiting': 'Vomiting/Defecating',
        'defecating': 'Vomiting/Defecating',
        'examining': 'Examining/Mapping',
        'mapping': 'Examining/Mapping',
        'stopping': 'Stopping/Freezing',
        'freezing': 'Stopping/Freezing'
    }

def parse_list_string(list_str: str) -> List[str]:
    try:
        cleaned_str = list_str.strip().replace("'", '"').replace(' ', '')
        parsed_list = ast.literal_eval(cleaned_str)
        
        if isinstance(parsed_list, list):
            return list(set([str(item).strip().lower() for item in parsed_list if str(item).strip() != '']))
        else:
            return []
    except (ValueError, SyntaxError, TypeError):
        return []

def extract_and_transform_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"Original CSV row count: {len(df)}")
    
    required_cols = ['emotion', 'behavior']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain columns '{required_cols}', please check format!")
    
    df['emotion_list'] = df['emotion'].apply(parse_list_string)
    df['behavior_list'] = df['behavior'].apply(parse_list_string)
    
    df_valid = df[
        (df['emotion_list'].str.len() > 0) & 
        (df['behavior_list'].str.len() > 0)
    ].copy()
    print(f"Valid rows after filtering empty lists: {len(df_valid)}")
    
    emotion_mapping = get_emotion_to_group_mapping()
    behavior_mapping = get_behavior_to_merged_mapping()
    
    transformed_pairs = []
    unmatched_pairs = []
    
    for _, row in df_valid.iterrows():
        emotions = row['emotion_list']
        behaviors = row['behavior_list']
        
        for emotion in emotions:
            emotion_group = emotion_mapping.get(emotion, "Other")
            
            for behavior in behaviors:
                merged_behavior = behavior_mapping.get(behavior, "Other")
                
                transformed_pairs.append({
                    'original_emotion': emotion,
                    'original_behavior': behavior,
                    'emotion_group': emotion_group,
                    'merged_behavior': merged_behavior
                })
                
                if emotion_group == "Other" or merged_behavior == "Other":
                    unmatched_pairs.append({
                        'emotion': emotion,
                        'behavior': behavior,
                        'unmatched_reason': 
                            "Emotion not matched" if emotion_group == "Other" else 
                            "Behavior not matched" if merged_behavior == "Other" else "Both not matched"
                    })
    
    df_transformed = pd.DataFrame(transformed_pairs)
    total_pairs = len(df_transformed)
    print(f"Total valid 'emotion-behavior' pairs after transformation: {total_pairs}")
    
    if unmatched_pairs:
        print("\n" + "="*60)
        print("Unmatched emotion-behavior pairs:")
        print("="*60)
        unmatched_df = pd.DataFrame(unmatched_pairs)
        unmatched_counts = unmatched_df.groupby(['emotion', 'behavior', 'unmatched_reason']).size().reset_index(name='count')
        print(unmatched_counts.to_string(index=False))
        
        unmatched_total = len(unmatched_pairs)
        print(f"\nTotal unmatched emotion-behavior pairs: {unmatched_total} (proportion: {round(unmatched_total/total_pairs*100, 2)}%)")
    else:
        print("\nCongratulations! All emotion-behavior pairs have been successfully matched.")
    
    if total_pairs == 0:
        raise ValueError("No valid data after parsing, please check CSV format!")
    
    return df_transformed, total_pairs

def calculate_emotion_behavior_matrix(df_transformed: pd.DataFrame, total_pairs: int) -> pd.DataFrame:
    emotion_groups = get_emotion_groups()
    merged_behaviors = get_merged_behaviors()
    
    matrix = pd.DataFrame(
        0.0, 
        index=emotion_groups, 
        columns=merged_behaviors
    )
    matrix.index.name = 'Emotion Groups'
    matrix.columns.name = 'Merged Behaviors'
    
    matched_data = df_transformed[
        (df_transformed['emotion_group'] != "Other") & 
        (df_transformed['merged_behavior'] != "Other")
    ]
    matched_count = len(matched_data)
    print(f"\nTotal matched emotion-behavior pairs: {matched_count} (proportion: {round(matched_count/total_pairs*100, 2)}%)")
    
    group_counts = matched_data.groupby(['emotion_group', 'merged_behavior']).size().reset_index(name='count')
    
    for _, row in group_counts.iterrows():
        emotion_group = row['emotion_group']
        merged_behavior = row['merged_behavior']
        count = row['count']
        
        if matched_count > 0:
            percentage = (count / matched_count) * 100
            matrix.at[emotion_group, merged_behavior] = round(percentage, 4)
    
    return matrix

def output_matrix(matrix: pd.DataFrame, output_dir: str):
    print(f"\n" + "="*80)
    print("Emotion Group - Merged Behavior Ratio Matrix (unit: %, calculated based on matched data):")
    print("="*80)
    print(matrix)
    
    matrix_total = round(matrix.values.sum(), 2)
    print(f"\nSum of all proportions in matrix: {matrix_total}%")
    if abs(matrix_total - 100) < 0.01:
        print("✅ Validation passed: Sum of proportions based on matched data is 100%")
    else:
        print(f"⚠️ Note: Sum of proportions deviates from 100% (normal due to calculation based only on matched data)")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'emotion_group_merged_behavior_ratio_matrix_1.csv')
    matrix.to_csv(output_path, encoding='utf-8-sig')
    
    print(f"\n" + "="*80)
    print(f"Matrix saved to: {output_path}")
    print(f"Matrix value meaning: (frequency of emotion group-behavior combination / total matched emotion-behavior pairs) × 100%")
    print(f"="*80)

def main(csv_path: str, output_dir: str = ".\result"):
    print("="*80)
    print("Starting: Emotion Group - Merged Behavior Ratio Matrix Calculation (with unmatched items display)")
    print("="*80)
    
    try:
        df_transformed, total_pairs = extract_and_transform_pairs(csv_path)
        
        matrix = calculate_emotion_behavior_matrix(df_transformed, total_pairs)
        
        output_matrix(matrix, output_dir)
        
        print(f"\nExecution completed! Matrix dimensions: {matrix.shape[0]} emotion groups × {matrix.shape[1]} merged behavior categories")
    
    except Exception as e:
        print(f"\nExecution error: {str(e)}")

if __name__ == "__main__":
    MY_CSV_PATH = ".\dilemmas_with_detail_by_action_0912.csv"
    main(MY_CSV_PATH)