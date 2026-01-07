import pandas as pd
import numpy as np
import os
from typing import Dict, List

def load_matrix_from_csv(file_path: str) -> pd.DataFrame:
    try:
        matrix = pd.read_csv(file_path, index_col=0)
        print(f"Successfully loaded matrix data, shape: {matrix.shape}")
        return matrix
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading matrix: {str(e)}")
        return pd.DataFrame()

def evaluate_matrix(conflict_matrix: pd.DataFrame) -> float:
    maslow_order: Dict[str, int] = {
        'physiological': 1,
        'safety': 2,
        'love and belonging': 3,
        'self-esteem': 4,
        'self-actualization': 5
    }
    
    matrix_needs = set(conflict_matrix.index)
    theory_needs = set(maslow_order.keys())
    
    if matrix_needs != theory_needs:
        missing = theory_needs - matrix_needs
        extra = matrix_needs - theory_needs
        error_msg = "Matrix need names do not match Maslow theory:\n"
        if missing:
            error_msg += f"Missing required needs: {', '.join(missing)}\n"
        if extra:
            error_msg += f"Extra unknown needs: {', '.join(extra)}\n"
        raise ValueError(error_msg)
    
    n = len(conflict_matrix)
    ideal_matrix = np.zeros((n, n), dtype=float)
    needs_list: List[str] = conflict_matrix.index.tolist()
    
    for i in range(n):
        for j in range(n):
            if i == j:
                ideal_matrix[i, j] = 0.0
            else:
                need_a_level = maslow_order[needs_list[i]]
                need_b_level = maslow_order[needs_list[j]]
                if need_a_level < need_b_level:
                    ideal_matrix[i, j] = 1.0
                else:
                    ideal_matrix[i, j] = -1.0
    
    mask = ~np.eye(n, dtype=bool)
    actual_values = conflict_matrix.values[mask]
    ideal_values = ideal_matrix[mask]
    
    errors = actual_values - ideal_values
    squared_errors = errors **2
    
    mse = np.mean(squared_errors)
    
    score = 100 - (mse / 4) * 100
    score = round(max(0, min(100, score)), 2)
    
    print("\n===== Matrix Evaluation Details =====")
    print("1. Theoretical Ideal Matrix:")
    ideal_df = pd.DataFrame(ideal_matrix, columns=needs_list, index=needs_list)
    print(ideal_df.to_string(float_format=lambda x: f"{x:.1f}"))
    
    print("\n2. Error Analysis:")
    print(f"   Mean Squared Error (MSE): {mse:.6f}")
    print(f"   Theoretical Consistency Score: {score} / 100")
    
    print("\n3. Main Discrepancies (Top 5 by absolute error):")
    error_indices = np.argwhere(mask).tolist()
    error_data = []
    
    for idx, (i, j) in enumerate(error_indices):
        need_a = needs_list[i]
        need_b = needs_list[j]
        error_data.append({
            'Conflict Pair': f"{need_a} → {need_b}",
            'Actual Value': round(actual_values[idx], 4),
            'Ideal Value': round(ideal_values[idx], 4),
            'Error': round(errors[idx], 4),
            'Squared Error': round(squared_errors[idx], 4)
        })
    
    error_df = pd.DataFrame(error_data)
    error_df = error_df.sort_values(by='Squared Error', ascending=False).head(5)
    print(error_df.to_string(index=False))
    
    return score

def get_evaluation_comment(score: float) -> str:
    if score >= 80:
        return "Evaluation: Matrix highly consistent with Maslow theory, model's need selection logic is reasonable"
    elif score >= 50:
        return "Evaluation: Matrix basically consistent with Maslow theory, model's need selection logic is acceptable"
    else:
        return "Evaluation: Matrix deviates significantly from Maslow theory, need to check model selection logic or data quality"

def main():
    csv_path = input("Please enter the path to the matrix CSV file: ").strip()
    
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' does not exist")
        return
    
    matrix = load_matrix_from_csv(csv_path)
    if matrix.empty:
        print("Unable to continue evaluation, program exiting")
        return
    
    print("\nLoaded matrix data:")
    print(matrix.to_string(float_format=lambda x: f"{x:.6f}"))
    
    try:
        score = evaluate_matrix(matrix)
        comment = get_evaluation_comment(score)
        
        print("\n===== Final Evaluation Result =====")
        print(f"Theoretical Consistency Score: {score} / 100")
        print(comment)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()