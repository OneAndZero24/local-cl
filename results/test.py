import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, combine_pvalues

files = {
    'mnist': 'mnist.csv',
    'fmnist': 'fmnist.csv',
    'cifar10': 'cifar10.csv',
    'svhn': 'svhn.csv',
}

local_methods = ['Local LT', 'Local LT Head', 'Local', 'Local Head']

def load_and_clean(filename):
    df = pd.read_csv(filename)
    df.replace('NA', np.nan, inplace=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col])
    return df

p_values = {method: [] for method in local_methods}
mean_diffs = {method: [] for method in local_methods}

for ds_name, file in files.items():
    df = load_and_clean(file)
    for method in local_methods:
        paired = df[['ReLU', method]].dropna()
        if len(paired) < 2:
            print(f"Not enough data for {method} vs ReLU on {ds_name}")
            continue

        stat, p = ttest_rel(paired[method], paired['ReLU'])
        p_values[method].append(p)

        mean_diff = (paired[method] - paired['ReLU']).mean()
        mean_diffs[method].append(mean_diff)

        print(f"{ds_name}: {method} vs ReLU p={p:.4f}, mean diff={mean_diff:.4f}")

print("\nCombined p-values across datasets (Fisher's method):")
alpha = 0.05
m = len(local_methods)
adjusted_alpha = alpha / m

for method in local_methods:
    ps = p_values[method]
    if len(ps) == 0:
        print(f"{method}: no p-values to combine")
        continue

    combined_stat, combined_p = combine_pvalues(ps, method='fisher')

    mean_diff_overall = np.mean(mean_diffs[method]) if mean_diffs[method] else float('nan')
    improved = mean_diff_overall > 0

    signif = combined_p < adjusted_alpha

    print(f"\nMethod: {method}")
    print(f"  Combined p-value = {combined_p:.6f}")
    print(f"  Bonferroni corrected alpha = {adjusted_alpha:.6f}")
    print(f"  Mean difference (local - ReLU) across datasets = {mean_diff_overall:.4f}")
    if signif and improved:
        print("  --> Statistically significant improvement over ReLU after correction.")
    elif signif and not improved:
        print("  --> Statistically significant difference but local method is worse on average.")
    else:
        print("  --> No statistically significant improvement over ReLU.")

