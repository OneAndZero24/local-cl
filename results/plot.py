import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

files = ['mnist.csv', 'fmnist.csv', 'svhn.csv', 'cifar10.csv']
titles = ['MLP MNIST', 'MLP Fashion-MNIST', 'LeNet SVHN', 'ResNet18 CIFAR-10']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cmap = cm.viridis

for idx, (file, title) in enumerate(zip(files, titles)):
    df = pd.read_csv(file, na_values='NA')
    numeric_df = df.drop(columns='Method')
    ax = axes[idx // 2, idx % 2]
    data = [numeric_df[col].dropna() for col in numeric_df.columns]
    colors = [cmap(i / len(data)) for i in range(len(data))]
    positions = range(1, len(data) + 1)

    box = ax.boxplot(data, positions=positions, patch_artist=True, showmeans=False)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    for element in ['whiskers', 'caps', 'medians']:
        for item in box[element]:
            item.set_color('black')

    ax.set_xticks(positions)
    ax.set_xticklabels(numeric_df.columns, rotation=15)
    ax.set_title(title)
    ax.set_ylabel('ACC')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('res_plot.png', dpi=300)
plt.close()
