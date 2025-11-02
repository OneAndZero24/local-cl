import torch
import matplotlib.pyplot as plt
import seaborn as sns


saved_dict = torch.load("model_checkpoint.pth")

# head.classifier.kernels_centers
# head.classifier.log_shapes
# layers.0.kernels_centers
# layers.0.log_shapes
# layers.1.kernels_centers
# layers.1.log_shapes

kernels_centers = saved_dict["head.classifier.kernels_centers"].cpu().numpy()
log_shapes = saved_dict["head.classifier.log_shapes"].cpu().numpy()

num_kernels, in_features = kernels_centers.shape

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(kernels_centers, cmap="coolwarm", center=0)
plt.title("Kernel Centers Heatmap")
plt.xlabel("Feature Index")
plt.ylabel("Kernel Index")

plt.subplot(1, 2, 2)
sns.heatmap(log_shapes, cmap="coolwarm", center=0)
plt.title("Log Shapes Heatmap")
plt.xlabel("Feature Index")
plt.ylabel("Kernel Index")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(kernels_centers.flatten(), bins=50, kde=True)
plt.title("Kernel Centers Distribution")

plt.subplot(1, 2, 2)
sns.histplot(log_shapes.flatten(), bins=50, kde=True)
plt.title("Log Shapes Distribution")

plt.tight_layout()
plt.show()
