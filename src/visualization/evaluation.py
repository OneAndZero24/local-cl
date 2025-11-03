import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_for_prompt_methods(base_dir: str) -> None:
    plt.figure(figsize=(10, 6))

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path):
            label = f"{folder_name.split('_')[1][3:]}"

            accuracies = []

            # Iterate through CSV files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".csv"):
                    csv_path = os.path.join(folder_path, file_name)
                    
                    # Load CSV without header
                    df = pd.read_csv(csv_path, header=None)
                    
                    # Flatten the values and append them
                    accuracies.extend(df.values.flatten())
            
            if accuracies:
                # Create task IDs starting from 1
                task_ids = list(range(1, len(accuracies) + 1))
                
                # Plot with LaTeX label
                plt.plot(task_ids, accuracies, label=rf"$\lambda_{{\text{{IntDrift}}}}={label}$")

    plt.xlabel("Task ID", fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("AA [%]", fontsize=14)
    plt.yticks(fontsize=14)
    plt.rc('legend',fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path_to_save = os.path.join(base_dir, "impact_of_int_drift_reg_on_coda_p.png")

    plt.savefig(path_to_save)
    plt.close()