import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

original_path = os.getcwd()

# Define the folder containing the files
folder_path = os.path.join(original_path, "Results/MAE_results")

# Find all files in the folder with "HPD" and "summary" in their names
file_pattern = os.path.join(folder_path, "*HPD*summary*.csv")
files_hpd = [f for f in glob.glob(file_pattern) if "gaussian_mixture" not in f]

file_pattern_waldo = os.path.join(folder_path, "*WALDO*summary*.csv")
files_waldo = [
    f
    for f in glob.glob(file_pattern_waldo)
    if all(excluded not in f for excluded in ["gaussian_mixture"])
]

files_all = files_hpd + files_waldo


def method_counting(files):
    # Initialize a counter for method counts
    method_counts = {
        "LOCART": 0,
        "A-LOCART": 0,
        "Global CP": 0,
        "Naive": 0,
        "CDF CP": 0,
        "L-CDF CP": 0,
    }
    keys = list(method_counts.keys())

    idx_array = np.arange(0, len(method_counts))

    # Process each file
    for file in files:
        # Load the data
        data = pd.read_csv(file)
        # performance array
        mae_array = data.iloc[0, 1:].to_numpy(dtype=float)
        se_array = 2 * data.iloc[1, 1:].to_numpy(dtype=float)

        # obtaining method with best performance
        best_method_index = np.argmin(mae_array)
        sel_key = keys[best_method_index]

        new_index_array = np.delete(
            idx_array,
            best_method_index,
        )

        excluded_mae_array = np.delete(
            mae_array,
            best_method_index,
        )
        excluded_se_array = np.delete(
            se_array,
            best_method_index,
        )

        # Increment the count for the best method
        method_counts[sel_key] += 1

        lim_sup = mae_array[best_method_index] + se_array[best_method_index]

        lim_inf = excluded_mae_array - excluded_se_array

        add_indexes = np.where(lim_sup - lim_inf > 0)[0]
        if add_indexes.size > 0:
            selected_indexes = new_index_array[add_indexes]
            for index in selected_indexes:
                sel_key = keys[index]
                # Increment the count for the method
                method_counts[sel_key] += 1
    return method_counts


method_counts_hpd = method_counting(files_hpd)
method_counts_waldo = method_counting(files_waldo)
method_counts_all = method_counting(files_all)

# Sorting the method counts in descending order
sorted_counts_hpd = dict(
    sorted(method_counts_hpd.items(), key=lambda item: item[1], reverse=True)
)

sorted_counts_waldo = dict(
    sorted(method_counts_waldo.items(), key=lambda item: item[1], reverse=True)
)

sorted_counts_all = dict(
    sorted(method_counts_all.items(), key=lambda item: item[1], reverse=True)
)


# Create subplots for each sorted count in a single row
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot for HPD
axes[0].bar(sorted_counts_hpd.keys(), sorted_counts_hpd.values(), color="skyblue")
axes[0].set_title("Best Performance Count (HPD)")
axes[0].set_ylabel("Count")
axes[0].set_xticks(range(len(sorted_counts_hpd)))
axes[0].set_xticklabels(
    [
        (
            f"$\\bf{{{label}}}$"
            if label in ["LOCART", "CDF CP", "A-LOCART", "L-CDF CP"]
            else label
        )
        for label in sorted_counts_hpd.keys()
    ]
)

# Plot for WALDO
axes[1].bar(
    sorted_counts_waldo.keys(), sorted_counts_waldo.values(), color="lightgreen"
)
axes[1].set_title("Best Performance Count (WALDO)")
axes[1].set_xticks(range(len(sorted_counts_waldo)))
axes[1].set_xticklabels(
    [
        (
            f"$\\bf{{{label}}}$"
            if label in ["LOCART", "CDF CP", "A-LOCART", "L-CDF CP"]
            else label
        )
        for label in sorted_counts_waldo.keys()
    ]
)

# Plot for All
axes[2].bar(sorted_counts_all.keys(), sorted_counts_all.values(), color="salmon")
axes[2].set_title("Best Performance Count (All)")
axes[2].set_xlabel("Methods")
axes[2].set_xticks(range(len(sorted_counts_all)))
axes[2].set_xticklabels(
    [
        (
            f"$\\bf{{{label}}}$"
            if label in ["LOCART", "CDF CP", "A-LOCART", "L-CDF CP"]
            else label
        )
        for label in sorted_counts_all.keys()
    ]
)
# Set individual y-axis limits for each plot
axes[0].set_ylim(0, max(sorted_counts_hpd.values()) + 1)
axes[1].set_ylim(0, max(sorted_counts_waldo.values()) + 1)
axes[2].set_ylim(0, max(sorted_counts_all.values()) + 1)

# Adjust layout and display the chart
plt.tight_layout()
plt.show()
