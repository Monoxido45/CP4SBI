import pandas as pd
import os
import glob
import numpy as np

original_path = os.getcwd()

# Define the folder containing the files
folder_path = os.path.join(original_path, "Results/MAE_results")

# Find all files in the folder with "HPD" and "summary" in their names
file_pattern = os.path.join(folder_path, "*HPD*summary*.csv")
files_hpd = [f for f in glob.glob(file_pattern)]

file_pattern_waldo = os.path.join(folder_path, "*WALDO*summary*.csv")
files_waldo = [f for f in glob.glob(file_pattern_waldo)]

files = files_hpd + files_waldo

# Initialize a counter for method counts
method_counts = {"LOCART": 0, "Global CP": 0, "Naive": 0, "CDF CP": 0}
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
    new_index_array = np.delete(
        idx_array,
        best_method_index,
    )

    sel_key = keys[best_method_index]
    # Increment the count for the best method
    method_counts[sel_key] += 1

    lim_inf = mae_array[-best_method_index] - se_array[-best_method_index]

    lim_sup = mae_array[best_method_index] + se_array[best_method_index]

    add_indexes = np.where(lim_sup - lim_inf > 0)[0]
    if add_indexes.size > 0:
        selected_indexes = new_index_array[add_indexes]
        for index in selected_indexes:
            sel_key = keys[index]
            # Increment the count for the method
            method_counts[sel_key] += 1

# Make a barchart of the counts
import matplotlib.pyplot as plt

# Sort the method counts in descending order
sorted_counts = dict(
    sorted(method_counts.items(), key=lambda item: item[1], reverse=True)
)

# Create the bar chart
bars = plt.bar(sorted_counts.keys(), sorted_counts.values(), color="skyblue")

# Add labels and title
plt.xlabel("Methods")
plt.ylabel("Best Performance Count (HPD score)")

# Highlight LOCART and CDF CP labels in bold on the x-axis
ax = plt.gca()
ax.set_xticks(range(len(sorted_counts)))
ax.set_xticklabels(
    [
        f"$\\bf{{{label}}}$" if label in ["LOCART", "CDF CP"] else label
        for label in sorted_counts.keys()
    ]
)

# Display the chart
plt.tight_layout()
plt.show()
