import pandas as pd
import os
import glob
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

original_path = os.getcwd()

# Define the folder containing the files
folder_path_npe = os.path.join(original_path, "Results/MAE_results")
folder_path_seq = os.path.join(original_path, "Experiments/vagner/Results/MAE_results")
folder_path_npse = os.path.join(original_path, "Results/MAE_results_NPSE")

# Find all files in the folder with "HPD" and "summary" in their names
file_pattern = os.path.join(folder_path_npe, "*HPD*summary*.csv")
files_hpd = [f for f in glob.glob(file_pattern)]

file_pattern_npse = os.path.join(folder_path_npse, "*KDE*summary*.csv")
files_hpd_npse = [f for f in glob.glob(file_pattern_npse)]


def create_heat_matrix(files, name="NPE"):
    sim_matrix = {}
    mae_matrices, se_matrices = {}, {}

    # Extract simulation budget from file paths
    simulation_budgets_lists = [
        int(os.path.basename(file).split("_")[-1].split(".")[0]) for file in files
    ]

    if name == "NPE":
        file_dict = {
            10000: [
                file
                for file, budget in zip(files, simulation_budgets_lists)
                if budget == 10000
            ],
            20000: [
                file
                for file, budget in zip(files, simulation_budgets_lists)
                if budget == 20000
            ],
        }

        budgets = [10000, 20000]
    elif name == "NPSE":
        file_dict = {
            10000: [
                file
                for file, budget in zip(files, simulation_budgets_lists)
                if budget == 10000
            ]
        }

        budgets = [10000]

    j = 0
    for file_list in file_dict.values():
        # Initialize a dictionary to store counts for each method per benchmark
        heat_matrix = {}
        benchmark_names = []

        # Initialize a DataFrame to store MAE values for visualization
        mae_matrix = pd.DataFrame(columns=heat_matrix.keys())
        se_matrix = pd.DataFrame(columns=heat_matrix.keys())

        for file in file_list:
            # extracting benchmark name from file name
            benchmark_name = "_".join(os.path.basename(file).split("_")[1:]).split(
                "_coverage"
            )[0]

            if benchmark_name not in benchmark_names:
                benchmark_names.append(benchmark_name)
            # Load the data
            data = pd.read_csv(file)

            # Save the column names of the dataset
            column_names = data.columns.tolist()[2:]

            # Replace "A-LOCART MAD" with "LOCART MAD" in column names
            column_names = [
                name.replace("A-LOCART MAD", "LOCART MAD") for name in column_names
            ]

            # Remove "MAD" from column names and modify "Local CDF" to "L-CDF"
            column_names = [
                name.replace(" MAD", "").replace("Local CDF", "L-CDF")
                for name in column_names
            ]

            # performance array
            mae_array = data.iloc[0, 2:].to_numpy(dtype=float)
            se_array = 2 * data.iloc[1, 2:].to_numpy(dtype=float)

            # Add MAE values to the matrix
            mae_matrix[benchmark_name] = mae_array
            se_matrix[benchmark_name] = se_array

        # Create a DataFrame to store significance coloring
        significance_matrix = pd.DataFrame(
            index=mae_matrix.index, columns=mae_matrix.columns
        )

        # Determine significant values
        k = 0
        for col in mae_matrix.columns:
            significance_matrix.loc[:, col] = False  # Initialize with False
            mae_values = mae_matrix.loc[:, col].to_numpy()
            se_values = se_matrix.loc[:, col].to_numpy()
            idxs = np.arange(0, len(mae_values))

            # Finding best index
            best_method_index = np.argmin(mae_values)
            if not np.isscalar(best_method_index):
                best_method_index = best_method_index[0]

            significance_matrix.iloc[best_method_index, k] = True

            # obtaining other best methods
            excluded_mae_array = np.delete(
                mae_values,
                best_method_index,
            )

            excluded_se_array = np.delete(
                se_values,
                best_method_index,
            )

            excluded_idxs_array = np.delete(
                idxs,
                best_method_index,
            )

            lim_sup = mae_values[best_method_index] + se_values[best_method_index]
            lim_inf = excluded_mae_array - excluded_se_array

            add_indexes = np.where(lim_sup >= lim_inf)[0]
            if add_indexes.size > 0:
                selected_indexes = excluded_idxs_array[add_indexes]
                significance_matrix.iloc[selected_indexes, k] = True
            k += 1

        significance_matrix.columns = benchmark_names
        se_matrix.columns = benchmark_names
        mae_matrix.columns = benchmark_names

        significance_matrix.index = column_names
        mae_matrix.index = column_names
        se_matrix.index = column_names

        sim_matrix[budgets[j]] = significance_matrix
        mae_matrices[budgets[j]] = mae_matrix
        se_matrices[budgets[j]] = se_matrix

        j += 1

    # producing the heatmap
    plt.rcParams.update({"font.size": 14})
    if len(budgets) > 1:
        fig, axes = plt.subplots(1, len(budgets), figsize=(18, 8))
        for idx, budget in enumerate(budgets):
            ax = axes[idx]
            mae_matrix = mae_matrices[budget]
            se_matrix = se_matrices[budget]
            significance_matrix = sim_matrix[budget]

            # Define a discrete colormap with two colors: green for significant, white for not significant
            cmap = ListedColormap(["white", "mediumseagreen"])  # Light green color

            sorted_benchmark_names = sorted(benchmark_names)

            # Define the desired order for specific columns
            desired_order = ["LOCART", "CDF", "L-CDF"]
            remaining_columns = [
                col for col in column_names if col not in desired_order
            ]
            ordered_columns = desired_order + remaining_columns

            # Reorder the columns (benchmarks) based on the sorted benchmark names
            mae_matrix = mae_matrix[sorted_benchmark_names]
            se_matrix = se_matrix[sorted_benchmark_names]
            significance_matrix = significance_matrix[sorted_benchmark_names]

            # Reorder the rows (methods) based on the desired order
            mae_matrix = mae_matrix.loc[ordered_columns]
            se_matrix = se_matrix.loc[ordered_columns]
            significance_matrix = significance_matrix.loc[ordered_columns]

            # Create a matrix for coloring based on significance
            color_matrix = significance_matrix.replace({False: 0, True: 1}).to_numpy()

            # Plot the heatmap with the discrete colormap
            heatmap = ax.imshow(color_matrix, cmap=cmap, aspect="auto")

            # Add gridlines to separate tiles
            ax.set_xticks(np.arange(-0.5, len(mae_matrix.columns), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(mae_matrix.index), 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Add text (MAE and SE values) to each tile
            for i in range(mae_matrix.shape[0]):
                for j in range(mae_matrix.shape[1]):
                    value = mae_matrix.iloc[i, j]
                    se_value = se_matrix.iloc[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value:.3f}\n({se_value:.4f})",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=10,
                    )

            # Set axis labels and ticks
            ax.set_xlabel("Benchmarks")
            if idx == 0:
                ax.set_ylabel("Methods")
            else:
                ax.set_ylabel("")

            ax.set_xticks(range(len(mae_matrix.columns)))
            ax.set_xticklabels(mae_matrix.columns, rotation=45, ha="right")

            ax.tick_params(axis="x", labelsize=10)
            ax.set_yticks(range(len(mae_matrix.index)))
            ax.set_yticklabels(mae_matrix.index)
            for tick, label in zip(ax.get_yticklabels(), mae_matrix.index):
                if label in ["LOCART", "CDF", "L-CDF"]:
                    tick.set_fontweight("bold")
            ax.set_title(f"Budget: {budget}")

        plt.tight_layout()
        # Save the figure to the specified path
        output_path = os.path.join(
            original_path, "Results", f"{name}_heatmap_figure.png"
        )
        output_path_pdf = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(output_path_pdf, format="pdf")
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        budget = budgets[0]
        mae_matrix = mae_matrices[budget]
        se_matrix = se_matrices[budget]
        significance_matrix = sim_matrix[budget]

        cmap = ListedColormap(["white", "mediumseagreen"])

        sorted_benchmark_names = sorted(benchmark_names)
        desired_order = ["LOCART", "CDF", "L-CDF"]
        remaining_columns = [col for col in column_names if col not in desired_order]
        ordered_columns = desired_order + remaining_columns

        mae_matrix = mae_matrix[sorted_benchmark_names]
        se_matrix = se_matrix[sorted_benchmark_names]
        significance_matrix = significance_matrix[sorted_benchmark_names]

        mae_matrix = mae_matrix.loc[ordered_columns]
        se_matrix = se_matrix.loc[ordered_columns]
        significance_matrix = significance_matrix.loc[ordered_columns]

        color_matrix = significance_matrix.replace({False: 0, True: 1}).to_numpy()

        heatmap = ax.imshow(color_matrix, cmap=cmap, aspect="auto")

        ax.set_xticks(np.arange(-0.5, len(mae_matrix.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(mae_matrix.index), 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(mae_matrix.shape[0]):
            for j in range(mae_matrix.shape[1]):
                value = mae_matrix.iloc[i, j]
                se_value = se_matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.3f}\n({se_value:.4f})",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        ax.set_xlabel("Benchmarks")
        ax.set_ylabel("Methods")
        ax.set_xticks(range(len(mae_matrix.columns)))
        ax.set_xticklabels(mae_matrix.columns, rotation=45, ha="right")
        ax.tick_params(axis="x", labelsize=10)
        ax.set_yticks(range(len(mae_matrix.index)))
        ax.set_yticklabels(mae_matrix.index)
        for tick, label in zip(ax.get_yticklabels(), mae_matrix.index):
            if label in ["LOCART", "CDF", "L-CDF"]:
                tick.set_fontweight("bold")
        ax.set_title(f"Budget: {budget}")

        plt.tight_layout()
        output_path = os.path.join(
            original_path, "Results", f"{name}_heatmap_figure.png"
        )
        output_path_pdf = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(output_path_pdf, format="pdf")
        plt.show()
    return sim_matrix, mae_matrices, se_matrices


# Example usage
sim_mat, mae_mat, se_mat = create_heat_matrix(
    files_hpd,
    name="NPE",
)

# For NPSE files
sim_mat_npse, mae_mat_npse, se_mat_npse = create_heat_matrix(
    files_hpd_npse,
    name="NPSE",
)


def method_counting(files):
    # Initialize a counter for method counts
    method_counts = {
        "LOCART": 0,
        "A-LOCART": 0,
        "Global CP": 0,
        "Naive": 0,
        "CDF CP": 0,
        "L-CDF CP": 0,
        "HDR": 0,
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

        add_indexes = np.where(lim_sup >= lim_inf)[0]
        if add_indexes.size > 0:
            selected_indexes = new_index_array[add_indexes]
            for index in selected_indexes:
                sel_key = keys[index]
                # Increment the count for the method
                method_counts[sel_key] += 1
    return method_counts


def method_counting_exclude_locart_cdf(files):
    # Initialize a counter for method counts excluding LOCART and CDF CP
    method_counts = {
        "LOCART": 0,
        "Global": 0,
        "Naive": 0,
        "CDF": 0,
        "L-CDF": 0,
        "HDR": 0,
    }
    keys = list(method_counts.keys())

    idx_array = np.arange(0, len(method_counts))

    # LOCART has index 0 and CDF CP has index 4
    # Process each file
    for file in files:
        # Load the data
        data = pd.read_csv(file)
        # performance array
        mae_array = data.iloc[0, 1:].to_numpy(dtype=float)
        se_array = 2 * data.iloc[1, 1:].to_numpy(dtype=float)
        mae_array = np.delete(mae_array, [0])
        se_array = np.delete(se_array, [0])

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

method_counts_hpd_2 = method_counting_exclude_locart_cdf(files_hpd)

method_counts_hpd_npse = method_counting(files_hpd_npse)

method_counts_hpd_npse_2 = method_counting_exclude_locart_cdf(files_hpd_npse)

# Sorting the method counts in descending order
sorted_counts_hpd = dict(
    sorted(method_counts_hpd.items(), key=lambda item: item[1], reverse=True)
)

sorted_counts_hpd_2 = dict(
    sorted(method_counts_hpd_2.items(), key=lambda item: item[1], reverse=True)
)

# Sorting the method counts for NPSE in descending order
sorted_counts_hpd_npse = dict(
    sorted(method_counts_hpd_npse.items(), key=lambda item: item[1], reverse=True)
)
sorted_counts_hpd_npse_2 = dict(
    sorted(method_counts_hpd_npse_2.items(), key=lambda item: item[1], reverse=True)
)

# Create a graph for HPD counting
fig_hpd, ax_hpd = plt.subplots(figsize=(10, 5))

ax_hpd.bar(sorted_counts_hpd_2.keys(), sorted_counts_hpd_2.values(), color="skyblue")
ax_hpd.set_title("Best Performance Count (HPD)")
ax_hpd.set_ylabel("Count")
ax_hpd.set_xticks(range(len(sorted_counts_hpd_2)))
ax_hpd.set_xticklabels(
    [
        (
            f"$\\bf{{{label}}}$"
            if label in ["LOCART", "CDF", "A-LOCART", "L-CDF"]
            else label
        )
        for label in sorted_counts_hpd_2.keys()
    ]
)
ax_hpd.set_ylim(0, max(sorted_counts_hpd_2.values()) + 1)

plt.tight_layout()
plt.show()


# Create a graph for HPD counting
fig_hpd, ax_hpd = plt.subplots(figsize=(10, 5))

ax_hpd.bar(sorted_counts_hpd_2.keys(), sorted_counts_hpd_2.values(), color="skyblue")
ax_hpd.set_title("Best Performance Count (HPD)")
ax_hpd.set_ylabel("Count")
ax_hpd.set_xticks(range(len(sorted_counts_hpd_2)))
ax_hpd.set_xticklabels(
    [
        (
            f"$\\bf{{{label}}}$"
            if label in ["LOCART", "CDF", "A-LOCART", "L-CDF"]
            else label
        )
        for label in sorted_counts_hpd_2.keys()
    ]
)
ax_hpd.set_ylim(0, max(sorted_counts_hpd_2.values()) + 1)

plt.tight_layout()
plt.show()
