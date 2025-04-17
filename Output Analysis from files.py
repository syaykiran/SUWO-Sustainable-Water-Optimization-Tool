import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.backends.backend_pdf import PdfPages
import webbrowser
import os

# Define the pattern for each type of CSV file
file_patterns = {
    "P_END_SQ": "EXP_PEN_END_SQTOT*.csv",
    "P_MIN_SQ": "EXP_PEN_MIN_SQTOT*.csv",
    "P_MAX_SQ": "EXP_PEN_MAX_SQTOT*.csv"
}

# Load data from CSV files by matching the pattern
data = {}

modern_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Modern color palette

for metric, pattern in file_patterns.items():
    files = glob.glob(pattern)
    for file in files:
        # Extract index from filename
        idx = int(os.path.splitext(os.path.basename(file))[0].split("TOT")[1])

        # Read CSV file
        df = pd.read_csv(file)

        # Rename columns consistently
        df.columns = ['Index', metric]
        df.set_index('Index', inplace=True)

        # Store data in a dictionary by index and metric
        if idx not in data:
            data[idx] = {}
        data[idx][metric] = df[metric]

    # Create a single PDF file with all plots on separate pages

output_pdf_path = 'DRY_All_Penalty_Values_Plots.pdf'

with PdfPages(output_pdf_path) as pdf:
    for idx, metrics in data.items():
        plt.figure(figsize=(10, 6))
        if "P_MIN_SQ" in metrics:
            plt.plot(metrics["P_MIN_SQ"], '-', color=modern_colors[0], label="P_MIN_SQ")
        if "P_MAX_SQ" in metrics:
            plt.plot(metrics["P_MAX_SQ"], '-', color=modern_colors[1], label="P_MAX_SQ")
        if "P_END_SQ" in metrics:
            plt.plot(metrics["P_END_SQ"], '-', color=modern_colors[2], label="P_END_SQ")

        plt.ylabel("Penalty Values")
        plt.xlabel("Index")
        plt.title(f"Penalty Values Over Generations - Run {idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Add the current figure as a new page in the PDF
        pdf.savefig()
        plt.close()

print("All plots saved in a single PDF file: All_Penalty_Values_Plots.pdf")
# Open the generated PDF

webbrowser.open('file://' + os.path.realpath(output_pdf_path))
