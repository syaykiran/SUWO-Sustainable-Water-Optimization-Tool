# import pandas as pd
#
# # Dictionary to hold the number of days for each month
# days_in_month = {
#     1: 31,  # January
#     2: 28,  # February (29 in leap years, handled later)
#     3: 31,  # March
#     4: 30,  # April
#     5: 31,  # May
#     6: 30,  # June
#     7: 31,  # July
#     8: 31,  # August
#     9: 30,  # September
#     10: 31,  # October
#     11: 30,  # November
#     12: 31  # December
# }
#
#
# def is_leap_year(year):
#     """Check if a given year is a leap year."""
#     return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
#
#
# def convert_to_cms(row):
#     """Convert hm³ to cms for a given row."""
#     year = int(row['YEAR'])
#     month = int(row['MONTH'])
#
#     # Adjust February days for leap years
#     days = days_in_month[month] + (1 if month == 2 and is_leap_year(year) else 0)
#
#     # Convert all columns except YEAR and MONTH
#     for col in row.index[2:]:
#         if row[col] != -9999:
#             row[col] = (row[col] * 1000000) / (days * 86400)  # Convert using correct days in the month
#         else:
#             row[col] = -9999  # Keep missing data as is
#     return row
#
#
# def convert_to_hm3(row):
#     """Convert cms to hm³ for a given row."""
#     year = int(row['YEAR'])
#     month = int(row['MONTH'])
#
#     # Adjust February days for leap years
#     days = days_in_month[month] + (1 if month == 2 and is_leap_year(year) else 0)
#
#     # Convert all columns except YEAR and MONTH
#     for col in row.index[2:]:
#         if row[col] != -9999:
#             row[col] = (row[col] * days * 86400) / 1000000  # Convert using correct days in the month
#         else:
#             row[col] = -9999  # Keep missing data as is
#     return row
#
#
# def process_csv(input_csv, output_csv, conversion_type="to_hm3"):
#     """Process the CSV file to convert units based on the specified conversion type."""
#     # Load the data from the CSV file
#     df = pd.read_csv(input_csv)
#
#     # Apply the conversion function row-wise
#     if conversion_type == "to_hm3":
#         df = df.apply(convert_to_hm3, axis=1)
#     elif conversion_type == "to_cms":
#         df = df.apply(convert_to_cms, axis=1)
#     else:
#         raise ValueError("Invalid conversion_type. Use 'to_hm3' or 'to_cms'.")
#
#     # Export the processed data to a new CSV file
#     df.to_csv(output_csv, index=False)
#
#
# # Example usage
# input_csv_file = "CMSCMS.csv"
# output_csv_file_hm3 = "output_data_hm3.csv"
# output_csv_file_cms = "output_data_cms.csv"
#
# # Convert to hm³
# process_csv(input_csv_file, output_csv_file_hm3, conversion_type="to_hm3")
#
# # Convert back to cms
# # process_csv(output_csv_file_hm3, output_csv_file_cms, conversion_type="to_cms")
##
import pandas as pd

def process_flow_data(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(input_file)

    # Replace missing data (if needed) - assuming missing data is denoted by -9999 in the dataset
    data.fillna(-9999, inplace=True)

    # Calculate the annual flow for each node by summing the monthly flows
    annual_flows = data.groupby('YEAR').sum().reset_index()

    # Remove the 'MONTH' column as it's not needed for annual calculations
    annual_flows = annual_flows.drop(columns=['MONTH'])

    # Initialize a dictionary to store wet years for each node
    wet_years = {}

    # Calculate the wet years for each node
    for node in annual_flows.columns[1:]:  # Skip the 'YEAR' column
        # Calculate the mean and standard deviation of the annual flows for the node
        mean_flow = annual_flows[node].mean()
        std_flow = annual_flows[node].std()

        # Define the threshold for wet years (mean + 1 standard deviation)
        threshold = mean_flow + std_flow

        # Identify the years where the annual flow exceeds the threshold
        wet_years[node] = annual_flows[annual_flows[node] > threshold]['YEAR'].tolist()

    # Find the intersection of wet years across all nodes
    intersection_years = set(wet_years[list(wet_years.keys())[0]])
    for node in wet_years.keys():
        intersection_years = intersection_years.intersection(set(wet_years[node]))

    # Convert the intersection years to a DataFrame
    intersection_years_df = pd.DataFrame({'Intersection Years': sorted(intersection_years)})

    # Export the intersection years to a CSV file
    intersection_years_df.to_csv(output_file, index=False)

    print(f"Intersection of wet years across all nodes has been exported to {output_file}")

# Input and Output file paths
input_file = 'sako.csv'  # Replace with your input CSV file path
output_file = 'intersection_wet_years.csv'  # Replace with your desired output CSV file path

# Call the function to process the data and find the intersection
process_flow_data(input_file, output_file)
