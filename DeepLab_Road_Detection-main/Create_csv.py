import csv
import os
import sys

dataset_dir = r'C:\Users\niles\Downloads\RAF28JUN2024039216009800058SSANSTUC00GTDA\BH_RAF28JUN2024039216009800058SSANSTUC00GTDA'
def generate_csv(dataset_dir):
    # Define metadata fields to be extracted
    meta_data = {
        'ProductID': '2465625271',
        'DateOfPass': '28-JUN-2024',
        'NoOfBands': '3',
        'BandNumbers': '234',
        'ProdULLat': '21.019888',
        'ProdULLon': '76.645805',
        'ProdLRLat': '20.259295',
        'ProdLRLon': '77.484794',
        # Add more fields as required
    }

    # Statistics from the XML
    band_stats = {
        'BAND2': {
            'Maximum': '215',
            'Mean': '146.85685564794',
            'Minimum': '0',
            'StdDev': '74.875195073175'
        },
        'BAND3': {
            'Maximum': '216',
            'Mean': '147.59067517202',
            'Minimum': '0',
            'StdDev': '75.298834834543'
        },
        'BAND4': {
            'Maximum': '216',
            'Mean': '148.63498064794',
            'Minimum': '0',
            'StdDev': '75.813045359875'
        }
    }

    # Prepare CSV data
    csv_data = [['Metadata', 'Value']]  # Header for metadata
    csv_data.extend([[key, value] for key, value in meta_data.items()])
    csv_data.append([])  # Add a blank row between metadata and statistics
    csv_data.append(['Band', 'Maximum', 'Mean', 'Minimum', 'StdDev'])  # Header for band stats

    # Add band stats to CSV data
    for band, stats in band_stats.items():
        csv_data.append([band, stats['Maximum'], stats['Mean'], stats['Minimum'], stats['StdDev']])

    # Define the path for the CSV file
    csv_file_path = os.path.join(dataset_dir, 'metadata.csv')

    # Save to CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"metadata.csv created successfully at {csv_file_path}!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_csv.py <directory_path>")
        sys.exit(1)

    target_directory = sys.argv[1]
    
    # Check if the provided directory path exists
    if not os.path.isdir(target_directory):
        print(f"Error: The directory '{target_directory}' does not exist.")
        sys.exit(1)
    
    generate_csv(target_directory)
