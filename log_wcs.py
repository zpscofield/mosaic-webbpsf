import csv
import os
from astropy.io import fits

# Define the directory containing the FITS files and the output CSV file path
directory = 'cal_files'
csv_filepath = 'wcs_metadata_info.csv'

# Function to extract all keywords from a FITS header
def extract_header_parameters(header):
    header_params = {}
    for key in header.keys():
        if key:
            value = header.get(key, 'N/A')
            if isinstance(value, float):
                formatted_value = f"{value:.13e}" if 'e' in f"{value:.13e}" else f"{value:.13f}" # Check for scientific notation
                header_params[key] = formatted_value
            else:
                header_params[key] = value
    return header_params

# This function updates the field names in case all field names are not present in all exposures. This is necessary
# for SIP coefficients.
def update_fieldnames(current_fieldnames, new_keys):
    for key in new_keys:
        if key not in current_fieldnames:
            current_fieldnames.append(key)
    return current_fieldnames

# Initialize fieldnames with common metadata fields - these are the items which are not in the image header but are
# instead in the primary header. A field name for the filename is also initialized.
initial_fieldnames = ['Filename', 'FILTER', 'DATE-OBS', 'DETECTOR']
all_fieldnames = initial_fieldnames.copy()

# Collect all possible fieldnames from all FITS files before populating the values.
for filename in os.listdir(directory):
    if filename.endswith('cal.fits'):
        filepath = os.path.join(directory, filename)
        with fits.open(filepath) as hdul:
            header1 = hdul[1].header
            all_fieldnames = update_fieldnames(all_fieldnames, header1.keys())

# Open the CSV file for writing
with open(csv_filepath, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
    writer.writeheader()

    # Iterate over FITS files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('cal.fits'):
            filepath = os.path.join(directory, filename)
            with fits.open(filepath) as hdul:
                header0 = hdul[0].header # Contains filter, observation date, and detector information
                header1 = hdul[1].header # Contains all relevant information for WCS coordinate transformations.

                # Extract all information from the image header
                header1_params = extract_header_parameters(header1)
                
                # Extract additional metadata from the primary header
                obs_filter = header0.get('FILTER', 'UNKNOWN')
                obs_date = header0.get('DATE-OBS', 'UNKNOWN')
                detector = header0.get('DETECTOR', 'UNKNOWN')

                # Create a dictionary with all the information to write to CSV
                info = {
                    'Filename': filename,
                    'FILTER': obs_filter,
                    'DATE-OBS': obs_date,
                    'DETECTOR': detector
                }
                info.update(header1_params) # Add the image header values

                # Ensure all keys are in the dictionary - This is to exclude section titles within the header.
                for key in all_fieldnames:
                    if key not in info:
                        info[key] = 'N/A'

                # Write the dictionary to the CSV file
                writer.writerow(info)

print(f"WCS and metadata information has been saved to {csv_filepath}")
