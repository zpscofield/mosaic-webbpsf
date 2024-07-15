import os
import glob
import json
import numpy as np
import copy as cp
import time
import datetime
import csv
from scipy.ndimage import rotate, gaussian_filter
from astropy.io import fits, ascii
from astropy.wcs import WCS, Sip
from jwst.resample.resample_utils import decode_context
from mpi4py import MPI
import webbpsf

# Load configuration from JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Set environment variables from config
os.environ['CRDS_PATH'] = config['crds_path']
os.environ['CRDS_SERVER_URL'] = config['crds_server_url']
os.environ['WEBBPSF_PATH'] = config['webbpsf_path']

def current_time_string():
    """
    Returns the current time as a formatted string.
    
    Returns:
        str: The current time formatted as "HH:MM:SS".
    """
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")

# List to denote which columns correspond to a string value rather than a float value. 
string_columns = [
    'Filename', 'XTENSION', 'EXTNAME', 'REFFRAME', 'EPH_TYPE', 'RADESYS', 'BUNIT', 'FILTER', 'DATE-OBS', 'DETECTOR', 'S_REGION', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2'
]

def load_wcs_metadata(csv_path):
    """
    Loads WCS metadata from a CSV file.
    
    Parameters:
        csv_path (str): Path to the CSV file containing WCS metadata for each exposure.
    
    Returns:
        dict: A dictionary where each key is a filename and each value is a dictionary of WCS metadata.
    """
    wcs_metadata = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            exposure_name = row['Filename']
            wcs_info = {}
            for key, value in row.items():
                # This section initializes values as either float values or string values. 
                if key in string_columns:
                    wcs_info[key] = value
                else:
                    try:
                        wcs_info[key] = float(value)
                    except ValueError:
                        wcs_info[key] = value
            wcs_metadata[exposure_name] = wcs_info # The final dictionary wcs_metadata includes all information from the image header as well as necessary
                                                   # information from the primary header (detector, filter, observation date). It is organized with each
                                                   # filename being a key.  
                                                   
    return wcs_metadata

# Create WCS object from CSV metadata
def create_wcs_from_csv(wcs_info):
    """
    Creates a WCS object from WCS metadata.
    
    Parameters:
        wcs_info (dict): Dictionary containing WCS metadata.
    
    Returns:
        WCS: An astropy WCS object created from the metadata.
    """
    header = fits.Header() # Initialize an empty header
    
    wcs_keys = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'WCSAXES', 'RADECSYS', 'RA_V1', 
                'DEC_V1', 'PA_V3', 'S_REGION', 'V2_REF', 'V3_REF', 'VPARITY', 'V3I_YANG', 
                'RA_REF', 'DEC_REF', 'ROLL_REF', 'VELOSYS'] # Set of keys that are relevant for creating a WCS object
    
    for key in wcs_keys:
        if key in wcs_info:
            if key == 'WCSAXES':
                header[key] = int(wcs_info[key]) 
            else:
                header[key] = wcs_info[key]
    
    # Handle SIP coefficients
    if 'A_ORDER' in wcs_info and wcs_info['A_ORDER'] != 'N/A':
        header['A_ORDER'] = int(wcs_info['A_ORDER'])
        header['B_ORDER'] = int(wcs_info['B_ORDER'])
        header['AP_ORDER'] = int(wcs_info['AP_ORDER'])
        header['BP_ORDER'] = int(wcs_info['BP_ORDER'])
        
        for key in wcs_info:
            if key.startswith('A_') and wcs_info[key] != 'N/A':
                header[key] = wcs_info[key]
            elif key.startswith('B_') and wcs_info[key] != 'N/A':
                header[key] = wcs_info[key]
            elif key.startswith('AP_') and wcs_info[key] != 'N/A':
                header[key] = wcs_info[key]
            elif key.startswith('BP_') and wcs_info[key] != 'N/A':
                header[key] = wcs_info[key]
    
    # Create WCS object from the header
    wcs = WCS(header)
    return wcs

def load_opd_map(nrc, date, opd_map_cache, filename_cache):
    """
    Loads the OPD map for a given date and caches it.
    
    Parameters:
        nrc (webbpsf.NIRCam): NIRCam object from WebbPSF.
        date (str): Observation date.
        opd_map_cache (dict): Cache of loaded OPD maps.
        filename_cache (dict): Cache of OPD map filenames.
    """
    if date not in opd_map_cache:
        file_name = filename_cache[date]
        nrc.load_wss_opd(file_name, plot=False, verbose=False)
        opd_map_cache[date] = nrc.pupilopd
    nrc.pupilopd = opd_map_cache[date]

def preload_opd_maps(wcs_metadata, rank):
    """
    Preloads OPD maps for all unique observation dates in the WCS metadata.
    
    Parameters:
        wcs_metadata (dict): Dictionary containing WCS metadata for each exposure.
        rank (int): Rank of the MPI process.
    
    Returns:
        dict: A cache of filenames for OPD maps.
    """
    opd_map_cache = {}
    filename_cache = {}
    if rank == 0:
        print('\nPreloading OPD maps.')
        dates = set()
        for metadata in wcs_metadata.values():
            dates.add(metadata['DATE-OBS'])
        for date in dates:
            print(f'[{current_time_string()}] Caching OPD map: {date}')
            nrc = webbpsf.NIRCam()
            nrc.load_wss_opd_by_date(date, plot=False, verbose=False, choice='closest') # Load the OPD map based on the date

            # Open the OPD based on whether it is an HDU list or a path to an OPD map.
            if isinstance(nrc.pupilopd, fits.HDUList):
                hdu_list = nrc.pupilopd
            elif isinstance(nrc.pupilopd, str):
                hdu_list = fits.open(nrc.pupilopd)
            else:
                raise ValueError("Unexpected type for nrc.pupilopd")
            
            # Access information about the OPD map from the primary header.
            header = hdu_list[0].header
            corr_id = header['corr_id']
            apername = header['apername']
            home_dir = os.getenv('HOME')
            opd_map_dir = 'webbpsf-data/MAST_JWST_WSS_OPDs'

            # Cache OPD maps based on the header information, given that this header information is included in the filename.
            pattern = f'{home_dir}/{opd_map_dir}/{corr_id}-{apername}*.fits'
            matching_files = glob.glob(pattern)
            if matching_files:
                full_file_path = matching_files[0]
                file_name = os.path.basename(full_file_path)
                print(f'[{current_time_string()}] OPD map [{file_name}] cached.\n')
                nrc.load_wss_opd(file_name, plot=False, verbose=False)
                opd_map_cache[date] = nrc.pupilopd
                filename_cache[date] = full_file_path
        print(f'[{current_time_string()}] All OPD maps loaded.')

    return filename_cache # Return the filename cache instead of a cache of OPD maps, because the list of OPD objects cannot be broadcast using MPI. 

def simulate_psf(mosaic_coord, exp_cal_coords_dict, fov, opd_map_cache, filename_cache, sigma, pixel_scale, wcs_metadata, rank):
    """
    Simulates a Point Spread Function (PSF) for a given mosaic coordinate.
    
    Parameters:
        mosaic_coord (tuple): The mosaic coordinate (x, y).
        exp_cal_coords_dict (dict): Dictionary of exposure calibration coordinates.
        fov (int): Field of view in pixels.
        opd_map_cache (dict): Cache of loaded OPD maps.
        filename_cache (dict): Cache of filenames for OPD maps.
        sigma (float): Standard deviation for Gaussian filter.
        pixel_scale (float): Pixel scale in arcseconds/pixel.
        wcs_metadata (dict): Dictionary containing WCS metadata for each exposure.
        rank (int): Rank of the MPI process.
    
    Returns:
        np.ndarray: The simulated PSF as a 2D numpy array.
    """
    nrc = webbpsf.NIRCam() # Initialize NIRCam object from WebbPSF
    mosaic_coord_tuple = tuple(mosaic_coord)

    contributing_exposures = [] # Empty list to be filled with contributing exposures.
    for exp, coords_list in exp_cal_coords_dict.items():
        if mosaic_coord_tuple in [(c['x_mosaic'], c['y_mosaic']) for c in coords_list]:
            contributing_exposures.append(exp)

    print(f'[{current_time_string()} - rank {rank}] Coordinate {mosaic_coord} - number of contributing exposures: {len(contributing_exposures)}')
    
    # For each contributing exposure, simulate a PSF at the correct coordinate.
    for j, exp in enumerate(contributing_exposures):
        
        combined_psf = np.zeros((31,31)) # Initial PSF stamp
        total_exposure_time = 0.0 # Initial exposure time
        wcs_data = wcs_metadata[exp] # WCS (and other relevant) information

        coord_info = next(c for c in exp_cal_coords_dict[exp] if (c['x_mosaic'], c['y_mosaic']) == mosaic_coord_tuple)
        
        # Check for negative coordinates, as occasionally there can be an error in the conversion between mosaic and exposure coordinates.
        # It is better to just skip these cases, since they are exceedingly rare. The exact cause of these negative coordinates is undetermined. 
        if coord_info['x_cal'] < 0 or coord_info['y_cal'] < 0:
            print(f"Skipping coordinate with negative values: x_cal={coord_info['x_cal']}, y_cal={coord_info['y_cal']}")
            continue

        try:
            nrc.detector_position = (coord_info['x_cal'], coord_info['y_cal'])
        except ValueError as e:
            print(f"Error setting detector position with coordinates: x_cal={coord_info['x_cal']}, y_cal={coord_info['y_cal']}")
            print(f"Error message: {e}")
            continue

        exposure_time = wcs_data['XPOSURE'] # Exposure time
        roll_ref = wcs_data['ROLL_REF'] # Roll angle of the detector
        obs_filter = wcs_data['FILTER'] # Filter
        obs_date = wcs_data['DATE-OBS'] # Observation date
        detector = wcs_data['DETECTOR'] # Detector name
        nrc.detector = detector
        nrc.filter = obs_filter
        nrc.pixelscale = pixel_scale

        load_opd_map(nrc, obs_date, opd_map_cache, filename_cache) # Load the necessary pre-downloaded OPD map.
        
        try:
            psf = nrc.calc_psf(oversample=1, fov_pixels=fov+6, normalize='exit_pupil')[0].data # Calculate PSF
        except Exception as e:
            print(f"Error during PSF calculation for exposure {exp} at {coord_info['x_cal']}, {coord_info['y_cal']}: {e}")
            continue

        rotated_psf = rotate(psf, -1*roll_ref, reshape=False, order=3, mode='constant') # Rotate the PSF according to the roll angle.
        psf_cut = rotated_psf[3:-3, 3:-3] # Cut out the desired stamp size to exclude any edge effects caused by the rotation.

        weighted_psf = psf_cut * exposure_time # Weight the resulting PSF based on the exposure time.
        combined_psf += weighted_psf
        total_exposure_time += exposure_time

    if len(contributing_exposures) > 0:
        combined_psf /= total_exposure_time # Divide by the total exposure time to produce the final PSF.
        blurred_psf = gaussian_filter(combined_psf, sigma=sigma) # Blur the PSF to account for differences between simulation and observation caused by
                                                                # detector effects that are not simulated in this case such as charge capacitance,
                                                                # interpixel capacitance, and dithering. There are options to include these effects,
                                                                # but they are simple Gaussian smoothing operations that can be done all at once instead
                                                                # to more closely match the observed PSFs.
        blurred_psf = blurred_psf / np.sum(blurred_psf) # Normalize the final blurred PSF.
        return blurred_psf
    else:
        print(f"[{current_time_string()} - rank {rank}] No valid exposures found, returning zero PSF for mosaic_coord: {mosaic_coord}")
        combined_psf = np.zeros((31,31))
        return combined_psf

def print_title():
    """
    Prints the title banner for the WebbPSF modeling.
    """
    print('\n----------------------------------------------------')
    print('                  WebbPSF Modeling')
    print('----------------------------------------------------')
    print(f'\n[{current_time_string()}] Organizing exposures and transforming coordinates...\n')

def process_image_and_assign_coordinates(mosaic_img_path, catalog_path, x_col, y_col, json_path, wcs_metadata):
    """
    Processes the mosaic image and assigns coordinates to exposures.
    
    Parameters:
        mosaic_img_path (str): Path to the mosaic image file.
        catalog_path (str): Path to the catalog file.
        x_col (str): Column name for x coordinates in the catalog file.
        y_col (str): Column name for y coordinates in the catalog file.
        json_path (str): Path to the JSON file containing association data.
        wcs_metadata (dict): Dictionary containing WCS metadata for each exposure.
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of mosaic coordinates.
            - dict: Dictionary of exposure calibration coordinates.
    """
    # Load mosaic image and context data, which is the third extension of an i2d.fits file from the JWST pipeline.
    with fits.open(mosaic_img_path) as mosaic_img:
        mosaic_context = mosaic_img[3].data
        mosaic_wcs = WCS(mosaic_img[1].header)

    # Loading coordinates from a catalog file
    cat = ascii.read(catalog_path)
    print(f'Number of coordinates to process: {len(cat)}\n')
    x_coords = cat[x_col]
    y_coords = cat[y_col]

    # Compute RA and DEC for all coordinates
    ra_list = []
    dec_list = []
    for x, y in zip(x_coords, y_coords):
        ra, dec = mosaic_wcs.pixel_to_world_values(x, y)
        ra_list.append(ra)
        dec_list.append(dec)

    mosaic_gal_coord = np.vstack((x_coords, y_coords)).T
    ra_dec_coords = np.vstack((ra_list, dec_list)).T

    with open(json_path, 'r') as file:
        asn_data = json.load(file)

    # Get an ordered list of the exposures which contribute to the mosaic.
    exp_filenames = [member['expname'] for product in asn_data['products'] for 
                     member in product['members'] if member['exptype'] == 'science']
    
    mos_coords_dict = {filename: [] for filename in exp_filenames}
    exp_cal_coords_dict = {filename: [] for filename in exp_filenames}

    for coord, (ra, dec) in zip(mosaic_gal_coord, ra_dec_coords):
        x, y = coord
        inputs = decode_context(mosaic_context, [int(np.round(x))], [int(np.round(y))])

        # Flatten to convert from a list of arrays to a list of indices.
        inputs_flat = [item for sublist in inputs for item in sublist]

        # Setup mos_coords_dict which contains information for each coordinate set on which exposures contribute.
        for input_idx in inputs_flat:
            if input_idx < len(exp_filenames):
                mos_coords_dict[exp_filenames[input_idx]].append((ra, dec, x, y))

    for i, exp in enumerate(exp_filenames):
        
        wcs_info = wcs_metadata[exp]

        # Create a new WCS object
        cal_wcs = create_wcs_from_csv(wcs_info)
        cal_wcs.array_shape = (2048, 2048)

        # If there are no contributing exposures, skip.
        if not mos_coords_dict[exp]:
            continue

        for (ra, dec, x_mosaic, y_mosaic) in mos_coords_dict[exp]:
            # Convert sky coordinates (RA, Dec) to cal.fits pixel coordinates
            x_cal, y_cal = cal_wcs.world_to_pixel_values(ra, dec)

            # Round to nearest pixel for use with webbpsf
            x_cal_int, y_cal_int = int(np.round(x_cal)), int(np.round(y_cal))
            ra_int, dec_int = cal_wcs.pixel_to_world_values(x_cal_int, y_cal_int)

            # Store the RA, Dec, and cal.fits coordinates in the new dictionary
            exp_cal_coords_dict[exp].append({'ra': float(ra_int), 'dec': float(dec_int), 'x_cal': float(x_cal), 
                                                'y_cal': float(y_cal), 'x_cal_int': x_cal_int, 'y_cal_int': y_cal_int, 
                                                'x_mosaic': float(x_mosaic), 'y_mosaic': float(y_mosaic)})

    return mosaic_gal_coord, exp_cal_coords_dict