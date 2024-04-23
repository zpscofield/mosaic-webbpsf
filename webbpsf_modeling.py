

import numpy as np
import copy as cp
import os
os.environ['CRDS_PATH'] = '/Users/zpscofield/crds_cache' # Replace with directory of your own crds_cache file
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
import glob
import json
from mpi4py import MPI
#import openmp
import time
import datetime
import webbpsf
from astropy.io import fits, ascii
from astropy.wcs import WCS
from scipy.ndimage import rotate, gaussian_filter
from jwst.datamodels import ImageModel
from jwst.resample.resample_utils import decode_context

# Note: installing mpi4py: 
# $ conda install -c conda-forge mpi4py openmpi
# mpiexec -n 4 python webbpsf_modeling_final.py

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def current_time_string():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")

def load_opd_map_if_needed(nrc, date, opd_map_cache):
    
    file_name = None
    if date not in opd_map_cache:
        # It is necessary to load new OPD maps on rank 0 initially so that a download error doesn't occur. The comm.Barrier() call synchronizes processes.
        if rank == 0:
            nrc.load_wss_opd_by_date(date, plot=False, verbose=False, choice='before')
            if isinstance(nrc.pupilopd, fits.HDUList):
                hdu_list = nrc.pupilopd
            elif isinstance(nrc.pupilopd, str):
                hdu_list = fits.open(nrc.pupilopd)
            else:
                raise ValueError("Unexpected type for nrc.pupilopd")

            # Access the primary HDU's header
            header = hdu_list[0].header
            corr_id = header['corr_id']
            apername = header['apername']

            # Construct the file name
            home_dir = os.getenv("HOME")
            opd_map_dir = "webbpsf-data/MAST_JWST_WSS_OPDs" 
            pattern = f"{home_dir}/{opd_map_dir}/{corr_id}-{apername}*.fits"
            matching_files = glob.glob(pattern)
            if matching_files:
                full_file_path = matching_files[0]
                file_name = os.path.basename(full_file_path)
            print(f'[{current_time_string()}] OPD map [{file_name}] cached.\n')
        comm.Barrier()
        file_name = comm.bcast(file_name if rank == 0 else None, root=0)
        nrc.load_wss_opd(file_name, plot=False, verbose=False)
        opd_map_cache[date] = nrc.pupilopd

    # If the opd map has already been loaded, it is taken from the OPD map cache. 
    else:
        nrc.pupilopd = opd_map_cache[date]

# Function to simulate PSF for a given star
def simulate_psf(mosaic_coord, exp_cal_coords_dict, fov, opd_map_cache):

    nrc = webbpsf.NIRCam()
    mosaic_coord_tuple = tuple(mosaic_coord)

    # Determine the exposures which contribute to this mosaic coordinate. 

    contributing_exposures = []
    for exp, coords_list in exp_cal_coords_dict.items():
        if mosaic_coord_tuple in [(c['x_mosaic'], c['y_mosaic']) for c in coords_list]:
            contributing_exposures.append(exp)


    # Shows the number of contributing exposures for each coordinate.
    print(f'[{current_time_string()}] Coordinate {mosaic_coord} - number of contributing exposures: {len(contributing_exposures)}')
    
    # Process each contributing exposure
    for j, exp in enumerate(contributing_exposures):

        # Determine the correct detector based on the filename

        # Short wavelength
        # if 'nrca1' in exp:
        #     nrc.detector = 'NRCA1'
        # elif 'nrcb1' in exp:
        #     nrc.detector = 'NRCB1'
        # elif 'nrca2' in exp:
        #     nrc.detector = 'NRCA2'
        # elif 'nrcb2' in exp:
        #     nrc.detector = 'NRCB2'
        # elif 'nrca3' in exp:
        #     nrc.detector = 'NRCA3'
        # elif 'nrcb3' in exp:
        #     nrc.detector = 'NRCB3'
        # elif 'nrca4' in exp:
        #     nrc.detector = 'NRCA4'
        # elif 'nrcb4' in exp:
        #     nrc.detector = 'NRCB4'
        
        # # Long wavelength
        # elif 'nrcalong' in exp:
        #     nrc.detector = 'NRCA5'
        # elif 'nrcblong' in exp:
        #     nrc.detector = 'NRCB5'
        # else:

        #     # Continue if detector is not recognized
        #     continue
        
        combined_psf = np.zeros((31,31))
        total_exposure_time = 0.0

        # Initializing the exposure as an image model in order to perform resampling.
        with ImageModel(f'./f200_cal/{exp}') as cal_image_model:

            data_copy = cp.deepcopy(cal_image_model.data)

            # Determining the filter and observation date of the exposure.
            exposure_time = cal_image_model.meta.exposure.exposure_time
            roll_ref = cal_image_model.meta.wcsinfo.roll_ref
            obs_filter = cal_image_model.meta.instrument.filter
            obs_date = cal_image_model.meta.observation.date
            detector = cal_image_model.meta.instrument.detector
            nrc.detector = detector
            nrc.filter = obs_filter
            nrc.pixelscale = 0.02

            load_opd_map_if_needed(nrc, obs_date, opd_map_cache)

            index = next(i for i, c in enumerate(exp_cal_coords_dict[exp]) 
                         if (c['x_mosaic'], c['y_mosaic']) == mosaic_coord_tuple)
            coord_info = exp_cal_coords_dict[exp][index]
            
            nrc.detector_position = (coord_info['x_cal'], coord_info['y_cal'])
            #nrc.options['charge_diffusion_sigma'] = 0.012

            psf = nrc.calc_psf(oversample=1, fov_pixels=fov+6, normalize='exit_pupil')[0].data
            
            rotated_psf = rotate(psf, -1*roll_ref, reshape=False, order=3, mode='constant')
            psf_cut = rotated_psf[3:-3, 3:-3]

            weighted_psf = psf_cut * exposure_time
            combined_psf += weighted_psf
            total_exposure_time += exposure_time

    combined_psf/=total_exposure_time
    blurred_psf = gaussian_filter(combined_psf, sigma=0.79)
    blurred_psf = blurred_psf/np.sum(blurred_psf)
    
    return blurred_psf

def print_title():
    print('\n----------------------------------------------------')
    print('                  WebbPSF Modeling')
    print('----------------------------------------------------')
    print(f'\n[{current_time_string()}] Organizing exposures and transforming coordinates...\n')
    
def image_attributes(mosaic_img_path):
     # Load mosaic image and context data, which is the third extension of an i2d.fits file from the JWST pipeline.
    with fits.open(mosaic_img_path) as mosaic_img:
        mosaic_context = mosaic_img[3].data
        mosaic_wcs = WCS(mosaic_img[1].header)

        # Gather WCS header information to use for resampling
        naxis1 = mosaic_img[1].header.get('NAXIS1', 'Unknown')
        naxis2 = mosaic_img[1].header.get('NAXIS2', 'Unknown')
        crpix1 = mosaic_img[1].header.get('CRPIX1', 'Unknown')
        crpix2 = mosaic_img[1].header.get('CRPIX2', 'Unknown')
        crval1 = mosaic_img[1].header.get('CRVAL1', 'Unknown')
        crval2 = mosaic_img[1].header.get('CRVAL2', 'Unknown')

    image_dimensions = [int(naxis1), int(naxis2)]
    crpix = [float(crpix1), float(crpix2)]
    crval = [float(crval1), float(crval2)]

    # Loading coordinates from a catalog file
    cat = ascii.read('/Users/zpscofield/2024_files/whl-2024_04/PSF_modeling/WEBBPSF/f200w_selected.cat')
    print(f'Number of coordinates to process: {len(cat)}\n')
    x_coords = cat['XWIN_IMAGE']-1
    y_coords = cat['YWIN_IMAGE']-1

    # Combine the selected coordinates into an array
    mosaic_gal_coord = np.vstack((x_coords, y_coords)).T # Usually selected_ but I want to use all stars for SMACS.

    return mosaic_context, mosaic_wcs, image_dimensions, crpix, crval, mosaic_gal_coord

def assign_coordinates(mosaic_gal_coord, mosaic_context, mosaic_wcs):
    # Load .json association file produced by the JWST pipeline.
    with open('/Users/zpscofield/2024_files/whl-2024_04/PSF_modeling/WEBBPSF/F200W_asn.json', 'r') as file:
        asn_data = json.load(file)

    # Get an ordered list of the exposures which contribute to the mosaic.
    exp_filenames = [member['expname'] for product in asn_data['products'] for member in product['members'] if member['exptype'] == 'science']

    # The exposure_coords_dict will hold indices of input exposures which correspond to a given mosaic coordinate.
    # The dictionary exp_cal_coords_dict will hold all relevant coordinate information for each exposure.
    exp_coords_dict = {filename: [] for filename in exp_filenames}
    exp_cal_coords_dict = {filename: [] for filename in exp_filenames}

    # Loop for determining contributing exposures.
    for i, coord in enumerate(mosaic_gal_coord):
        x, y = coord

        # The function decode_context allows us to determine which input exposures contribute to any given pixel (int)
        inputs = decode_context(mosaic_context, [int(np.round(x))], [int(np.round(y))])

        # Flatten to convert from a list of arrays to a list of indices.
        inputs_flat = [item for sublist in inputs for item in sublist]

        # Setup exp_coords_dict - contains information for each coordinate set on which exposures contribute. 
        for input_idx in inputs_flat:
            if input_idx < len(exp_filenames):
                exp_coords_dict[exp_filenames[input_idx]].append((x, y))

    # This loop produces all necessary coordinates, including RA, DEC, x, and y for the mosaic and exposures.
    for i, exp in enumerate(exp_filenames):

        with fits.open('./f200_cal/' + exp) as hdul:
            cal_wcs = WCS(hdul[1].header)

            # If there are no contributing exposures, skip.
            if not exp_coords_dict[exp]:
                continue

            for (x_mosaic, y_mosaic) in exp_coords_dict[exp]:
                # Convert mosaic pixel coordinates to sky coordinates (RA, Dec)
                ra, dec = mosaic_wcs.pixel_to_world_values(x_mosaic, y_mosaic)
                
                # Convert sky coordinates (RA, Dec) to cal.fits pixel coordinates
                x_cal, y_cal = cal_wcs.world_to_pixel_values(ra, dec)
                
                # Round to nearest pixel for use with webbpsf
                x_cal_int, y_cal_int = int(np.round(x_cal)), int(np.round(y_cal))
                ra_int, dec_int = cal_wcs.pixel_to_world_values(x_cal_int, y_cal_int)
                
                # Store the RA, Dec, and cal.fits coordinates in the new dictionary
                exp_cal_coords_dict[exp].append({'ra': float(ra_int), 'dec': float(dec_int), 'x_cal': float(x_cal), 'y_cal': float(y_cal), 'x_cal_int': x_cal_int, 'y_cal_int': y_cal_int, 'x_mosaic': x_mosaic, 'y_mosaic': y_mosaic})

    return exp_cal_coords_dict

def split_coords(mosaic_gal_coord):
    # This section splits the coordinates into subgroups to be used with WebbPSF.
    num_pairs = len(mosaic_gal_coord)
    pairs_per_process = num_pairs // size
    start = rank * pairs_per_process
    end = start + pairs_per_process if rank != size - 1 else num_pairs
    coords = mosaic_gal_coord[start:end, :] # Switched to be slicing through different rows instead of columns based on array setup. May need to be changed depending on
                                            # whether .T is used. 
    
    return coords
# Main function for PSF calculation
def main():
    
    start_time = time.time()

    if rank == 0:
        print_title()
        mosaic_context, mosaic_wcs, image_dimensions, crpix, crval, mosaic_gal_coord = image_attributes('/Users/zpscofield/2024_files/whl-2024_04/WHL_nircam_clear-F200W_i2d.fits')
        exp_cal_coords_dict = assign_coordinates(mosaic_gal_coord, mosaic_context, mosaic_wcs)
    else:
        mosaic_gal_coord = None
        exp_cal_coords_dict = None
        image_dimensions = None
        crpix = None
        crval = None

    # Broadcast necessary information from rank 0 to other ranks.
    mosaic_gal_coord = comm.bcast(mosaic_gal_coord, root=0)
    exp_cal_coords_dict = comm.bcast(exp_cal_coords_dict, root=0)
    image_dimensions = comm.bcast(image_dimensions, root=0)
    crpix = comm.bcast(crpix, root=0)
    crval = comm.bcast(crval, root=0)

    opd_map_cache = {}

    if rank == 0:
        print(f'[{current_time_string()}] Beginning PSF modeling process...\n')

    coords = split_coords(mosaic_gal_coord)
    dimension = 31

        # PSF array for each process, it should be a 3D array with axis 0 dimension corresponding to the size of the subgroup.
    psfs = np.zeros((len(coords), dimension, dimension)) 

    for i, mosaic_coord in enumerate(coords):

        psfs[i,:,:] = simulate_psf(mosaic_coord=mosaic_coord, exp_cal_coords_dict=exp_cal_coords_dict, fov=dimension, opd_map_cache=opd_map_cache)
        print(f'[{current_time_string()}] PSF completed for coordinate {mosaic_coord}')

    all_psf_results = comm.gather(psfs, root=0)
    
    if rank == 0:
        final_psfs = np.concatenate(all_psf_results, axis=0)
        psf_array_filename = './WHL_WebbPSF_model.fits'
        fits.writeto(psf_array_filename, final_psfs, overwrite=True)

    # Check the execution time. 
    end_time = time.time()
    total_time = end_time - start_time
    if rank == 0:
        print(f"Total execution time: {total_time:.2f} seconds")

    # Terminate the MPI environment. 
    MPI.Finalize()


# Run the main function
if __name__ == '__main__':
    main()