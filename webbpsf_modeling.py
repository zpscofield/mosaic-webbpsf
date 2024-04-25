

import numpy as np
import copy as cp
import os
os.environ['CRDS_PATH'] = '/path/to/crds_cache' # Replace with directory of your own crds_cache folder
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

# This method allows for the caching of OPD maps so that the process does not need to repeatedly call the API.
def load_opd_map(nrc, date, opd_map_cache):
    
    file_name = None
    if date not in opd_map_cache:
        # It is necessary to load new OPD maps on rank 0 initially so that a download error doesn't occur. The comm.Barrier() 
        # call synchronizes processes.
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

# Function to simulate PSF for a given coordinate
def simulate_psf(mosaic_coord, exp_cal_coords_dict, fov, opd_map_cache, sigma, pixel_scale):

    # Initialize a NIRCam object.
    nrc = webbpsf.NIRCam()
    mosaic_coord_tuple = tuple(mosaic_coord)

    # Determine the exposures which contribute to this mosaic coordinate.
    contributing_exposures = []
    for exp, coords_list in exp_cal_coords_dict.items():
        if mosaic_coord_tuple in [(c['x_mosaic'], c['y_mosaic']) for c in coords_list]:
            contributing_exposures.append(exp)

    # Show the number of contributing exposures for each coordinate
    print(f'[{current_time_string()}] Coordinate {mosaic_coord} - number of contributing exposures: {len(contributing_exposures)}')
    
    # Process each contributing exposure
    for j, exp in enumerate(contributing_exposures):
        
        combined_psf = np.zeros((31,31))
        total_exposure_time = 0.0

        with ImageModel(f'./cal_files/{exp}') as cal_image_model: # Initializing the exposure as an image model.

            data_copy = cp.deepcopy(cal_image_model.data)

            # Access the necessary metadata from the exposure.
            exposure_time = cal_image_model.meta.exposure.exposure_time
            roll_ref = cal_image_model.meta.wcsinfo.roll_ref
            obs_filter = cal_image_model.meta.instrument.filter
            obs_date = cal_image_model.meta.observation.date
            detector = cal_image_model.meta.instrument.detector
            nrc.detector = detector
            nrc.filter = obs_filter
            nrc.pixelscale = pixel_scale # Pixel scale of the mosaic image. The WebbPSF simulation oversamples the PSF
                                         # to match the desired pixel scale of the final PSF.

            # Loads the OPD map, either through an API call or through the cache.
            load_opd_map(nrc, obs_date, opd_map_cache)

            # This section searches within the exp_cal_coords_dict dictionary to find the entry for this 
            # specific exposure which matches the current mosaic coordinate.
            index = next(i for i, c in enumerate(exp_cal_coords_dict[exp]) 
                         if (c['x_mosaic'], c['y_mosaic']) == mosaic_coord_tuple)
            coord_info = exp_cal_coords_dict[exp][index]
            
            nrc.detector_position = (coord_info['x_cal'], coord_info['y_cal']) # Setting simulation position.
            #nrc.options['charge_diffusion_sigma'] = 0.012

            psf = nrc.calc_psf(oversample=1, fov_pixels=fov+6, normalize='exit_pupil')[0].data # PSF calculation
            
            # Note for the PSF rotation, the only in-simulation rotation option is the "pupil rotation," which
            # "only has an effect for optical models that have something at an intermediate pupil plane between 
            # the telescope aperture and the detector." Therefore, I use my own rotation algorithm.
            rotated_psf = rotate(psf, -1*roll_ref, reshape=False, order=3, mode='constant') # PSF rotation
            psf_cut = rotated_psf[3:-3, 3:-3] # Exclude the edge since the simulation was made to be larger to
                                              # account for any edge effects.

            weighted_psf = psf_cut * exposure_time
            combined_psf += weighted_psf
            total_exposure_time += exposure_time

    combined_psf/=total_exposure_time # This final (pre-smoothed) PSF is the combined PSF weighted based on exposure 
                                      # time.
    blurred_psf = gaussian_filter(combined_psf, sigma=sigma) # Smoothing is necessary because WebbPSF simulations are 
                                                             # systematically smaller than observed stars. The 
                                                             # default charge diffusion and jitter options do not 
                                                             # adequately smooth the simulated PSFs enough so that
                                                             # the sizes match observed stars, so this code simply
                                                             # smooths the PSFs with a Gaussian filter before the
                                                             # final normalization. The same result can be achieved
                                                             # by fine-tuning the simulation options, but the current
                                                             # method is simpler. The sigma value should be customized 
                                                             # by the user to best matched observed star sizes.

    blurred_psf = blurred_psf/np.sum(blurred_psf) # Final normalization.
    
    return blurred_psf

# Printing method
def print_title():
    print('\n----------------------------------------------------')
    print('                  WebbPSF Modeling')
    print('----------------------------------------------------')
    print(f'\n[{current_time_string()}] Organizing exposures and transforming coordinates...\n')
    
# Method which retrieves relevant data from the input FITS file as well as the source catalog information
def image_attributes(mosaic_img_path, catalog_path):
    
    # Load mosaic image and context data, which is the third extension of an i2d.fits file from the JWST pipeline.
    with fits.open(mosaic_img_path) as mosaic_img:
        mosaic_context = mosaic_img[3].data
        mosaic_wcs = WCS(mosaic_img[1].header) # WCS information is contained in the second header for JWST data.

    # Loading coordinates from a catalog file
    cat = ascii.read(catalog_path)
    print(f'Number of coordinates to process: {len(cat)}\n')
    x_coords = cat['XWIN_IMAGE']-1
    y_coords = cat['YWIN_IMAGE']-1

    # Combine the selected coordinates into an array
    mosaic_gal_coord = np.vstack((x_coords, y_coords)).T # Usually selected_ but I want to use all stars for SMACS.

    return mosaic_context, mosaic_wcs, mosaic_gal_coord

def assign_coordinates(mosaic_gal_coord, mosaic_context, mosaic_wcs, json_path):

    # Load .json association file. This can either be the default association file downloaded from the archive, or
    # it can be a custom association file. It just needs to match the input image you are using and correspond to
    # how this image was created.
    with open(json_path, 'r') as file:
        asn_data = json.load(file)

    # Get an ordered list of the exposures which contribute to the mosaic.
    exp_filenames = [member['expname'] for product in asn_data['products'] for 
                     member in product['members'] if member['exptype'] == 'science']

    # The exposure_coords_dict will hold indices of input exposures which correspond to a given mosaic coordinate.
    # The dictionary exp_cal_coords_dict will hold all relevant coordinate information for each exposure.
    exp_coords_dict = {filename: [] for filename in exp_filenames}
    exp_cal_coords_dict = {filename: [] for filename in exp_filenames}

    # Loop for determining contributing exposures.
    for i, coord in enumerate(mosaic_gal_coord):
        x, y = coord

        # The JWST pipeline convenience function decode_context allows us to determine which input exposures 
        # contribute to any given pixel (integer). 
        inputs = decode_context(mosaic_context, [int(np.round(x))], [int(np.round(y))])

        # Flatten to convert from a list of arrays to a list of indices.
        inputs_flat = [item for sublist in inputs for item in sublist]

        # Setup exp_coords_dict - contains information for each coordinate set on which exposures contribute. 
        for input_idx in inputs_flat:
            if input_idx < len(exp_filenames):
                exp_coords_dict[exp_filenames[input_idx]].append((x, y))

    # This loop produces all necessary coordinates, including RA, DEC, x, and y for the mosaic and exposures.
    for i, exp in enumerate(exp_filenames):

        with fits.open('./cal_files/' + exp) as hdul:
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
                exp_cal_coords_dict[exp].append({'ra': float(ra_int), 'dec': float(dec_int), 'x_cal': float(x_cal), 
                                                 'y_cal': float(y_cal), 'x_cal_int': x_cal_int, 'y_cal_int': y_cal_int, 
                                                 'x_mosaic': x_mosaic, 'y_mosaic': y_mosaic})

    # The exp_coords_dict dictionary describes which exposures correspond to each source position, where the
    # source positions are the keys. If there is a source centered (100,100), then the dictionary tells us which 
    # exposures contribute to that pixel. 
    #
    # The exp_cal_coords_dict dictionary has each exposure as a key. For each of these exposures, the dictionary
    # contains various sets of coordinates. This includes RA and DEC, calibrated exposure coordinates (float and 
    # rounded), and mosaic coordinates. Each set corresponds to a different mosaic coordinate. For example, one
    # exposure may contribute to 30 mosaic coordinates. This dictionary then has 30 entries, one for each mosaic
    # coordinate. Each entry has the aforementioned information. 
    return exp_cal_coords_dict

# This method splits the mosaic coordinates into subgroups to be used with MPI.
def split_coords(mosaic_gal_coord):

    num_pairs = len(mosaic_gal_coord)
    pairs_per_process = num_pairs // size
    start = rank * pairs_per_process
    end = start + pairs_per_process if rank != size - 1 else num_pairs
    coords = mosaic_gal_coord[start:end, :] # Switched to be slicing through different rows instead of columns based on array 
                                            # setup. May need to be changed depending on whether transposition (.T) is used. 
    
    return coords

# Main function for PSF calculation
def main():
    
    start_time = time.time()

    # Relevant paths
    img_path = '/path/to/image' # Mosaic image
    catalog_path = 'path/to/catalog' # Catalog
    json_path = 'path/to/json' # JSON association file (see "assign_coordinates" method for more information
                               # regarding the requirements for the association file).

    # Final model filename.
    psf_array_filename = './WebbPSF_model.fits'

    # Gaussian smoothing kernel sigma - user specified. A value around 0.8 has proven to be applicable to
    # multiple observations. 
    sigma = 0.79
    pixel_scale = 0.02

    # For efficiency, only access data on rank 0 (one process), then broadcast the information to other processes.
    if rank == 0:
        print_title()
        mosaic_context, mosaic_wcs, mosaic_gal_coord = image_attributes(img_path, catalog_path)
        exp_cal_coords_dict = assign_coordinates(mosaic_gal_coord, mosaic_context, mosaic_wcs, json_path)
    else:
        mosaic_gal_coord = None
        exp_cal_coords_dict = None

    # Broadcast necessary information from rank 0 to other ranks.
    mosaic_gal_coord = comm.bcast(mosaic_gal_coord, root=0)
    exp_cal_coords_dict = comm.bcast(exp_cal_coords_dict, root=0)

    # Initialize the OPD map cache for each process.
    opd_map_cache = {}

    if rank == 0:
        print(f'[{current_time_string()}] Beginning PSF modeling process...\n')

    coords = split_coords(mosaic_gal_coord) # Split coordinates
    dimension = 31 # PSF stamp dimension
    psfs = np.zeros((len(coords), dimension, dimension)) # PSF array for each process, it should be a 
                                                         # 3D array with the axis 0 dimension corresponding 
                                                         # to the size of the subgroup.

    # For each subgroup, run the simulation.
    for i, mosaic_coord in enumerate(coords):
        psfs[i,:,:] = simulate_psf(mosaic_coord=mosaic_coord, exp_cal_coords_dict=exp_cal_coords_dict, fov=dimension, 
                                   opd_map_cache=opd_map_cache, sigma=sigma, pixel_scale=pixel_scale)
        print(f'[{current_time_string()}] PSF completed for coordinate {mosaic_coord}')

    all_psf_results = comm.gather(psfs, root=0) # Gather the simulations from different processes.
    
    # Concatenate the PSF results from each process and save the PSF model.
    if rank == 0:
        final_psfs = np.concatenate(all_psf_results, axis=0)
        fits.writeto(psf_array_filename, final_psfs, overwrite=True)

    # Check the execution time. 
    end_time = time.time()
    total_time = end_time - start_time
    if rank == 0:
        print(f"Total execution time: {total_time:.2f} seconds")

    # Terminate the MPI environment. 
    MPI.Finalize()

if __name__ == '__main__':
    main()