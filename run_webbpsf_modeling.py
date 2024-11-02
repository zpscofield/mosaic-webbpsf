# Simulate JWST PSFs for coordinate positions in a mosaic image produced by the JWST pipeline.

__author__ = "Zachary P. Scofield"
__version__ = "1.0.0"
__license__ = "MIT"

"""
Currently, this process only supports mosaic images produced by the JWST pipeline because of how the code handles context images. In the future,
support will be added for users who used a separate reduction pipeline.
"""
import time
import json
from mpi4py import MPI
import numpy as np
from astropy.io import fits
from webbpsf_fns import print_title, load_wcs_metadata, process_image_and_assign_coordinates, preload_opd_maps, current_time_string, simulate_psf

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    start_time = time.time()

    # Load configuration from JSON file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Relevant paths
    img_path = config['img_path'] # Mosaic image
    catalog_path = config['catalog_path'] # Catalog
    json_path = config['json_path'] # JSON association file
    csv_path = config['csv_path'] # Path to the CSV file with WCS and metadata
    x_col = config['x_col'] # Column name for x coordinates in the catalog file.
    y_col = config['y_col'] # Column name for y coordinates in the catalog file.

    # Final model filename
    psf_array_filename = config['psf_array_filename']

    sigma = config['sigma'] # Gaussian smoothing kernel sigma - user specified. A value around 0.79 has shown to be applicable to
                 # multiple observations through a comparison between PSF simulations and observed stars.
    pixel_scale = config['pixel_scale'] # Pixel scale of the mosaic image.
    dimension = config['dimension'] # Final desired dimension of the PSF stamp.

    # For efficiency, only access data on rank 0 (one process), then broadcast the information to other processes.
    if rank == 0:
        print_title()
        wcs_metadata = load_wcs_metadata(csv_path)
        mosaic_gal_coord, exp_cal_coords_dict = process_image_and_assign_coordinates(img_path, catalog_path, x_col, y_col, json_path, wcs_metadata)
    else:
        mosaic_gal_coord = None
        exp_cal_coords_dict = None
        wcs_metadata = None

    # Broadcast necessary information from rank 0 to other ranks.
    mosaic_gal_coord = comm.bcast(mosaic_gal_coord, root=0)
    exp_cal_coords_dict = comm.bcast(exp_cal_coords_dict, root=0)
    wcs_metadata = comm.bcast(wcs_metadata, root=0)
    if rank == 0:
        filename_cache = preload_opd_maps(wcs_metadata, rank)
    filename_cache = comm.bcast(filename_cache, root=0)
    comm.Barrier()  # Ensure all ranks wait until OPD maps are preloaded
    opd_map_cache = {}

    if rank == 0:
        print(f'[{current_time_string()}] Beginning PSF modeling process...\n')

    # This section computes the PSFs for each coordinate.
    psfs = []
    coord_idx = rank
    while coord_idx < len(mosaic_gal_coord):
        mosaic_coord = mosaic_gal_coord[coord_idx]
        psf = simulate_psf(mosaic_coord, exp_cal_coords_dict, fov=dimension, opd_map_cache=opd_map_cache, filename_cache=filename_cache, sigma=sigma, pixel_scale=pixel_scale, wcs_metadata=wcs_metadata, rank=rank)
        psfs.append((coord_idx, psf))  # Include the index with the PSF
        print(f'[{current_time_string()} - rank {rank}] PSF completed for coordinate {mosaic_coord}')
        coord_idx += size
    psfs = np.array(psfs, dtype=object)
    comm.Barrier()
    all_psf_results = comm.gather(psfs, root=0) # Gather the simulations from different processes.
    
    # Concatenate the PSF results from each process and save the PSF model.
    if rank == 0:
        final_psfs = sorted(np.concatenate(all_psf_results, axis=0), key=lambda x: x[0])
        final_psfs = np.array([psf for _, psf in final_psfs])
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
