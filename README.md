# Yonsei Observable UNiverse Group (YOUNG) STPSF Modeling

A streamlined implementation of the STPSF simulation for JWST mosaic images.

Note: This project is not affiliated with the original authors of STPSF.

## What this WebbPSF implementation does

- Takes a catalog of source positions as input.
    - Note that the catalog should be an Astropy table saved in ASCII format.
    - For example, one might open a *SExtractor* catalog with "data = fits.open('/path/to/file.cat')[2].data. The catalog to be used for WebbPSF modeling should be saved as follows:
        - catalog = Table(data[selected_sources]) 
        - catalog.write('final_catalog.cat', format='ascii', overwrite=True) 
        - Here, "selected_sources" would be the indices of sources selected as background sources.
- Produces a PSF at each source position which is the exposure time-weighted combination of PSFs from each relevant input exposure. Proper rotations, pixel scales, and observation dates are taken into account.
    - Note that the final PSFs are smoothed to better match the size of observed stars. Currently, this sigma value should be determined empirically on a case-by-case basis. It should be based on a statistical comparison between the sizes of stars in the field and the WebbPSF simulated PSFs. However, a value of around 0.8 has proven to be applicable in various observations so far. 
- Saves the PSFs in the same order as the input catalog file.
- Utilizes MPI to significantly speed up the simulation process.
- Uses an Optical Path Difference (OPD) map caching process to prevent redundant API calls across multiple MPI processes. If multiple processes attempt to download the same OPD map simultaneously, the modeling process will fail. Even if the relevant OPD map is already downloaded onto the user's computer, the *load_wss_opd_by_date()* method will still call the API. This can significantly slow down the modeling process or cause it to fail when using a considerable number of processes. The *load_opd_map()* method addresses these issues by ensuring that each OPD map is preloaded only once by the root process (rank 0), and then the relevant information is shared to the other processes. Caching the OPD maps in this way prevents any data access conflicts and significantly reduces the execution time.

## Installation and Requirements

It is necessary to have both the [*JWST pipeline*](https://jwst-pipeline.readthedocs.io/en/latest/) package and the [*WebbPSF*](https://webbpsf.readthedocs.io/en/latest/) package installed when using this code.

It is also required that the user has a CSV file with various important WCS information and other metadata from each contributing exposure. The script *log_wcs.py* can create this CSV file given a directory with *cal.fits* files. If you want to create multiple PSF models for different filters, the mosaic image, exposures, JSON association file, and CSV file should be replaced for each run. There is no option for processing multiple filters at once. 

The *JSON* association file **must** be the same one used to create the mosaic image in stage 3 of the JWST pipeline. An example *JSON* file is included. If you did not use the JWST pipeline to create the mosaic image and do not have the proper association, you will need to create one. The file *create_association.py* contains a basic method which can create an association from *cal.fits* files in a given directory. It may need to be altered for the user's specific purposes.

Finally, the user must have *mpi4py* installed to utilize parallel processing with this code. The following command can be used to install *mpi4py*:
- $ conda install -c conda-forge mpi4py openmpi

## Usage

Before running *run_webbpsf_modeling.py*, the *config.json* file should be updated with the necessary information. This includes the image path, catalog path, association path, and the CSV exposure information path. Additionally, The *psf_array_filename* and *sigma* variables can also be changed as needed, where the *sigma* variable determines the Gaussian smoothing kernel sigma to use. The *pixel_scale* variable should be changed to match the pixel scale of the mosaic image, and the *dimension* variable should be changed to whatever the desired stamp size is for the final PSFs. Finally, the user should set their *JWST* and *WebbPSF* environment variables within the *config.json* file.

Then, the code can be run as follows:
- $ mpiexec -n num_proc python run_webbpsf_modeling.py

Change *num_proc* to be the desired number of processes to use when running the code. 
