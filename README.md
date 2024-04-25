# Yonsei Observable UNiverse Group (YOUNG) WebbPSF Modeling

A simplified implementation of the WebbPSF simulation, intended to be used for weak gravitational lensing analysis.

## What this implementation does

- Takes a catalog of source positions as input, with this catalog having the SExtractor keys "XWIN_IMAGE" and "YWIN_IMAGE."
    - Note that even though the key format is the same as [*SExtractor*](https://sextractor.readthedocs.io/en/latest/Introduction.html), the catalog should not be a *SExtractor* catalog. Instead, the catalog should be an Astropy table saved in ASCII format.
    - For example, one might open a *SExtractor* catalog with "data = fits.open('/path/to/file.cat')[2].data. The catalog to be used for WebbPSF modeling should be saved as follows:
        - catalog = Table(data[selected_sources]) 
        - catalog.write('final_catalog.cat', format='ascii', overwrite=True) 
        - Here, "selected_sources" would be the indices of sources selected as background sources.
- Produces a PSF at each source position which is the exposure time-weighted combination of PSFs from each relevant input exposure. Proper rotations, pixel scales, and observation dates are taken into account.
    - Note that the final PSFs are smoothed to better match the size of observed stars. Currently, this sigma value should be determined empirically on a case-by-case basis. It should be based on a statistical comparison between the sizes of stars in the field and the WebbPSF simulated PSFs. However, a value of around 0.8 has proven to be applicable in various observations so far. 
- Saves the PSFs in the same order as the input catalog file.
- Utilizes MPI to significantly speed up the simulation process.

## Installation and Requirements

It is necessary to have both the [*JWST pipeline*](https://jwst-pipeline.readthedocs.io/en/latest/) package and the [*WebbPSF*](https://webbpsf.readthedocs.io/en/latest/) package installed when using this code. Additionally, it is important that the user changes the environment variable "CRDS_PATH" to their own crds_cache directory. This can be changed in the imports section of the code. The variable "CRDS_SERVER_URL" should not need to be changed. 

It is also required that the user fills the cal_files directory with the unrectified exposures (cal.fits) used to create the mosaic image for the chosen filter. If you want to create multiple PSF models for different filters, the mosaic image, exposures, and JSON association file should be replaced for each run. There is no option for processing multiple filters at once. 

The *JSON* file **must** be the same one used to create the mosaic image in stage 3 of the JWST pipeline. An example *JSON* file is included. If you did not use the JWST pipeline to create the mosaic image and do not have the proper association, you will need to create one. The file *create_association.py* contains a basic method which can create an association from *cal.fits* files in a given directory. It may need to be altered for the user's specific purposes.

Finally, the user must have *mpi4py* installed to use this code. The following command can be used to install *mpi4py*:
- $ conda install -c conda-forge mpi4py openmpi

## Usage

Once the input files have been prepared, the *img_path*, *catalog_path*, and *json_path* variables must be changed to match the user's setup. The *psf_array_filename* and *sigma* variables can also be changed as needed, where the *sigma* variable determines the Gaussian smoothing kernel sigma to use. The *pixel_scale* variable should be changed to match the pixel scale of the mosaic image. 

Then, the code can be run as follows:
- $ mpiexec -n num_proc python webbpsf_modeling.py

Change *num_proc* to be the desired number of processes to use when running the code. 