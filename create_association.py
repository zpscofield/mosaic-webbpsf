import os
import json

def create_custom_association(exp_dir, output_filename, program, target, instrument, filter, pupil="clear", subarray="full", exp_type="nrc_image"):
    """
    Creates a single custom association file with specified metadata, including all exposures in the directory.

    :param directory: The directory containing the FITS files.
    :param output_filename: The path to save the association JSON file.
    :param program: The program ID.
    :param target: The target ID.
    :param instrument: The instrument used.
    :param filter: The filter used.
    :param pupil: The pupil setting (default "clear").
    :param subarray: The subarray setting (default "full").
    :param exp_type: The exposure type (default "nrc_image").
    """
    members = []

    for file in os.listdir(exp_dir):
        if file.endswith('.fits'):
            # Assuming all files should be included as science exposures
            members.append({
                "expname": file,
                "exptype": "science",
                "exposerr": None,
                "asn_candidate": "(custom, observation)"
            })

    # Construct the detailed association structure
    association = {
        "asn_type": "image3",
        "asn_rule": "candidate_Asn_Lv3Image",
        "version_id": None,
        "code_version": "1.9.6",
        "degraded_status": "No known degraded exposures in association.",
        "program": program,
        "constraints": f"DMSAttrConstraint('{{'name': 'program', 'sources': ['program'], 'value': '{program[1:]}'}})\n"
                       f"DMSAttrConstraint('{{'name': 'instrument', 'sources': ['instrume'], 'value': '{instrument}'}})\n"
                       f"DMSAttrConstraint('{{'name': 'opt_elem', 'sources': ['filter'], 'value': '{filter}'}})\n"
                       f"DMSAttrConstraint('{{'name': 'opt_elem2', 'sources': ['pupil'], 'value': '{pupil}'}})\n"
                       f"DMSAttrConstraint('{{'name': 'subarray', 'sources': ['subarray'], 'value': '{subarray}'}})\n"
                       f"Constraint_Image('{{'name': 'exp_type', 'sources': ['exp_type'], 'value': '{exp_type}'}})",
        "asn_id": filter + "_asn",  # Use a custom ID if combining different observations
        "target": target,
        "asn_pool": filter + "_pool",
        "products": [
            {
                "name": f"{target}_nircam_clear-{filter}",
                "members": members
            }
        ]
    }

    # Save the association to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(association, f, indent=4)

# It is important to make sure that the exp_dir is filled with exposures of the same filter corresponding to the mosaic image.
exp_dir = './cal_files/' # Or whichever directory your exposures reside in.
output_filename = 'my_association.json'
program = 'program_ID'
target = 'target_ID'
instrument = 'nircam'
filter = 'filter'

create_custom_association(exp_dir, output_filename, program, target, instrument, filter, pupil="clear", subarray="full", exp_type="nrc_image")