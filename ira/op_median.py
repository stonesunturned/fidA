# fidA/ira/op_median.py

import numpy as np
from scipy.fft import fftshift, ifft
from fidA.utils.fid import FID

def op_median(in_fid):
    """
    Combine the averages in a scan by calculating the median of all averages.

    Parameters:
    in_fid (FID): Input data in FID object format.

    Returns:
    FID: Output dataset following median calculation.
    """
    # Check if the data is already averaged or has less than 2 averages
    if in_fid.flags['averaged'] or in_fid.dims['averages'] == -1 or in_fid.averages < 2:
        print('WARNING: No averages found! Returning input without modification!')
        return in_fid

    # Calculate the median along the averages dimension
    fids_real_median = np.median(np.real(in_fid.fids), axis=in_fid.dims['averages'])
    fids_imag_median = np.median(np.imag(in_fid.fids), axis=in_fid.dims['averages'])
    fids = fids_real_median + 1j * fids_imag_median
    fids = np.squeeze(fids)

    # Update dims
    dims = in_fid.dims.copy()
    averages_dim = in_fid.dims['averages']

    # Adjust dimensions based on the averages dimension
    for key in dims.keys():
        if dims[key] > averages_dim:
            dims[key] -= 1
        elif dims[key] == averages_dim:
            dims[key] = -1  # Set to -1 since we've averaged over it

    # Create the output FID object
    out = in_fid.copy()
    out.fids = fids
    out.dims = dims
    out.averages = 1
    out.flags = in_fid.flags.copy()
    out.flags['writtentostruct'] = True
    out.flags['averaged'] = True

    return out
