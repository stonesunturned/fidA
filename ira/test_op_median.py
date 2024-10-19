# fidA/ira/test_op_median.py

import pytest
import numpy as np
from fidA.utils.fid import FID
from fidA.ira.op_median import op_median

def test_op_median():
    # Create a sample FID object with averages
    dims = {'t': 0, 'averages': 1, 'coils': -1, 'subSpecs': -1, 'extras': -1}
    flags = {'averaged': False, 'writtentostruct': True}
    fids = (np.random.randn(1024, 32) + 1j * np.random.randn(1024, 32))  # 32 averages
    in_fid = FID(fids=fids, raw_avgs=32, spectralwidth=2000, txfreq=123.25e6,
                 te=30, tr=2000, dims=dims, flags=flags)
    in_fid.averages = 32

    # Apply op_median
    out_fid = op_median(in_fid)

    # Assertions
    assert out_fid.fids.shape == (1024,), "Output FID should have dimensions of (1024,)"
    assert out_fid.averages == 1, "Output FID should have averages set to 1"
    assert out_fid.flags['averaged'] == True, "Flag 'averaged' should be True"
    assert out_fid.dims['averages'] == -1, "Dimension 'averages' should be set to -1"

    # Check that the output FID is indeed the median along the averages dimension
    fids_real_median = np.median(np.real(in_fid.fids), axis=in_fid.dims['averages'])
    fids_imag_median = np.median(np.imag(in_fid.fids), axis=in_fid.dims['averages'])
    expected_fids = fids_real_median + 1j * fids_imag_median
    expected_fids = np.squeeze(expected_fids)

    np.testing.assert_array_almost_equal(out_fid.fids, expected_fids, decimal=6)
