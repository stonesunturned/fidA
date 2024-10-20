#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:13:32 2022


fidA_io.py
Colleen Bailey, Sunnybrook Research Institute
Based on the FID-A Matlab code by Jamie Near, McGill University 2014.


"""
import os
import numpy as np
from scipy.fft import fftshift, ifft, fft
from datetime import date
GAMMAP=42577000
import matplotlib.pyplot as plt
import fidA_processing as fop
from spec2nii.GE.ge_pfile import read_pfile


class FID(object):
    """
    class FID(object)
    A FID object holds the fid that has been read in from a numpy array, either
    individual averages, coil readings, subspecs, or combinations. This information
    for each dimension is held in the dims attribute. There are also flags to
    describe what processing has been done and some sequence parameters.
    The spectrum "specs" from the fid and several other properties are calculated
    from existing attributes
    These objects can be processed using the functions in fidA_processing.
    """
    def __init__(self,fids,raw_avgs,spectralwidth,txfreq,te,tr,sequence=None,subSpecs=None,rawSubspecs=None,pts_to_left_shift=0,flags=None,dims=None):
        """
        FID(fids,raw_avgs,spectralwidth,txfreq,te,tr,sequence=None,subSpecs=None,rawSubspecs=None,pts_to_left_shift=0,flags=None,dims=None))

        """
        self.fids=fids
        if dims:
            self.dims=dims.copy()
        else:
            self.dims={'t':0,'averages':-1,'subSpecs':-1}
        self.spectralwidth=spectralwidth
        self.txfreq=txfreq
        self.te=te
        self.tr=tr
        self.sequence=sequence
        self.date=date.today()
        self.subSpecs=subSpecs
        if self.dims['subSpecs']!=-1 and self.dims['averages']==-1:
            self.averages=raw_avgs
            self.rawAverages=1
        else:
            self.averages=raw_avgs
            self.rawAverages=raw_avgs
        self.rawSubspecs=rawSubspecs
        self.pointsToLeftshift=pts_to_left_shift
        self.added_ph0=0 #this may need to be a vector for rawdata files with multiple averages)
        self.added_ph1=0
        self._ppmmin=4.65+self.spectralwidth/(self.txfreq/1e6)/2
        self._ppmmax=4.65-self.spectralwidth/(self.txfreq/1e6)/2
        if flags:
            self.flags=flags.copy()
        else:
            flagpars=['getftshifted','filtered','zeropadded','freqcorrected','phasecorrected',
                      'addedrcvrs','subtracted','writtentotext','downsampled','avgNormalized',
                      'isFourSteps']
            self.flags={parnm:False for parnm in flagpars}
            self.flags['writtentostruct']=True
            self.flags['gotparams']=True
            self.flags['addedrcvrs']=True
        if raw_avgs is None or raw_avgs==1:
            self.flags['averaged']=True
        else:
            self.flags['averaged']=False

    @property
    def specs(self):
        return fftshift(ifft(self.fids,axis=self.dims['t']),axes=self.dims['t'])
    @property
    def Bo(self):
        return self.txfreq/GAMMAP
    @property
    def spectralwidthppm(self):
        return self.spectralwidth/(self.txfreq/1e6)
    @property
    def ppm(self):
        self._ppm=np.linspace(self._ppmmin,self._ppmmax,len(self.specs))
        return self._ppm
    @property
    def dwelltime(self):
        return 1/self.spectralwidth
    @property
    def t(self):
        #t2=np.r_[0:self.fids.shape[self.dims['t']]*self.dwelltime:self.dwelltime]
        t2=np.linspace(0,self.fids.shape[self.dims['t']]*self.dwelltime,self.fids.shape[self.dims['t']]+1)[:-1]
        # Corrects for potential rounding errors
        #if len(t2)!=self.fids.shape[self.dims['t']]:
        #    t2=np.r_[0:(self.fids.shape[self.dims['t']]-1)*self.dwelltime:self.dwelltime]
        return t2
    @property
    def sz(self):
        return self.specs.shape
    def __mul__(self,mult1):
        if isinstance(mult1, FID):
            out1=self.copy()
            out1.fids=self.fids*mult1.fids
            return out1
        elif isinstance(mult1, int) or isinstance(mult1, float):
            out1=self.copy()
            out1.fids=self.fids*mult1
            return out1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
    def __rmul__(self,mult1):
        if isinstance(mult1, int) or isinstance(mult1, float):
            out1=self.copy()
            out1.fids=self.fids*mult1
            return out1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
    def __add__(self,add1):
            if isinstance(add1, FID):
                out1=self.copy()
                out1.fids=self.fids+add1.fids
                return out1
            elif isinstance(add1, int) or isinstance(add1, float):
                out1=self.copy()
                out1.fids=self.fids+add1
                return out1
            else:
                raise TypeError(f"sorry, don't know how to add {type(add1).__name__}")
    def __sub__(self,add1):
            if isinstance(add1, FID):
                out1=self.copy()
                out1.fids=self.fids-add1.fids
                return out1
            elif isinstance(add1, int) or isinstance(add1, float):
                out1=self.copy()
                out1.fids=self.fids-add1
                return out1
            else:
                raise TypeError(f"sorry, don't know how to add {type(add1).__name__}")
    def __div__(self,mult1):
        if isinstance(mult1, FID):
            out1=self.copy()
            out1.fids=self.fids/mult1.fids
            return out1
        elif isinstance(mult1, int) or isinstance(mult1, float):
            out1=self.copy()
            out1.fids=self.fids/mult1
            return out1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
    def __truediv__(self,mult1):
        return self.__div__(mult1)
    def copy(self):
        newfid=FID(self.fids,self.rawAverages,self.spectralwidth,self.txfreq,
                   self.te,self.tr,self.sequence,self.subSpecs,
                   self.rawSubspecs,self.pointsToLeftshift,self.flags.copy())
        newfid.dims=self.dims.copy()
        newfid.date=date.today()
        newfid.averages=self.averages
        return newfid
    def plot_spec(self,xlims=[4.5,0],xlab='Chemical Shift (ppm)',ylab='Signal',title='',plotax=None):
        # Need to update to deal with multiple averages and other possible dimensions, as in Matlab op_plotspec
        if plotax is None:
            [f1,plotax]=plt.subplots(1,1)
        # This attempt to generate a plot for multiple coils is bad. Whatever is happening with the
        # Fourier Transform seems to be adding in phasing or something. The absolute spectra
        # look okay but the real component has lots of weird phase additions (although maybe that
        # is to be expected and Bruker deals with that automatically in processing)
        # plt.plot(np.abs(fop.op_averaging(indat).specs)) looks very different than indat.plot_specs()
        if self.fids.ndim==3: #probably a more generic way to do this, but for Bruker multi-coil, I'll combine averages to see each coil
            #spec_for_plot=np.real(np.mean(self.specs,axis=self.dims['averages']))
            spec_for_plot=np.real(fop.op_averaging(self).specs)
        else:
            spec_for_plot=np.real(self.specs)
        plotax.plot(self.ppm,spec_for_plot)
        plotax.set_xlim(xlims)
        plotax.set_xlabel(xlab)
        plotax.set_ylabel(ylab)
        plotax.set_title(title)
    def add_ph0(self,ph0):
        # Note that this is a permanent change to the FID object. To test different phases, use op_add_phase
        self.fids=self.fids*np.exp(1j*ph0*np.pi/180)
        self.added_ph0=self.added_ph0+ph0
        self.flags['phasecorrected']=True
    def add_ph1(self,ph1,ppm0=4.65):
        # Note that this is a permanent change to the FID object. To test different phases, use op_add_phase
        newspec=fop.add_phase1(self.specs,self.ppm,ph1,ppm0,self.Bo)
        self.set_fid_from_specs(newspec)
        self.added_ph1=self.added_ph1+ph1
        self.flags['phasecorrected']=True
    def do_averaging(self):
        tmpfid=fop.op_averaging(self.copy())
        self.fids=tmpfid.fids
        self.dims=tmpfid.dims
        self.flags=tmpfid.flags
        self.averages=1
    def autophase(self,ppm1=1.9,ppm2=2.1,phstart=0):
        # Permanent change. Default is to use the NAA peak
        newfid, phnew=fop.op_autophase(self.copy(),ppm1,ppm2,ph=phstart)
        self.fids=newfid.fids
        self.added_ph0=newfid.added_ph0
    def set_fid_from_specs(self, newspec):
        # Need to recalculate fid from spec, but you have to do a circshift when the
        # length of fids is odd so you don't introduce a small frequency shift into fids
        # Note that newspec should be calculated from the existing specs but this is not checked.
        # Could create problems if newspec doesn't match existing ppm, etc.
        # Also note that this function SETS self.fids. It DOES NOT return a FIDS object
        # If you're getting errors, check that you are not calling the function expecting a return
        if np.mod(newspec.shape[self.dims['t']],2)==0:
            self.fids=fft(fftshift(newspec,axes=self.dims['t']),axis=self.dims['t'])
        else:
            self.fids=fft(np.roll(fftshift(newspec,axes=self.dims['t']),1,axis=self.dims['t']),axis=self.dims['t'])
    
def get_par(fname,parname,vartype='float'):
    """
    raw_par=get_par(fname,parname,vartype='float')
    Helper file to return the value for a particular parameter in a Bruker
    method, proc, etc. file

    Parameters
    ----------
    fname : FILE STRING
        Full or relative path to file to open.
    parname : STRING
        Variable name to find, including '$' but not '='. eg. $PVM_DigNp.
    vartype : STRING, optional
        Variable type, used to convert from string before return. The default is 'float'.

    Returns
    -------
    raw_par : float, int or string, depending on value of 'vartype'
        Value of parname.
    """
    line=''
    with open(fname) as f:
        while parname+'=' not in line:
            line=f.readline()
    if vartype=='float':
        raw_par=float(line[line.find('=')+1:])
    elif vartype=='int':
        raw_par=int(line[line.find('=')+1:])
    else:
        raw_par=line[line.find('=')+1:].strip()
    return raw_par

def io_loadspec_bruk(inDir, spectrometer=False, try_raw=False, info_dict=False, ADC_OFFSET=68):
    # Get relevant parameters (this could be done more efficiently by opening files once at the start and reading all relevant parameters from one opening)
    if spectrometer:
        dic1 = read_jcamp_spec(os.path.join(inDir, 'acqu'))
        spectralwidth = dic1['SW_h']
        txfrq = dic1['SFO1'] * 1e6
        te = -1
        tr = -1
        sequence = dic1['PULPROG'][1:-1]
        if info_dict:
            info_dict = dic1
        else:
            info_dict = None
    else:
        spectralwidth = get_par(os.path.join(inDir, 'method'), '$PVM_DigSw')
        txfrq = get_par(os.path.join(inDir, 'acqp'), '$BF1') * 1e6
        te = get_par(os.path.join(inDir, 'method'), '$PVM_EchoTime')
        tr = get_par(os.path.join(inDir, 'method'), '$PVM_RepetitionTime')
        sequence = get_par(os.path.join(inDir, 'method'), '$Method', 'string')
        if info_dict:
            info_dict = read_jcamp_spec(os.path.join(inDir, 'method'))
        else:
            info_dict = None
    # Specify the number of subspecs.  For now, this will always be one.
    subSpecs = 1
    rawSubspecs = 1

    def get_spec(fname, raw_avgs=None):
        fid_data = np.fromfile(fname, dtype=np.int32)
        # Corrected indexing to match MATLAB behavior
        imag_fid = fid_data[0::2]
        real_fid = fid_data[1::2]
        fids_raw = real_fid - 1j * imag_fid
        if fname.endswith('fid.raw') or fname.endswith('rawdata.job0'):
            # Additional code to handle specific file types
            raw_avgs = get_par(os.path.join(inDir, 'method'), '$PVM_NAverages', 'int')
            expmode = get_par(os.path.join(inDir, 'acqp'), '$ACQ_experiment_mode', 'string')
            if expmode.strip() == 'zzParallelExperiment':
                ncoil = int(get_par(os.path.join(inDir, 'acqp'), '$ACQ_ReceiverSelect', 'string').split()[1])
                coildim = 1
                avgdim = 2
                fids_raw = np.reshape(fids_raw, [raw_avgs * ncoil, -1]).T
            else:
                coildim = -1
                avgdim = 1
                fids_raw = np.reshape(fids_raw, [raw_avgs, -1]).T
        elif fname.endswith('fid'):
            if os.path.exists(os.path.join(inDir, 'method')):
                raw_dat_pts = get_par(os.path.join(inDir, 'method'), '$PVM_DigNp', 'int')
            else:
                raw_dat_pts = len(real_fid)
            raw_avgs = int(np.shape(real_fid)[0] / raw_dat_pts)
            if np.mod(real_fid.shape[0], raw_dat_pts) != 0:
                print('Number of repetitions cannot be accurately found')
            fids_raw = np.reshape(fids_raw, [-1, raw_dat_pts]).T
            avgdim = -1
            coildim = -1
        elif fname.endswith('fid.ref'):
            fids_raw = np.reshape(fids_raw, [raw_avgs, -1]).T
            avgdim = 1
            coildim = -1
        elif fname.endswith('fid.refscan'):
            raw_dat_pts = len(real_fid)
            raw_avgs = 1
            if np.mod(real_fid.shape[0], raw_dat_pts) != 0:
                print('Number of repetitions cannot be accurately found for refscan file')
            fids_raw = np.reshape(fids_raw, [-1, raw_dat_pts]).T
            avgdim = -1
            coildim = -1
        try:
            fids_trunc = fids_raw[ADC_OFFSET:, :]
        except IndexError:
            fids_trunc = np.expand_dims(fids_raw, axis=1)
            fids_trunc = fids_trunc[ADC_OFFSET:, :]
        fids = np.pad(fids_trunc, pad_width=[[0, ADC_OFFSET], [0, 0]])
        if coildim != -1:
            fids = np.transpose(np.reshape(fids, [-1, raw_avgs, ncoil]), [0, 2, 1])
        fid1 = FID(fids, raw_avgs, spectralwidth, txfrq, te, tr, sequence, subSpecs, rawSubspecs)
        fid1.dims['averages'] = avgdim
        fid1.dims['t'] = 0
        fid1.dims['coils'] = coildim
        fid1.dims['subSpecs'] = -1
        fid1.dims['extras'] = -1
        if fid1.dims['subSpecs'] == 0:
            fid1.flags['isFourSteps'] = 0
        else:
            fid1.flags['isFourSteps'] = (fid1.sz[fid1.dims['subSpecs']] == 4)
        return fid1, raw_avgs

    # First try to load fid.raw file. If that does not work, use regular fid
    if try_raw:
        try:
            outfid, rawavg1 = get_spec(os.path.join(inDir, 'rawdata.job0'))
        except FileNotFoundError:
            outfid, rawavg1 = get_spec(os.path.join(inDir, 'fid.raw'))
    else:
        # Untested since I don't have an example dataset with .raw missing
        print('WARNING: /fid.raw not found. Using /fid ....')
        outfid, rawavg1 = get_spec(os.path.join(inDir, 'fid'))

    # Now try loading in the reference scan data (if it exists)
    if os.path.isfile(os.path.join(inDir, 'fid.ref')):
        reffid, rawavg2 = get_spec(os.path.join(inDir, 'fid.ref'), raw_avgs=rawavg1)
    elif os.path.isfile(os.path.join(inDir, 'fid.refscan')):
        reffid, rawavg2 = get_spec(os.path.join(inDir, 'fid.refscan'), raw_avgs=rawavg1)
    else:
        print('Could not find reference scan. Skipping')
        reffid = 0
    return outfid, reffid, info_dict


def read_jcamp_spec(fname):
    """
    read_jcamp_spec(fname)
    Opens a Bruker file with scan information and reads in each variable to a dictionary
    
    inputs:
    fname - string of the Bruker file
                
    returns:
    dic - a dictionary of scan parameters from the file fname. It attempts to get the datatype of the values correct
          (int, float, string; single values or arrays (which are saved in Python's list format)). Some arrays 
          (eg. ACQ_coil_elements) are arrays of lists in brackets and these are left as a single string rather than
          converting into a list of lists.
    """    
    dic=dict()

    #read everything that's not a comment ("$$) or header("##(not$)")
    with open(fname,'r') as f:
        linelist=[line.rstrip() for line in f if (line[:2]!="$$" and not (line[:2]=="##" and line[2]!="$"))]
                
    #The file has line breaks for some parameters, some of which are arrays and some of which are not. The strategy 
    #I've taken is to join the entire file into one line, separated by spaces, then split again wherever there's a 
    #comment indicator (##$). This should put each parameter on its own line. The first element of the new list 
    #will be empty since the file starts with ##$
    plist=" ".join(linelist).split("##$")[1:]
    for par in plist:
        dkey,dval=par.split('=',maxsplit=1)        
        if "(" in dval and " )" in dval: #array case
            try:
                arrShape=[int(d.strip()) for d in dval[dval.index("(")+2:dval.index(" )")].split(', ')]
                dval=dval[dval.index(" )")+3:] #everything after the array size is the variable we want
                olddval=dval #save in case the array reshape doesn't work (which should mean a single value string)
                dval=[type_it(delem.strip()) for delem in dval.split()]
                try: #sometimes it just gives you the shape with no values
                    np.array(dval).reshape(arrShape)
                    dic[dkey]=dval
                except ValueError:
                    dic[dkey]=olddval #if we can't get the shape right, just leave it
            except ValueError: #for CPMG sequence, one of the excitatory pulses has a space before ')' and gets read as an array.
                dval=dval[dval.index("(")+1:dval.index(")")].split(", ")
                dval=[type_it(delem.strip()) for delem in dval]
                dic[dkey]=dval        
        elif "(" in dval and " )" not in dval: #multi-line parameter case
            #here there is no array size, so the thing in the brackets is the parameter we want
            dval=dval[dval.index("(")+1:dval.index(")")].split(", ")
            dval=[type_it(delem.strip()) for delem in dval]
            dic[dkey]=dval
        else: #simple scalar parameter
            dic[dkey]=type_it(dval.strip())     
    return dic
        
def type_it(dval):
    """
    type_it(dval)
    Values read in using fread are strings. This should cast them as the correct type
    
    inputs:
    dval - string of the parameter value
    returns:
    dval - parameter value recast as the correct type (float, int or string)
    """
    try:
        dval=int(dval)
    except ValueError:
        try:
            dval=float(dval)
        except ValueError:
            pass #string case
    return dval

def io_readlcmcoord(fname):
    lcm_dict=dict()
    info_dict=dict()
    with open(fname) as f:
        coord_text=f.read()
    div_strs=[' points on ppm-axis = NY','NY phased data points follow',
              'NY points of the fit to the data follow','NY background values follow',
              ' Conc. = ']
    coord_lines=coord_text.split('\n')
    nlines=int(coord_lines[2].split()[0])
    info_dict['names']=[eachlines.split()[3] for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['conc']=[float(eachlines.split()[0]) for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['SD_perc']=[int(eachlines.split()[1][:-1]) for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['SNR']=int(coord_lines[4+nlines].split()[-1])
    tmpline=coord_lines[4+nlines]
    info_dict['FWHM']=float(tmpline[tmpline.index('=')+1:tmpline.index('ppm')])
    tmpline=coord_lines[4+nlines+1]
    info_dict['shift']=float(tmpline[tmpline.index('=')+1:tmpline.index('ppm')])
    tmpline=coord_lines[4+nlines+2]
    info_dict['phase']=float(tmpline[tmpline.index('deg')+3:tmpline.index('deg/ppm')])
    for divct,(divstr,partstr) in enumerate(zip(div_strs[:-1],['ppm','data','fit','bgrd'])):
        tmppts=coord_text[coord_text.find(divstr)+len(divstr):coord_text.find(div_strs[divct+1])]
        # one possibility is to save each of these as a FID object so that you can use the same
        # functions and stuff that you do for raw data but not sure how useful this is. Mostly
        # it might work for plotting but also you often want to plot multiple metabolites as a 
        # ridgeplot and I already have a function for that.
        if partstr=='bgrd': # Need to cut metabolite name after split. Can't specify metabolite in string because depends on basis set
            lcm_dict[partstr]=np.array([float(numstr) for numstr in tmppts.split()[:-1]])
        else:
            lcm_dict[partstr]=np.array([float(numstr) for numstr in tmppts.split()])
    # split up text into chunks for each metabolite
    tmpconc=coord_text.split('Conc. = ')
    npts=len(lcm_dict['ppm'])+1
    for concct,eachconc in enumerate(tmpconc[1:]):
        metname=tmpconc[concct].split()
        metname=metname[-1].strip()
        #print(metname)
        lcm_dict[metname]=np.array([float(numstr) for numstr in eachconc.split()[1:npts]])
    return lcm_dict, info_dict

    
def io_writelcm(infid,outfile,te=None,vol=8.0):
    # Need to add the error checks, equivalent to those in Matlab
    RF=np.zeros([infid.fids.shape[infid.dims['t']],2])
    RF[:,0]=np.imag(infid.fids[:,0])
    RF[:,1]=np.real(infid.fids[:,0])
    with open(outfile,'w') as f:
        f.write(' $SEQPAR')
        f.write('\n echot= {:4.3f}'.format(infid.te))
        f.write("\n seq= 'PRESS'")
        f.write('\n hzpppm= {:5.6f}'.format(infid.txfreq/1e6))
        f.write('\n NumberOfPoints= {:n}'.format(infid.fids.shape[infid.dims['t']]))
        f.write('\n dwellTime= {:5.6f}'.format(infid.dwelltime))
        f.write('\n $END')
        f.write('\n $NMID')
        f.write("\n id='ANONYMOUS ', fmtdat='(2E15.6)'")
        f.write('\n volume={:4.3e}'.format(vol))
        f.write('\n tramp=1.0')
        f.write('\n $END\n')
        for eachct in range(RF.shape[0]):
            f.write('  {:7.6e}  {:7.6e}\n'.format(RF[eachct,0],RF[eachct,1]))

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

    # No need to assign to out.specs; it will be computed automatically using the updated fids

    return out



if __name__ == '__main__':
    """
    for debugging
    """
    fname='/mnt/d/Bruker_press'
    fpath = ''
    outfid, reffid, info_dict = io_loadspec_bruk(fname, try_raw=True)
    print(outfid.sz)
    median_outfid = op_median(outfid)
    print(median_outfid.specs)