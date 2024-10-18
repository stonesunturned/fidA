# fidA/utils/fid.py
import os
import numpy as np
from scipy.fft import fftshift, ifft, fft
from datetime import date
GAMMAP=42577000
import matplotlib.pyplot as plt
import fidA.fidA_processing as fop
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