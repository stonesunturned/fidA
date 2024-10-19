def op_addrcvrs(indat,phasept=0,mode='w',coilcombos=None):
    if indat.flags['addedrcvrs'] or indat.dims['coils']>-1:
        print('WARNING:  Only one receiver channel found!  Returning input without modification!')
        outdat=indat.copy()
        outdat.flags['addedrcvrs']=1
        fids_presum=indat.fids
        specs_presum=indat.specs
        coilcombos={'phs':0, 'sigs':1}
    else:
        #To get best possible SNR, add the averages together (if it hasn't already been done):
        if not indat.flags['averaged']:
            av=op_averaging(indat)
        else:
            av=indat
        # also, for best results, we will combine all subspectra:
        if coilcombos is None:
            if indat.flags['isFourSteps']:
                av=op_fourStepCombine(av)
            if indat.dims['subSpecs']>-1:
                av=op_combinesubspecs(av,'summ')
        avfids=av.fids
        # Find the relative phases between the channels and populate the ph matrix
        # Won't avfids have different size depending on the size of fids??
        if coilcombos is None:
            # Jamie unwraps the phase in the Matlab version but only ever uses the angle. Maybe for viewing? Leaving it wrapped for now
            phs=[np.angle(avfids[phasept,nct,0,0])*180/np.pi for nct in range(avfids.shape[1])]
            if mode=='w':
                sigs=np.abs(avfids[phasept,:,0,0])
            elif mode=='h':
                S=np.max(np.abs(avfids[:,:,0,0]),axis=0)
                N=np.std(avfids[-100:,:,0,0],axis=0)
                sigs=S/(N**2)
        else:
            phs=coilcombos['phs']
            sigs=coilcombos['sigs']
        #now replicate the phase matrix to equal the size of the original matrix:
        sigs=sigs/np.linalg.norm(sigs.flatten())
        ph=np.ones(indat.sz)
        sig=np.ones(indat.sz)
        # and then adjust the phase in each dimension
        if indat.dims['coils']==0:
            for nct in range(indat.sz[0]):
                ph[nct,:]=phs[nct]*ph[nct,:]
                sig[nct,:]=sigs[nct]*sig[nct,:]
        elif indat.dims['coils']==1:
            for nct in range(indat.sz[1]):
                ph[:,nct,:]=phs[nct]*ph[:,nct,:]
                sig[:,nct,:]=sigs[nct]*sig[:,nct,:]
        elif indat.dims['coils']==2:
            for nct in range(indat.sz[2]):
                ph[:,:,nct,:]=phs[nct]*ph[:,:,nct,:]
                sig[:,:,nct,:]=sigs[nct]*sig[:,:,nct,:]
        elif indat.dims['coils']==3:
            for nct in range(indat.sz[3]):
                ph[:,:,:,nct,:]=phs[nct]*ph[:,:,:,nct,:]
                sig[:,:,:,nct,:]=sigs[nct]*sig[:,:,:,nct,:]
        elif indat.dims['coils']==4:
            for nct in range(indat.sz[4]):
                ph[:,:,:,:,nct,:]=phs[nct]*ph[:,:,:,:,nct,:]
                sig[:,:,:,:,nct,:]=sigs[nct]*sig[:,:,:,:,nct,:]
        #now apply the phases by multiplying the data by exp(-i*ph);
        fids=indat.fids*np.exp(-1j*ph*np.pi/180)
        fids_presum=fids
        specs_presum=fids.specs
        # Apply the amplitude factors my multiplying by the amplitude
        if mode=='w' or mode=='h':
            fids=fids*sig
        #Make the coilcombos structure:
        coilcombos={'phs':phs,'sigs':sigs}
        #now sum along coils dimension
        fids=np.sum(fids,axis=indat.dims['coils'])
        fids=np.squeeze(fids)
        outdat=indat.copy()
        outdat.fids=fids
        # change the dims variables
        for dimnm in indat.dims.keys():
            if indat.dims[dimnm]>indat.dims['coils']:
                outdat.dims[dimnm]=outdat.dims[dimnm]-1
        outdat.dims['coils']=-1
        outdat.flags['addedrcvrs']=True
        outdat.flags['writtentostruct']=True      
    return outdat,fids_presum,specs_presum,coilcombos