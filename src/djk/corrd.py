#!/usr/bin/env python3

import numpy as np
from scipy import stats
import treecorr
from astropy.cosmology import FlatLambdaCDM
from scipy.special import binom
from itertools import combinations
from jackknife import jackknife

###rotational matrix of angle pi/4###
matrix = np.array([[0.70710678, -0.70710678],[ 0.70710678,  0.70710678]])


class deleted_jackknife(jackknife):
    
    def __init__(self,*args,**qwargs):
                
        if 'bin_slop' in qwargs:
            self.bin_slop = qwargs['bin_slop']
        else:
            self.bin_slop = 0.
        
        # Oversampling factor for rp_cut accuracy
        if 'oversample_factor' in qwargs:
            self.oversample_factor = qwargs['oversample_factor']
        else:
            self.oversample_factor = 1  # No oversampling by default
            
        self.jackpairs = []    
        self.read_input(**qwargs)
        self.get_combinations()
        self.compute_RR = qwargs['compute_RR']
        
        # Weighting scheme for cross-pairs: 'simple' (default) or 'match' (Mohammad & Percival 2022)
        if 'cross_patch_weight' in qwargs:
            self.cross_patch_weight = qwargs['cross_patch_weight']
        else:
            self.cross_patch_weight = 'simple'
        
        # Compute cross-pair weight factor (Eq. 27 of Mohammad & Percival 2022)
        if self.cross_patch_weight == 'match':
            # w = 1 - alpha, where alpha = n_sv / (2 + sqrt(2) * (n_sv - 1))
            alpha = self.Ns / (2.0 + np.sqrt(2.0) * (self.Ns - 1))
            self.w_cross = 1.0 - alpha
            print(f"Using 'match' cross-patch weighting (Mohammad & Percival 2022)")
            print(f"  Ns = {self.Ns}, alpha = {alpha:.4f}, w_cross = {self.w_cross:.4f}")
        else:
            # 'simple' weighting: exclude cross-pairs with excluded region entirely
            self.w_cross = 0.0
            print(f"Using 'simple' cross-patch weighting (standard jackknife)")
        
        super().__init__(**qwargs)

    def dist(self,x,y):
        return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)


    def get_combinations(self):
        
        """Total number of combination"""
        self.comb = int(binom(self.Ns,self.Nd))
        
        """get a list of all d-deleted jackknife combinations"""
        
        comb = [",".join(map(str, comb)) for comb in combinations(self.upatches, self.Ns - self.Nd)]        
        list_comb = list(map(eval,comb))
        self.array_comb = np.array(list_comb)

        """Get all one by one combinations between unique patches"""
        patcht = self.upatches
        for i in range(0,len(self.upatches)):
            self.jackpairs.append([(i,ii) for ii in patcht if ii != i])           
            
        self.jackpairs = np.vstack(self.jackpairs)

    def read_input(self,**qwargs):

        self.cosmo = qwargs['cosmology']

        self.Ns = qwargs['Ns']
        self.Nd = qwargs['Nd']
        self.computation = qwargs['computation']
        self.machine = qwargs['machine']
        binsfile = qwargs['binsfile']
        self.upatches = np.linspace(0,self.Ns-1,self.Ns,dtype='int')

        if self.computation in ['WGP','WGG','WPP','WCC','xil','W_+','xi2p']:
            self.twoD = True
            
            # Store coarse bins (user's desired output)
            self.bins1_coarse = binsfile[0]
            self.bins2 = binsfile[1]
            
            # Create fine bins for internal computation
            if self.oversample_factor > 1:
                self.bins1 = self._create_fine_bins(self.bins1_coarse, self.oversample_factor)
                print(f"Oversampling s-bins: {len(self.bins1_coarse)-1} coarse -> {len(self.bins1)-1} fine bins (factor={self.oversample_factor})")
            else:
                self.bins1 = self.bins1_coarse
            
            self.pimax = np.max(self.bins2)
            self.du = abs(self.bins2[1]-self.bins2[0])
            self.bins2_val = (self.bins2[:-1]+self.bins2[1:])/2.
        else:
            self.twoD = False
            self.bins1_coarse = binsfile[0]
            self.bins1 = binsfile[0]

        """read data/random"""

        ra =  qwargs['RA']
        dec =  qwargs['DEC']
        w =  qwargs['W']
        if 'Z' in qwargs:
            z =  qwargs['Z']
            dc = self.cosmo.comoving_distance(z).value
        else:
            dc = qwargs['Dc']
        
        g1 = np.zeros(len(ra))
        g2 = np.zeros(len(ra))
        g12 = np.zeros(len(ra))
        g22 = np.zeros(len(ra))
        
        if 'g1' in qwargs:
            g1 = qwargs['g1']
            g2 = qwargs['g2']
            
        if 'patch_centers' in qwargs:
            self.patch_centers = qwargs['patch_centers']
            self.data1_ = treecorr.Catalog(ra=ra,dec=dec,r=dc,g1=g1,g2=g2,w=w,ra_units='degree',\
                dec_units='degree',npatch=self.Ns,patch_centers=self.patch_centers)
        else:
            data1_ =  treecorr.Catalog(ra=ra,dec=dec,ra_units='degree',\
                dec_units='degree',npatch=self.Ns)
            self.patch_centers = data1_.patch_centers

            self.data1_ = treecorr.Catalog(ra=ra,dec=dec,r=dc,g1=g1,g2=g2,w=w,ra_units='degree',\
                dec_units='degree',npatch=self.Ns,patch_centers=self.patch_centers)

        if 'thetamax' in qwargs:
            self.thetamax = qwargs['thetamax']
            
        ra2 =  qwargs['RA2']
        dec2 =  qwargs['DEC2']
        w2 =  qwargs['W2'] 
        
        if 'Z2' in qwargs:
            z2 =  qwargs['Z2']
            dc2 = self.cosmo.comoving_distance(z2).value
        else:
            dc2 = qwargs['Dc2']

        if 'g12' in qwargs:
            g12 = qwargs['g12']
            g22 = qwargs['g22']

        self.data2_ = treecorr.Catalog(ra=ra2,dec=dec2,r=dc2,w=w2,g1=g12,g2=g22,ra_units='degree',\
                dec_units='degree',npatch=self.Ns,patch_centers=self.patch_centers)

        posC = np.column_stack([ra,dec])
        pos = np.column_stack([ra2,dec2])
        nrows, ncols = pos.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
               'formats':ncols * [posC.dtype]}

        uniquelen = np.maximum(len(np.unique(posC,axis=0)),len(np.unique(pos,axis=0)))
        C,xind,yind = np.intersect1d(posC.view(dtype), pos.view(dtype),return_indices=True)
        self.w_int1 = self.data1_.w[xind]
        self.w_int2 = self.data2_.w[yind]        

        if (len(self.w_int1) > 0) & (len(self.w_int1) < uniquelen):
            self.corr = 'subsample'
        elif len(self.w_int1)==0:
            self.corr = 'cross'
        elif len(self.w_int1)==uniquelen:
            self.corr  = 'auto'

        ra_r =  qwargs['RA_r']
        dec_r =  qwargs['DEC_r']
        w_r =  qwargs['W_r']
        
        if 'Z_r' in qwargs:
            z_r =  qwargs['Z_r']
            dc_r = self.cosmo.comoving_distance(z_r).value
        else:
            dc_r = qwargs['Dc_r']
            
        self.rand1_ = treecorr.Catalog(ra=ra_r,dec=dec_r,r=dc_r,w=w_r,ra_units='degree',\
            dec_units='degree',npatch=self.Ns,patch_centers=self.patch_centers,is_rand=1)

        
        if 'RA_r2' in qwargs:
            
            self.cross_RR = True
            ra_r2 =  qwargs['RA_r2']
            dec_r2 =  qwargs['DEC_r2']
            w_r2 =  qwargs['W_r2']
            if 'Z_r2' in qwargs:
                z_r2 =  qwargs['Z_r2']
                dc_r2 = self.cosmo.comoving_distance(z_r2).value
            else:
                dc_r2 = qwargs['Dc_r2']

            self.rand2_ = treecorr.Catalog(ra=ra_r2,dec=dec_r2,r=dc_r2,w=w_r2,ra_units='degree',\
                dec_units='degree',npatch=self.Ns,patch_centers=self.patch_centers,is_rand=1)
            
        else:
            self.rand2_ = self.rand1_
            self.cross_RR = False
    
    def _create_fine_bins(self, bins_coarse, oversample_factor):
        """
        Create fine bins by subdividing each coarse bin.
        Respects log vs linear spacing.
        
        Parameters
        ----------
        bins_coarse : np.ndarray
            Coarse bin edges
        oversample_factor : int
            Number of fine bins per coarse bin
            
        Returns
        -------
        np.ndarray
            Fine bin edges
        """
        from jackknife import is_log_spaced
        
        fine_edges = []
        
        if is_log_spaced(bins_coarse):
            # Log-spaced bins
            for i in range(len(bins_coarse) - 1):
                sub_edges = np.geomspace(bins_coarse[i], bins_coarse[i+1], oversample_factor + 1)
                fine_edges.extend(sub_edges[:-1])
            fine_edges.append(bins_coarse[-1])
        else:
            # Linear-spaced bins
            for i in range(len(bins_coarse) - 1):
                sub_edges = np.linspace(bins_coarse[i], bins_coarse[i+1], oversample_factor + 1)
                fine_edges.extend(sub_edges[:-1])
            fine_edges.append(bins_coarse[-1])
        
        return np.array(fine_edges)