#!/usr/bin/env python3

from pycorr.corrfunc import CorrfuncTwoPointCounter
from corrd import *
# dsigma is only needed by the (currently commented-out) GGL excess-surface-density
# path; recent dsigma (>=1.2) renamed/removed dsigma_table, so guard the import to
# keep the clustering/IA path working without it.
try:
    from dsigma.helpers import dsigma_table
    from dsigma.stacking import excess_surface_density
    from dsigma.precompute import precompute
except ImportError:
    dsigma_table = excess_surface_density = precompute = None

import os
# Thread count for pycorr/treecorr pair counting. os.cpu_count() reports the
# physical node's cores regardless of the Slurm cgroup, so on a job allocated
# fewer CPUs it oversubscribes (e.g. 24 threads pinned to 1 core -> ~1-core
# throughput, which silently killed the BGS projected run). Prefer the cgroup
# affinity set, and honour SLURM_CPUS_PER_TASK when present; fall back to
# cpu_count() only off the cluster.
def _detect_ncores():
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return int(slurm)
        except ValueError:
            pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on all platforms
        return os.cpu_count() or 1

ncores = _detect_ncores()



def get_cartesian_positions(ra, dec, dc):
    """Convert RA/Dec (in rad)/Dc to Cartesian coordinates (Mpc/h)."""
    x = dc * np.cos(dec) * np.cos(ra)
    y = dc * np.cos(dec) * np.sin(ra)
    z = dc * np.sin(dec)
    return np.column_stack([x, y, z])


class compute_pairs():
    
    def __init__(self,computation,machine,bins1,bins2=None,bin_slop = 0.):
        self.bins1 = bins1
        self.bins2 = bins2
        self.min_sep = np.min(self.bins1)
        self.max_sep = np.max(self.bins1)
        self.bin_slop = bin_slop
        self.computation = computation
        self.machine = machine

        if self.machine == 'CPU':
            self.compute_clustering = self.compute_clustering_pycorr
            self.compute_DS = self.compute_DS_treecorr
            self.compute_SS = self.compute_SS_treecorr
            
            if self.computation in ['WGG','WGP','WPP','WCC','W_+']:
                self.mode = 'rppi'
            elif self.computation == 'xil' and machine == 'CPU':
                self.mode = 'smu'
                
        elif self.machine == 'GPU':
            from cucount.numpy import Particles, count2, BinAttrs, WeightAttrs
            if self.computation in ['xil','xi2p']:
                self.battrs = BinAttrs(s=bins1, mu=(bins2, 'midpoint'))
            elif self.computation in ['WGG','WGP','WPP','WCC','W_+']:
                self.battrs = BinAttrs(rp=(bins1,'midpoint'), pi=(bins2, 'midpoint'))

            self.compute_clustering = self.compute_clustering_cucount
            self.compute_DS = self.compute_DS_cucount
            self.compute_SS = self.compute_SS_cucount
            self.Particles = Particles  # ← ADDED
            self.count2 = count2        # ← ADDED
            self.WeightAttrs = WeightAttrs  # ← ADD THIS LINE

            

    def get_position_particles(self,ra, dec , dc, w ):

        """ Assume that ra,dec are already in radian """ 
        sky_coords = np.column_stack([ra,dec])
        cartesian = get_cartesian_positions(ra,dec,dc)
        # particles = self.Particles(cartesian, w, sky_coords)
        kwargs = dict()
        particles = self.Particles(cartesian, w,**kwargs)

        return particles


    def get_shape_particles(self,ra, dec , dc, w , e1 , e2 ):

        """ Assume that ra,dec are already in radian """ 
        sky_coords = np.column_stack([ra,dec])
        cartesian = get_cartesian_positions(ra,dec,dc)

        ellipticities = np.column_stack([e1,e2])
        # particles = self.Particles(cartesian, w, sky_coords,ellipticities)

        kwargs = dict()
        kwargs.update(spin_values=ellipticities)
        particles = self.Particles(cartesian, w,**kwargs)

        return particles


            

    def compute_clustering_pycorr(self,catalog1,catalog2=None):

        """treecorr automatically converted (ra,dec) in radian, convert it back to degree for pycorr"""

        ra = np.rad2deg(catalog1.ra)
        dec = np.rad2deg(catalog1.dec)
        r = catalog1.r
        w = catalog1.w
        pos = np.column_stack([ra,dec,r]).T

        if catalog2 is not None:
            ra2 = np.rad2deg(catalog2.ra)
            dec2 = np.rad2deg(catalog2.dec)
            r2 = catalog2.r
            w2 = catalog2.w
            pos2 = np.column_stack([ra2,dec2,r2]).T
            DD = self.compute_pairs_pycorr(pos,w,pos2=pos2,weights2=w2)
        else :
            DD = self.compute_pairs_pycorr(pos,w)
        return DD

    def compute_clustering_cucount(self,catalog1,catalog2=None):

        """treecorr automatically converted (ra,dec) in radian, convert it back to degree for pycorr"""

        
        ra = catalog1.ra
        dec = catalog1.dec
        r = catalog1.r
        w = catalog1.w

        particles =  self.get_position_particles(ra,dec,r,w)

        if catalog2 is not None:
            ra2 = catalog2.ra
            dec2 = catalog2.dec
            r2 = catalog2.r
            w2 = catalog2.w
            particles2 =  self.get_position_particles(ra2,dec2,r2,w2)
        #     DD =  self.count2(particles, particles2, battrs=self.battrs, spin1=0, spin2=0)
        # else :
        #     DD =  self.count2(particles, particles, battrs=self.battrs, spin1=0, spin2=0)    
            DD =  self.count2(particles, particles2, battrs=self.battrs)['weight']
        else :
            DD =  self.count2(particles, particles, battrs=self.battrs)['weight']
        return DD.reshape(len(self.bins1)-1,len(self.bins2)-1)

    def compute_pairs_pycorr(self,pos,weight,pos2=None,weights2=None):
            pycorr = CorrfuncTwoPointCounter(self.mode, [self.bins1,self.bins2], positions1=pos,positions2=pos2, \
                                     position_type='rdd', weights1=weight,weights2=weights2 ,nthreads=ncores,compute_sepsavg=False)
            return pycorr.wcounts

    def compute_sepavg_pycorr(self,pos,weight,pos2,weights2):
            pycorr = CorrfuncTwoPointCounter(self.mode, [self.bins1,self.bins2], positions1=pos,positions2=pos2, \
                                     position_type='rdd', weights1=weight,weights2=weights2 ,nthreads=ncores,compute_sepsavg=True)
            return pycorr.sepavg()
        
        
    def compute_DS_treecorr(self,catalog1,catalog2):
        DS = np.zeros((len(self.bins1)-1,len(self.bins2)-1))
        #wDS = np.zeros((len(self.bins1)-1,len(self.bins2)-1))
        for i in range(0,len(self.bins2)-1):
            pi_min = self.bins2[i]
            pi_max = self.bins2[i+1]
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=len(self.bins1)-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop)
            ng.process(catalog1,catalog2)
            DS[:,i] = ng.xi*ng.weight

        return DS

    def compute_DS_cucount(self,catalog1,catalog2):
        """treecorr automatically converted (ra,dec) in radian"""
        
        ra = catalog1.ra
        dec = catalog1.dec
        r = catalog1.r
        w = catalog1.w

        ra2 = catalog2.ra
        dec2 = catalog2.dec
        r2 = catalog2.r
        w2 = catalog2.w
        e1 = catalog2.g1
        e2 = catalog2.g2

        particles =  self.get_position_particles(ra,dec,r,w)
        particles2 =  self.get_shape_particles(ra2,dec2,r2,w2,e1,e2)

        wattrs = self.WeightAttrs(spin=(0, 2))
        PS_result = self.count2(particles, particles2, battrs=self.battrs,wattrs=wattrs)
        pairs = PS_result['weight_plus']#.reshape(len(self.bins1)-1,len(self.bins2)-1)

        # PS_result = self.count2(particles, particles2, battrs=self.battrs, spin1=0, spin2=2)

        return pairs

        

        
    def compute_SS_treecorr(self,catalog1,catalog2):
        SS = np.zeros((len(self.bins1)-1,len(self.bins2)-1))
        SS2 = np.zeros((len(self.bins1)-1,len(self.bins2)-1))

        #wDS = np.zeros((len(self.bins1)-1,len(self.bins2)-1))
        for i in range(0,len(self.bins2)-1):
            pi_min = self.bins2[i]
            pi_max = self.bins2[i+1]
            ng = treecorr.GGCorrelation(bin_type='Log',nbins=len(self.bins1)-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop)
            ng.process(catalog1,catalog2)
            if self.computation == 'WPP':
                SS[:,i] = ng.weight*(ng.xip + ng.xim)/2.
            elif self.computation == 'WCC':
                SS[:,i] = ng.weight*(ng.xip- ng.xim)/2.
            elif self.computation == 'W_+':
                SS[:,i] = ng.weight*(ng.xip + ng.xim)/2.
                SS2[:,i] = ng.weight*(ng.xip - ng.xim)/2.


        return SS,SS2
        
    def compute_SS_cucount(self,catalog1,catalog2):
        """treecorr automatically converted (ra,dec) in radian"""
        
        ra = catalog1.ra
        dec = catalog1.dec
        r = catalog1.r
        w = catalog1.w
        e1 = catalog1.g1
        e2 = catalog1.g2

        ra2 = catalog2.ra
        dec2 = catalog2.dec
        r2 = catalog2.r
        w2 = catalog2.w
        e12 = catalog2.g1
        e22 = catalog2.g2

        particles =  self.get_shape_particles(ra,dec,r,w,e1,e2)
        particles2 =  self.get_shape_particles(ra2,dec2,r2,w2,e12,e22)
        wattrs = self.WeightAttrs(spin=(2, 2))

        PS_result = self.count2(particles, particles2, battrs=self.battrs, wattrs=wattrs)
        pairs_plus = PS_result['weight_plus_plus'].reshape(len(self.bins1)-1,len(self.bins2)-1)
        pairs_cross = PS_result['weight_cross_cross'].reshape(len(self.bins1)-1,len(self.bins2)-1)

        return pairs_plus,pairs_cross

        
    def compute_clustering_rp(self,catalog1,catalog2):
        ra = np.rad2deg(catalog1.ra)
        dec = np.rad2deg(catalog1.dec)
        r = catalog1.r
        w = catalog1.w
        pos = np.column_stack([ra,dec,r]).T

        ra2 = np.rad2deg(catalog2.ra)
        dec2 = np.rad2deg(catalog2.dec)
        r2 = catalog2.r
        w2 = catalog2.w
        pos2 = np.column_stack([ra2,dec2,r2]).T
        rp=self.compute_sepavg_pycorr(pos,w,pos2=pos2,weights2=w2)       
        return rp 

    
#     def compute_dsigma(self,catalog1,catalog2,catalog_r):

#         """compute dsigma with the dsigma python package. Since dsigma loops over each lense, we don't need to actually 
#         compute the signal per patch if we provide to the dsigma table the exact patch for each lenses"""
        
#         ra = np.rad2deg(catalog1.ra)
#         dec = np.rad2deg(catalog1.dec)
#         r = catalog1.r
#         w = catalog1.w
#         patch = catalog1.patch

#         ra2 = np.rad2deg(catalog2.ra)
#         dec2 = np.rad2deg(catalog2.dec)
#         r2 = catalog2.r
#         w2 = catalog2.w
#         patch2 = catalog2.patch
        
#         ra_r = np.rad2deg(catalog_r.ra)
#         dec_r = np.rad2deg(catalog_r.dec)
#         r_r = catalog2.r
#         w_r = catalog2.w
#         patch_r = catalog2.patch
        
#         T1 = Table([ra,dec,r,w,data1.patch],names=('ra','dec','z','w','patch'))
#         T2 = Table([ra_r,dec_r,z_r,w_r,rand1.patch],names=('ra','dec','z','w','patch'))
        
        
        

        
        
