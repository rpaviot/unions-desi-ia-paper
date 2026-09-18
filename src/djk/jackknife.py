#!/usr/bin/env python3

from corrd import *
from compute_pairs import compute_pairs
from scipy.special import lpmv
import math


def is_log_spaced(x, rtol=1e-3, atol=1e-6):
    lx = np.log(x)
    diffs = lx[1:] - lx[:-1]
    return np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)


# --- Legendre helpers ---
def L00(mu):
    """Legendre polynomial L_{0,0}(μ) = 1 (monopole, spin-0)"""
    return np.ones_like(mu)

def L20(mu):
    """Legendre polynomial L_{2,0}(μ) = (3μ²-1)/2 (quadrupole, spin-0)"""
    return (3.0 * mu**2 - 1.0) / 2.0

def L40(mu):
    """Legendre polynomial L_{4,0}(μ) = (35μ⁴-30μ²+3)/8 (hexadecapole, spin-0)"""
    return (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0

def L22(mu):
    """Associated Legendre polynomial L_{2,2}(μ) = 3 (1-μ²) (quadrupole, spin-2)"""
    return 3.0 * (1.0 - mu**2)

def L42(mu):
    """Associated Legendre polynomial L_{4,2}(μ) = (15/2) (1-μ^2)(7μ^2-1) (spin-2)"""
    return 15.0 / 2.0 * (1.0 - mu**2) * (7.0 * mu**2 - 1.0)


def compute_sky_patch_radius(ramin, ramax, decmin, decmax, ra_center, dec_center):
    """
    Compute approximate radius of a sky patch in radians.
    """
    Area = (ramax - ramin) * (np.sin(decmax) - np.sin(decmin))
    max_radius = np.sqrt(Area / np.pi)
    return max_radius


def great_circle_distance(ra1, dec1, ra2, dec2):
    """
    Compute great circle distance between two points on sphere.
    """
    costheta = np.sin(dec2) * np.sin(dec1) + np.cos(dec2) * np.cos(dec1) * np.cos(ra1 - ra2)
    costheta = np.clip(costheta, -1.0, 1.0)
    return np.arccos(costheta)
    

class jackknife():

    def __init__(self, **qwargs):
        # NEW: correction_cov parameter (used as default, can be overridden)
        if 'correction_cov' in qwargs:
            self.correction_cov = qwargs['correction_cov']
        else:
            self.correction_cov = False  # Default: no correction (alpha=1)
        
        self._pairs_computed = False
        self._totals_computed = False
        
        self.define_matrice()
        self.set_engine()
        self.pairs_counter = compute_pairs(self.computation, self.machine, self.bins1, self.bins2, self.bin_slop)

    def get_mohammad_alpha(self):
        """
        Compute Mohammad+21 correction factor (Eq. 27 of arXiv:2109.07071).
        
        α = n_sv / (2 + √2 * (n_sv - 1))
        
        This factor down-weights cross-pairs relative to auto-pairs
        to achieve correct variance scaling.
        
        Returns
        -------
        float
            The correction factor α. Returns 1.0 if Ns < 3.
        """
        n_sv = self.Ns
        if n_sv < 3:
            return 1.0
        return n_sv / (2.0 + np.sqrt(2) * (n_sv - 1))
    
    def define_matrice(self):
        if self.twoD is True:
            self.NN_ = np.zeros((self.comb, self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
            self.NN2_ = np.zeros((self.comb, self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
            self.NR2_ = np.zeros((self.comb, self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
            self.NR_ = np.zeros((self.comb, self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
            self.RR_ = np.zeros((self.comb, self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
            self.xi = np.zeros((self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
            self.xi2 = np.zeros((self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
        else:
            self.NN_ = np.zeros((self.comb, self.comb, len(self.bins1) - 1))
            self.NR_ = np.zeros((self.comb, self.comb, len(self.bins1) - 1))
            self.RR_ = np.zeros((self.comb, self.comb, len(self.bins1) - 1))
            self.xi = np.zeros((self.comb, len(self.bins1) - 1))

                
    def set_engine(self):
        if self.machine == 'GPU':
            self.pipeline = 'cucount'
            self.combine_all_wgp = self.combine_all_wgp_cucount

        elif self.machine == 'CPU':
            self.combine_all_wgp = self.combine_all_wgp_treecorr
            if self.computation in ['WGG', 'xil']:
                self.pipeline = 'pycorr'            
            elif self.computation in ['WGP', 'WPP', 'WCC', 'W_+']:
                self.pipeline = 'treecorr'
            
    def compute_all_pairs(self):
        self.compute_clustering_pairs()
        if self.computation in ['WGP', 'xi2p']:
            self.compute_ng_pairs()
        elif self.computation in ['WPP', 'WCC', 'W_+']:
            self.compute_nn_pairs()

    def compute_clustering_pairs(self):
        self.compute_auto_clustering_pairs()
        self.compute_cross_clustering_pairs()

    def compute_ng_pairs(self):
        self.compute_auto_ng_pairs()
        self.compute_cross_ng_pairs()
        
    def compute_nn_pairs(self):
        self.compute_auto_nn_pairs()
        self.compute_cross_nn_pairs()

    def compute_auto_clustering_pairs(self, calc_RR=True):
        for i in range(0, len(self.upatches)):
            patchi = self.upatches[i]
            data1i = self.data1_.patches[patchi]
            data2i = self.data2_.patches[patchi]
            rand1i = self.rand1_.patches[patchi]
            rand2i = self.rand2_.patches[patchi]

            if self.computation in ['xil', 'WGG']:
                DD = self.pairs_counter.compute_clustering(data1i)
                DR = self.pairs_counter.compute_clustering(data1i, rand1i)
                RR = self.pairs_counter.compute_clustering(rand1i)
                self.NN_[i][i] = DD
                self.NR_[i][i] = DR
                self.RR_[i][i] = RR

            elif self.computation in ['WGP', 'WPP', 'WCC', 'W_+', 'xi2p'] and (self.compute_RR == True):
                RR = self.pairs_counter.compute_clustering(rand1i, rand2i)
                self.RR_[i][i] = RR
            else:
                DR = self.pairs_counter.compute_clustering(rand1i, data2i)
                self.NR2_[i][i] = DR
                

    def compute_cross_clustering_pairs(self, calc_RR=True):
        """
        Compute clustering pairs between different patches.
        Uses angular filtering to skip distant patch pairs for speed.
        """
        from astropy.coordinates import cartesian_to_spherical
        
        n_skipped = 0
        n_computed = 0
        
        for patches in self.jackpairs:
            patchi, patchj = patches
            
            if hasattr(self, 'patch_centers') and hasattr(self, 'thetamax'):
                centeri = self.patch_centers[patchi]
                centerj = self.patch_centers[patchj]
                
                _, dec_patchi, ra_patchi = cartesian_to_spherical(centeri[0], centeri[1], centeri[2])
                _, dec_patchj, ra_patchj = cartesian_to_spherical(centerj[0], centerj[1], centerj[2])
                
                deci = dec_patchi.value
                decj = dec_patchj.value
                rai = ra_patchi.value
                raj = ra_patchj.value
                
                dist_theta = great_circle_distance(rai, deci, raj, decj)
                
                data1i = self.data1_.patches[patchi]
                data1j = self.data1_.patches[patchj]
                
                ra_mini = data1i.ra.min()
                ra_maxi = data1i.ra.max()
                dec_mini = data1i.dec.min()
                dec_maxi = data1i.dec.max()
                
                ra_minj = data1j.ra.min()
                ra_maxj = data1j.ra.max()
                dec_minj = data1j.dec.min()
                dec_maxj = data1j.dec.max()
                
                thetai = compute_sky_patch_radius(ra_mini, ra_maxi, dec_mini, dec_maxi, rai, deci)
                thetaj = compute_sky_patch_radius(ra_minj, ra_maxj, dec_minj, dec_maxj, raj, decj)
                
                theta_diff = dist_theta - thetai - thetaj
                
                if theta_diff >= np.sqrt(2) * self.thetamax:
                    n_skipped += 1
                    continue
            
            n_computed += 1
            
            if not (hasattr(self, 'patch_centers') and hasattr(self, 'thetamax')):
                data1i = self.data1_.patches[patchi]
                data1j = self.data1_.patches[patchj]
            
            data2j = self.data2_.patches[patchj]
            rand1i = self.rand1_.patches[patchi]
            rand1j = self.rand1_.patches[patchj]
            rand2j = self.rand2_.patches[patchj]
    
            if self.computation in ['xil', 'WGG']:
                self.NR_[patchi][patchj] = self.pairs_counter.compute_clustering(data1i, rand1j)
                self.RR_[patchi][patchj] = self.pairs_counter.compute_clustering(rand1i, rand1j)
                self.NN_[patchi][patchj] = self.pairs_counter.compute_clustering(data1i, data1j)
            elif self.computation in ['WGP', 'WPP', 'WCC', 'W_+', 'xi2p'] and (self.compute_RR == True):
                self.RR_[patchi][patchj] = self.pairs_counter.compute_clustering(rand1i, rand2j)
            else:
                DR = self.pairs_counter.compute_clustering(rand1i, data2j)
                self.NR2_[patchi][patchj] = DR
        
        if hasattr(self, 'patch_centers') and hasattr(self, 'thetamax'):
            total = n_computed + n_skipped
            print(f"Cross-patch clustering: computed {n_computed}/{total} pairs, " 
                  f"skipped {n_skipped} ({100*n_skipped/total:.1f}%)")
                
    
    def compute_auto_ng_pairs(self):
        for i in range(0, len(self.upatches)):
            patchi = self.upatches[i]
            data1i = self.data1_.patches[patchi]
            data2i = self.data2_.patches[patchi]
            rand1i = self.rand1_.patches[patchi]
            NN = self.pairs_counter.compute_DS(data1i, data2i)
            NR = self.pairs_counter.compute_DS(rand1i, data2i)
            self.NN_[i][i] = NN
            self.NR_[i][i] = NR

    def compute_cross_ng_pairs(self):
        """
        Compute galaxy-shear pairs between different patches.
        Uses angular filtering for speed.
        """
        from astropy.coordinates import cartesian_to_spherical
        
        n_skipped = 0
        n_computed = 0
        
        for patches in self.jackpairs:
            patchi, patchj = patches
    
            if hasattr(self, 'patch_centers') and hasattr(self, 'thetamax'):
                centeri = self.patch_centers[patchi]
                centerj = self.patch_centers[patchj]
                
                _, dec_patchi, ra_patchi = cartesian_to_spherical(centeri[0], centeri[1], centeri[2])
                _, dec_patchj, ra_patchj = cartesian_to_spherical(centerj[0], centerj[1], centerj[2])
                
                deci = dec_patchi.value
                decj = dec_patchj.value
                rai = ra_patchi.value
                raj = ra_patchj.value
                
                dist_theta = great_circle_distance(rai, deci, raj, decj)
                
                data1i = self.data1_.patches[patchi]
                data1j = self.data1_.patches[patchj]
                
                ra_mini = data1i.ra.min()
                ra_maxi = data1i.ra.max()
                dec_mini = data1i.dec.min()
                dec_maxi = data1i.dec.max()
                
                ra_minj = data1j.ra.min()
                ra_maxj = data1j.ra.max()
                dec_minj = data1j.dec.min()
                dec_maxj = data1j.dec.max()
                
                thetai = compute_sky_patch_radius(ra_mini, ra_maxi, dec_mini, dec_maxi, rai, deci)
                thetaj = compute_sky_patch_radius(ra_minj, ra_maxj, dec_minj, dec_maxj, raj, decj)
                
                theta_diff = dist_theta - thetai - thetaj
                
                if theta_diff >= np.sqrt(2) * self.thetamax:
                    n_skipped += 1
                    continue
            
            n_computed += 1
            
            if not (hasattr(self, 'patch_centers') and hasattr(self, 'thetamax')):
                data1i = self.data1_.patches[patchi]
            
            data2j = self.data2_.patches[patchj]
            rand1i = self.rand1_.patches[patchi]
    
            NN = self.pairs_counter.compute_DS(data1i, data2j)
            NR = self.pairs_counter.compute_DS(rand1i, data2j)
    
            self.NN_[patchi][patchj] = NN
            self.NR_[patchi][patchj] = NR
    
        if hasattr(self, 'patch_centers') and hasattr(self, 'thetamax'):
            total = n_computed + n_skipped
            print(f"Cross-patch GGL: computed {n_computed}/{total} pairs, "
                  f"skipped {n_skipped} ({100*n_skipped/total:.1f}%)")

   
    def compute_auto_nn_pairs(self):
        for i in range(0, len(self.upatches)):
            patchi = self.upatches[i]
            data1i = self.data1_.patches[patchi]
            data2i = self.data2_.patches[patchi]
            NN, NN2 = self.pairs_counter.compute_SS(data1i, data2i)
            self.NN_[i][i] = NN
            self.NN2_[i][i] = NN2

    def compute_cross_nn_pairs(self):
        """
        Compute shear-shear pairs between different patches.
        Uses angular filtering for speed.
        """
        from astropy.coordinates import cartesian_to_spherical
        
        n_skipped = 0
        n_computed = 0
        
        for patches in self.jackpairs:
            patchi, patchj = patches
            
            if hasattr(self, 'patch_centers') and hasattr(self, 'thetamax'):
                centeri = self.patch_centers[patchi]
                centerj = self.patch_centers[patchj]
                
                _, dec_patchi, ra_patchi = cartesian_to_spherical(centeri[0], centeri[1], centeri[2])
                _, dec_patchj, ra_patchj = cartesian_to_spherical(centerj[0], centerj[1], centerj[2])
                
                deci = dec_patchi.value
                decj = dec_patchj.value
                rai = ra_patchi.value
                raj = ra_patchj.value
                
                dist_theta = great_circle_distance(rai, deci, raj, decj)
                
                data1i = self.data1_.patches[patchi]
                data1j = self.data1_.patches[patchj]
                
                ra_mini = data1i.ra.min()
                ra_maxi = data1i.ra.max()
                dec_mini = data1i.dec.min()
                dec_maxi = data1i.dec.max()
                
                ra_minj = data1j.ra.min()
                ra_maxj = data1j.ra.max()
                dec_minj = data1j.dec.min()
                dec_maxj = data1j.dec.max()
                
                thetai = compute_sky_patch_radius(ra_mini, ra_maxi, dec_mini, dec_maxi, rai, deci)
                thetaj = compute_sky_patch_radius(ra_minj, ra_maxj, dec_minj, dec_maxj, raj, decj)
                
                theta_diff = dist_theta - thetai - thetaj
                
                if theta_diff >= np.sqrt(2) * self.thetamax:
                    n_skipped += 1
                    continue
            
            n_computed += 1
            
            if not (hasattr(self, 'patch_centers') and hasattr(self, 'thetamax')):
                data1i = self.data1_.patches[patchi]
            
            data2j = self.data2_.patches[patchj]
            
            NN, NN2 = self.pairs_counter.compute_SS(data1i, data2j)
            self.NN_[patchi][patchj] = NN
            self.NN2_[patchi][patchj] = NN2
        
        if hasattr(self, 'patch_centers') and hasattr(self, 'thetamax'):
            total = n_computed + n_skipped
            print(f"Cross-patch shear-shear: computed {n_computed}/{total} pairs, "
                  f"skipped {n_skipped} ({100*n_skipped/total:.1f}%)")

    # =========================================================================
    # Methods for Mohammad+21 correction
    # =========================================================================
    
    def compute_total_pairs(self):
        """
        Compute total pair counts summed over ALL patches.
        This is done once before the jackknife loop.
        """
        self.NN_total = np.sum(self.NN_, axis=(0, 1))
        self.NR_total = np.sum(self.NR_, axis=(0, 1))
        self.RR_total = np.sum(self.RR_, axis=(0, 1))
        
        if hasattr(self, 'NN2_') and np.any(self.NN2_):
            self.NN2_total = np.sum(self.NN2_, axis=(0, 1))
        
        if hasattr(self, 'NR2_') and np.any(self.NR2_):
            self.NR2_total = np.sum(self.NR2_, axis=(0, 1))
    
    def compute_total_norm(self):
        """
        Compute normalization using ALL patches.
        Follows the same logic as compute_kpatch_norm but for all patches.
        """
        w_d1 = self.data1_.w
        w_r1 = self.rand1_.w
        w_d2 = self.data2_.w
        w_r2 = self.rand2_.w
        
        if self.corr == 'auto':
            self.normDD_total = np.sum(w_d1)**2 - np.sum(w_d1**2)
            self.normDR_total = np.sum(w_d1) * np.sum(w_r1)
            self.normDR2_total = np.sum(w_d2) * np.sum(w_r1)
            self.normRR_total = np.sum(w_r1)**2 - np.sum(w_r1**2)
            
        elif self.corr == 'cross':
            self.normDD_total = np.sum(w_d2) * np.sum(w_d1)
            self.normDR_total = np.sum(w_d2) * np.sum(w_r1)
            self.normDR2_total = np.sum(w_d2) * np.sum(w_r1)
            if self.cross_RR == False:
                self.normRR_total = np.sum(w_r1)**2 - np.sum(w_r1**2)
            else:
                self.normRR_total = np.sum(w_r1) * np.sum(w_r2)
                
        elif self.corr == 'subsample':
            self.normDR_total = np.sum(w_d2) * np.sum(w_r1)
            self.normDR2_total = np.sum(w_d2) * np.sum(w_r1)
            
            # Use the pre-computed intersection weights from read_input
            self.normDD_total = np.sum(w_d1) * np.sum(w_d2) - np.sum(self.w_int1 * self.w_int2)
            
            if self.cross_RR == False:
                self.normRR_total = np.sum(w_r1)**2 - np.sum(w_r1**2)
            else:
                self.normRR_total = np.sum(w_r1) * np.sum(w_r2)

    def get_excluded_pair_counts(self, excluded_indices, alpha=1.0):
        """
        Compute the pair counts to SUBTRACT for a given set of excluded patches.
        
        Following pycorr/Mohammad+21:
        - Auto-pairs within excluded region: subtract fully (weight = 1)
        - Cross-pairs between excluded and kept: subtract with weight α
        
        Parameters
        ----------
        excluded_indices : array-like
            Indices of patches that are EXCLUDED (removed) in this realization
        alpha : float
            Mohammad correction factor. α=1 means standard jackknife,
            α<1 means cross-pairs are down-weighted.
            
        Returns
        -------
        dict
            Dictionary with keys 'NN', 'NR', 'RR', etc. containing the 
            pair counts to subtract.
        """
        excluded = np.array(excluded_indices, dtype=int)
        all_patches = set(self.upatches)
        kept = np.array(list(all_patches - set(excluded)), dtype=int)
        
        result = {}
        
        if len(excluded) == 0:
            # Nothing to subtract
            result['NN'] = 0
            result['NR'] = 0
            result['RR'] = 0
            if hasattr(self, 'NN2_') and np.any(self.NN2_):
                result['NN2'] = 0
            if hasattr(self, 'NR2_') and np.any(self.NR2_):
                result['NR2'] = 0
            return result
        
        # --- Auto-pairs within excluded region (subtract fully) ---
        NN_auto_exc = np.sum(self.NN_[np.ix_(excluded, excluded)], axis=(0, 1))
        NR_auto_exc = np.sum(self.NR_[np.ix_(excluded, excluded)], axis=(0, 1))
        RR_auto_exc = np.sum(self.RR_[np.ix_(excluded, excluded)], axis=(0, 1))
        
        # --- Cross-pairs between excluded and kept (subtract with weight α) ---
        if len(kept) > 0:
            # excluded -> kept
            NN_cross = np.sum(self.NN_[np.ix_(excluded, kept)], axis=(0, 1))
            # kept -> excluded  
            NN_cross += np.sum(self.NN_[np.ix_(kept, excluded)], axis=(0, 1))
            
            NR_cross = np.sum(self.NR_[np.ix_(excluded, kept)], axis=(0, 1))
            NR_cross += np.sum(self.NR_[np.ix_(kept, excluded)], axis=(0, 1))
            
            RR_cross = np.sum(self.RR_[np.ix_(excluded, kept)], axis=(0, 1))
            RR_cross += np.sum(self.RR_[np.ix_(kept, excluded)], axis=(0, 1))
        else:
            NN_cross = 0
            NR_cross = 0
            RR_cross = 0
        
        # Total to subtract: auto (full) + cross (weighted by α)
        result['NN'] = NN_auto_exc + alpha * NN_cross
        result['NR'] = NR_auto_exc + alpha * NR_cross
        result['RR'] = RR_auto_exc + alpha * RR_cross
        
        # Handle NN2 for WPP/WCC/W_+ computations
        if hasattr(self, 'NN2_') and np.any(self.NN2_):
            NN2_auto_exc = np.sum(self.NN2_[np.ix_(excluded, excluded)], axis=(0, 1))
            if len(kept) > 0:
                NN2_cross = np.sum(self.NN2_[np.ix_(excluded, kept)], axis=(0, 1))
                NN2_cross += np.sum(self.NN2_[np.ix_(kept, excluded)], axis=(0, 1))
            else:
                NN2_cross = 0
            result['NN2'] = NN2_auto_exc + alpha * NN2_cross
        
        # Handle NR2 if used
        if hasattr(self, 'NR2_') and np.any(self.NR2_):
            NR2_auto_exc = np.sum(self.NR2_[np.ix_(excluded, excluded)], axis=(0, 1))
            if len(kept) > 0:
                NR2_cross = np.sum(self.NR2_[np.ix_(excluded, kept)], axis=(0, 1))
                NR2_cross += np.sum(self.NR2_[np.ix_(kept, excluded)], axis=(0, 1))
            else:
                NR2_cross = 0
            result['NR2'] = NR2_auto_exc + alpha * NR2_cross
        
        return result

    def get_excluded_norm(self, excluded_indices, alpha=1.0):
        """
        Compute the normalization to SUBTRACT for excluded patches.
        
        Same logic as pair counts:
        - Auto-norm within excluded: subtract fully (weight = 1)
        - Cross-norm between excluded and kept: subtract with weight α
        
        Parameters
        ----------
        excluded_indices : array-like
            Indices of patches that are EXCLUDED
        alpha : float
            Mohammad correction factor
            
        Returns
        -------
        dict
            Dictionary with normDD, normDR, normRR, etc. to subtract
        """
        excluded = np.array(excluded_indices, dtype=int)
        all_patches = set(self.upatches)
        kept = np.array(list(all_patches - set(excluded)), dtype=int)
        
        result = {}
        
        if len(excluded) == 0:
            result['normDD'] = 0
            result['normDR'] = 0
            result['normDR2'] = 0
            result['normRR'] = 0
            return result
        
        # Get weights for excluded patches
        w_d1_exc = np.hstack([self.data1_.patches[x].w for x in excluded])
        w_r1_exc = np.hstack([self.rand1_.patches[x].w for x in excluded])
        w_d2_exc = np.hstack([self.data2_.patches[x].w for x in excluded])
        w_r2_exc = np.hstack([self.rand2_.patches[x].w for x in excluded])
        
        # Get weights for kept patches
        if len(kept) > 0:
            w_d1_kept = np.hstack([self.data1_.patches[x].w for x in kept])
            w_r1_kept = np.hstack([self.rand1_.patches[x].w for x in kept])
            w_d2_kept = np.hstack([self.data2_.patches[x].w for x in kept])
            w_r2_kept = np.hstack([self.rand2_.patches[x].w for x in kept])
        else:
            w_d1_kept = np.array([])
            w_r1_kept = np.array([])
            w_d2_kept = np.array([])
            w_r2_kept = np.array([])
        
        # Sums for convenience
        sum_w_d1_exc = np.sum(w_d1_exc)
        sum_w_r1_exc = np.sum(w_r1_exc)
        sum_w_d2_exc = np.sum(w_d2_exc)
        sum_w_r2_exc = np.sum(w_r2_exc)
        
        sum_w_d1_kept = np.sum(w_d1_kept)
        sum_w_r1_kept = np.sum(w_r1_kept)
        sum_w_d2_kept = np.sum(w_d2_kept)
        sum_w_r2_kept = np.sum(w_r2_kept)
        
        if self.corr == 'auto':
            # --- Auto normalization (within excluded) ---
            # normDD_auto = (Σ w_exc)² - Σ w_exc²
            normDD_auto = sum_w_d1_exc**2 - np.sum(w_d1_exc**2)
            normDR_auto = sum_w_d1_exc * sum_w_r1_exc
            normDR2_auto = sum_w_d2_exc * sum_w_r1_exc
            normRR_auto = sum_w_r1_exc**2 - np.sum(w_r1_exc**2)
            
            # --- Cross normalization (between excluded and kept) ---
            # normDD_cross = 2 × Σ w_exc × Σ w_kept
            normDD_cross = 2 * sum_w_d1_exc * sum_w_d1_kept
            normDR_cross = sum_w_d1_exc * sum_w_r1_kept + sum_w_d1_kept * sum_w_r1_exc
            normDR2_cross = sum_w_d2_exc * sum_w_r1_kept + sum_w_d2_kept * sum_w_r1_exc
            normRR_cross = 2 * sum_w_r1_exc * sum_w_r1_kept
            
            result['normDD'] = normDD_auto + alpha * normDD_cross
            result['normDR'] = normDR_auto + alpha * normDR_cross
            result['normDR2'] = normDR2_auto + alpha * normDR2_cross
            result['normRR'] = normRR_auto + alpha * normRR_cross
            
        elif self.corr == 'cross':
            # For cross-correlation: normDD = Σ_1 w × Σ_2 w
            
            # Auto: both in excluded
            normDD_auto = sum_w_d1_exc * sum_w_d2_exc
            normDR_auto = sum_w_d2_exc * sum_w_r1_exc
            normDR2_auto = sum_w_d2_exc * sum_w_r1_exc
            
            # Cross: one in excluded, one in kept
            normDD_cross = sum_w_d1_exc * sum_w_d2_kept + sum_w_d1_kept * sum_w_d2_exc
            normDR_cross = sum_w_d2_exc * sum_w_r1_kept + sum_w_d2_kept * sum_w_r1_exc
            normDR2_cross = sum_w_d2_exc * sum_w_r1_kept + sum_w_d2_kept * sum_w_r1_exc
            
            result['normDD'] = normDD_auto + alpha * normDD_cross
            result['normDR'] = normDR_auto + alpha * normDR_cross
            result['normDR2'] = normDR2_auto + alpha * normDR2_cross
            
            if self.cross_RR == False:
                normRR_auto = sum_w_r1_exc**2 - np.sum(w_r1_exc**2)
                normRR_cross = 2 * sum_w_r1_exc * sum_w_r1_kept
            else:
                normRR_auto = sum_w_r1_exc * sum_w_r2_exc
                normRR_cross = sum_w_r1_exc * sum_w_r2_kept + sum_w_r1_kept * sum_w_r2_exc
            
            result['normRR'] = normRR_auto + alpha * normRR_cross
            
        elif self.corr == 'subsample':
            # For subsample: need to handle intersection terms using complex number hash
            
            # Get coordinates for excluded patches
            ra_d1_exc = np.hstack([self.data1_.patches[x].ra for x in excluded])
            dec_d1_exc = np.hstack([self.data1_.patches[x].dec for x in excluded])
            ra_d2_exc = np.hstack([self.data2_.patches[x].ra for x in excluded])
            dec_d2_exc = np.hstack([self.data2_.patches[x].dec for x in excluded])
            
            # Get coordinates for kept patches
            if len(kept) > 0:
                ra_d1_kept = np.hstack([self.data1_.patches[x].ra for x in kept])
                dec_d1_kept = np.hstack([self.data1_.patches[x].dec for x in kept])
                ra_d2_kept = np.hstack([self.data2_.patches[x].ra for x in kept])
                dec_d2_kept = np.hstack([self.data2_.patches[x].dec for x in kept])
            else:
                ra_d1_kept = np.array([])
                dec_d1_kept = np.array([])
                ra_d2_kept = np.array([])
                dec_d2_kept = np.array([])
            
            # --- normDR (simpler, no intersection) ---
            normDR_auto = sum_w_d2_exc * sum_w_r1_exc
            normDR_cross = sum_w_d2_exc * sum_w_r1_kept + sum_w_d2_kept * sum_w_r1_exc
            
            result['normDR'] = normDR_auto + alpha * normDR_cross
            result['normDR2'] = result['normDR']
            
            # --- normDD with intersection ---
            # Auto: intersection within excluded
            posC_hash_exc = ra_d1_exc + 1j * dec_d1_exc
            pos_hash_exc = ra_d2_exc + 1j * dec_d2_exc
            _, xind_exc, yind_exc = np.intersect1d(posC_hash_exc, pos_hash_exc, return_indices=True)
            w_int1_exc = w_d1_exc[xind_exc]
            w_int2_exc = w_d2_exc[yind_exc]
            
            normDD_auto = sum_w_d1_exc * sum_w_d2_exc - np.sum(w_int1_exc * w_int2_exc)
            
            # Cross: intersection between excluded and kept
            # We need intersections: (exc_d1 with kept_d2) and (kept_d1 with exc_d2)
            normDD_cross_raw = sum_w_d1_exc * sum_w_d2_kept + sum_w_d1_kept * sum_w_d2_exc
            
            if len(kept) > 0:
                # Intersection: exc_d1 with kept_d2
                posC_hash_exc_d1 = ra_d1_exc + 1j * dec_d1_exc
                pos_hash_kept_d2 = ra_d2_kept + 1j * dec_d2_kept
                _, xind_ek, yind_ek = np.intersect1d(posC_hash_exc_d1, pos_hash_kept_d2, return_indices=True)
                w_int_exc_kept = np.sum(w_d1_exc[xind_ek] * w_d2_kept[yind_ek]) if len(xind_ek) > 0 else 0
                
                # Intersection: kept_d1 with exc_d2
                posC_hash_kept_d1 = ra_d1_kept + 1j * dec_d1_kept
                pos_hash_exc_d2 = ra_d2_exc + 1j * dec_d2_exc
                _, xind_ke, yind_ke = np.intersect1d(posC_hash_kept_d1, pos_hash_exc_d2, return_indices=True)
                w_int_kept_exc = np.sum(w_d1_kept[xind_ke] * w_d2_exc[yind_ke]) if len(xind_ke) > 0 else 0
                
                normDD_cross = normDD_cross_raw - w_int_exc_kept - w_int_kept_exc
            else:
                normDD_cross = 0
            
            result['normDD'] = normDD_auto + alpha * normDD_cross
            
            # --- normRR ---
            if self.cross_RR == False:
                normRR_auto = sum_w_r1_exc**2 - np.sum(w_r1_exc**2)
                normRR_cross = 2 * sum_w_r1_exc * sum_w_r1_kept
            else:
                # Get random coordinates for excluded patches
                ra_r1_exc = np.hstack([self.rand1_.patches[x].ra for x in excluded])
                dec_r1_exc = np.hstack([self.rand1_.patches[x].dec for x in excluded])
                ra_r2_exc = np.hstack([self.rand2_.patches[x].ra for x in excluded])
                dec_r2_exc = np.hstack([self.rand2_.patches[x].dec for x in excluded])
                
                # Get random coordinates for kept patches
                if len(kept) > 0:
                    ra_r1_kept = np.hstack([self.rand1_.patches[x].ra for x in kept])
                    dec_r1_kept = np.hstack([self.rand1_.patches[x].dec for x in kept])
                    ra_r2_kept = np.hstack([self.rand2_.patches[x].ra for x in kept])
                    dec_r2_kept = np.hstack([self.rand2_.patches[x].dec for x in kept])
                else:
                    ra_r1_kept = np.array([])
                    dec_r1_kept = np.array([])
                    ra_r2_kept = np.array([])
                    dec_r2_kept = np.array([])
                
                # Auto: intersection within excluded randoms
                posC_hash_r_exc = ra_r1_exc + 1j * dec_r1_exc
                pos_hash_r_exc = ra_r2_exc + 1j * dec_r2_exc
                _, xind_r_exc, yind_r_exc = np.intersect1d(posC_hash_r_exc, pos_hash_r_exc, return_indices=True)
                w_int_r1_exc = w_r1_exc[xind_r_exc] if len(xind_r_exc) > 0 else np.array([])
                w_int_r2_exc = w_r2_exc[yind_r_exc] if len(yind_r_exc) > 0 else np.array([])
                
                normRR_auto = sum_w_r1_exc * sum_w_r2_exc - np.sum(w_int_r1_exc * w_int_r2_exc)
                
                # Cross: intersection between excluded and kept randoms
                normRR_cross_raw = sum_w_r1_exc * sum_w_r2_kept + sum_w_r1_kept * sum_w_r2_exc
                normRR_cross= normRR_cross_raw
                # if len(kept) > 0:
                #     # Intersection: exc_r1 with kept_r2
                #     pos_hash_r_kept_2 = ra_r2_kept + 1j * dec_r2_kept
                #     _, xind_r_ek, yind_r_ek = np.intersect1d(posC_hash_r_exc, pos_hash_r_kept_2, return_indices=True)
                #     w_int_r_exc_kept = np.sum(w_r1_exc[xind_r_ek] * w_r2_kept[yind_r_ek]) if len(xind_r_ek) > 0 else 0
                    
                #     # Intersection: kept_r1 with exc_r2
                #     posC_hash_r_kept_1 = ra_r1_kept + 1j * dec_r1_kept
                #     _, xind_r_ke, yind_r_ke = np.intersect1d(posC_hash_r_kept_1, pos_hash_r_exc, return_indices=True)
                #     w_int_r_kept_exc = np.sum(w_r1_kept[xind_r_ke] * w_r2_exc[yind_r_ke]) if len(xind_r_ke) > 0 else 0
                    
                #     normRR_cross = normRR_cross_raw - w_int_r_exc_kept - w_int_r_kept_exc
                # else:
                #     normRR_cross = 0
            
            result['normRR'] = normRR_auto + alpha * normRR_cross
        
        return result

    # =========================================================================
    # Combination methods — now accept correction_cov parameter
    # =========================================================================

    def _resolve_alpha(self, correction_cov):
        """
        Resolve the alpha value from a correction_cov flag.
        
        Parameters
        ----------
        correction_cov : bool or None
            If None, uses self.correction_cov.
            If True/False, overrides the default.
        
        Returns
        -------
        float
            The α correction factor.
        """
        if correction_cov is None:
            correction_cov = self.correction_cov
        
        if correction_cov:
            alpha = self.get_mohammad_alpha()
            print(f"Using Mohammad+21 correction with α = {alpha:.4f}")
        else:
            alpha = 1.0
        
        return alpha

    def _ensure_totals(self):
        """Compute total pairs and norms if not already done."""
        if not self._totals_computed:
            self.compute_total_pairs()
            self.compute_total_norm()
            self._totals_computed = True

    def combine_all_clustering(self, correction_cov=None):
        """
        Combine clustering pairs using Mohammad+21 correction if enabled.
        
        Parameters
        ----------
        correction_cov : bool or None
            If None, uses self.correction_cov.
        """
        alpha = self._resolve_alpha(correction_cov)
        self._ensure_totals()
        
        # Reset xi arrays for fresh combination
        if self.twoD is True:
            self.xi = np.zeros((self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
        else:
            self.xi = np.zeros((self.comb, len(self.bins1) - 1))
        
        for i in range(len(self.array_comb)):
            indices_kept = self.array_comb[i]  # Patches KEPT in this realization
            indices_excluded = np.setdiff1d(self.upatches, indices_kept)  # Patches REMOVED
            
            # Get pairs and norms to subtract (both with α weighting)
            subtract_pairs = self.get_excluded_pair_counts(indices_excluded, alpha=alpha)
            subtract_norms = self.get_excluded_norm(indices_excluded, alpha=alpha)
            
            # Compute jackknife realization: total - excluded
            DDpairs = self.NN_total - subtract_pairs['NN']
            DRpairs = self.NR_total - subtract_pairs['NR']
            RRpairs = self.RR_total - subtract_pairs['RR']
            
            # Normalizations also corrected
            normDD = self.normDD_total - subtract_norms['normDD']
            normDR = self.normDR_total - subtract_norms['normDR']
            normRR = self.normRR_total - subtract_norms['normRR']
            
            # Compute xi using Landy-Szalay
            xi = (DDpairs / RRpairs) * (normRR / normDD) \
                 - 2 * (DRpairs / RRpairs) * (normRR / normDR) + 1.
            
            self.xi[i] = xi

    def combine_all_wgp_treecorr(self, correction_cov=None):
        """
        Combine WGP pairs using Mohammad+21 correction if enabled.
        
        Parameters
        ----------
        correction_cov : bool or None
            If None, uses self.correction_cov.
        """
        alpha = self._resolve_alpha(correction_cov)
        self._ensure_totals()
        
        # Reset xi arrays
        if self.twoD is True:
            self.xi = np.zeros((self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
        else:
            self.xi = np.zeros((self.comb, len(self.bins1) - 1))
        
        for i in range(len(self.array_comb)):
            indices_kept = self.array_comb[i]
            indices_excluded = np.setdiff1d(self.upatches, indices_kept)
            
            # Get pairs and norms to subtract
            subtract_pairs = self.get_excluded_pair_counts(indices_excluded, alpha=alpha)
            subtract_norms = self.get_excluded_norm(indices_excluded, alpha=alpha)
            
            # Jackknife realization: total - excluded
            NGxi = self.NN_total - subtract_pairs['NN']
            RGxi = self.NR_total - subtract_pairs['NR']
            
            # Normalizations
            normDD = self.normDD_total - subtract_norms['normDD']
            normDR = self.normDR_total - subtract_norms['normDR']
            
            if self.compute_RR == True:
                RRpairs = self.RR_total - subtract_pairs['RR']
                normRR = self.normRR_total - subtract_norms['normRR']
                xi = NGxi / RRpairs * (normRR / normDD) \
                     - RGxi / RRpairs * (normRR / normDR)
            else:
                RGpairs2 = self.NR2_total - subtract_pairs.get('NR2', 0)
                normDR2 = self.normDR2_total - subtract_norms.get('normDR2', subtract_norms['normDR'])
                xi = NGxi / RGpairs2 * (normDR2 / normDD) \
                     - RGxi / RGpairs2 * (normDR2 / normDR)
            
            self.xi[i] = xi
    
    def combine_all_wgp_cucount(self, correction_cov=None):
        """
        Combine WGP pairs (GPU version) using Mohammad+21 correction if enabled.
        
        Parameters
        ----------
        correction_cov : bool or None
            If None, uses self.correction_cov.
        """
        alpha = self._resolve_alpha(correction_cov)
        self._ensure_totals()
        
        # Reset xi arrays
        if self.twoD is True:
            self.xi = np.zeros((self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
        else:
            self.xi = np.zeros((self.comb, len(self.bins1) - 1))
        
        for i in range(len(self.array_comb)):
            indices_kept = self.array_comb[i]
            indices_excluded = np.setdiff1d(self.upatches, indices_kept)
            
            subtract_pairs = self.get_excluded_pair_counts(indices_excluded, alpha=alpha)
            subtract_norms = self.get_excluded_norm(indices_excluded, alpha=alpha)
            
            DSpairs = self.NN_total - subtract_pairs['NN']
            RSpairs = self.NR_total - subtract_pairs['NR']
            
            normDD = self.normDD_total - subtract_norms['normDD']
            normDR = self.normDR_total - subtract_norms['normDR']
            
            if self.compute_RR == True:
                RRpairs = self.RR_total - subtract_pairs['RR']
                normRR = self.normRR_total - subtract_norms['normRR']
                xi = DSpairs / RRpairs * (normRR / normDD) \
                     - RSpairs / RRpairs * (normRR / normDR)
            else:
                RGpairs2 = self.NR2_total - subtract_pairs.get('NR2', 0)
                normDR2 = self.normDR2_total - subtract_norms.get('normDR2', subtract_norms['normDR'])
                xi = DSpairs / RGpairs2 * (normDR2 / normDD) \
                     - RSpairs / RGpairs2 * (normDR2 / normDR)
            
            self.xi[i] = xi

    def combine_all_wpp(self, correction_cov=None):
        """
        Combine WPP pairs using Mohammad+21 correction if enabled.
        
        Parameters
        ----------
        correction_cov : bool or None
            If None, uses self.correction_cov.
        """
        alpha = self._resolve_alpha(correction_cov)
        self._ensure_totals()
        
        # Reset xi arrays
        if self.twoD is True:
            self.xi = np.zeros((self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
            self.xi2 = np.zeros((self.comb, len(self.bins1) - 1, len(self.bins2) - 1))
        else:
            self.xi = np.zeros((self.comb, len(self.bins1) - 1))
        
        for i in range(len(self.array_comb)):
            indices_kept = self.array_comb[i]
            indices_excluded = np.setdiff1d(self.upatches, indices_kept)
            
            subtract_pairs = self.get_excluded_pair_counts(indices_excluded, alpha=alpha)
            subtract_norms = self.get_excluded_norm(indices_excluded, alpha=alpha)
            
            NGxi = self.NN_total - subtract_pairs['NN']
            NGxi2 = self.NN2_total - subtract_pairs.get('NN2', 0)
            RRpairs = self.RR_total - subtract_pairs['RR']
            
            normDD = self.normDD_total - subtract_norms['normDD']
            normRR = self.normRR_total - subtract_norms['normRR']
            
            xi = NGxi / RRpairs * (normRR / normDD)
            xi2 = NGxi2 / RRpairs * (normRR / normDD)
            
            self.xi[i] = xi
            self.xi2[i] = xi2
            
    def compute_kpatch_norm(self, indices, corr='auto'):
        """
        Compute normalization for a subset of patches (kept patches).
        This method is kept for backward compatibility but is not used
        when correction_cov is enabled.
        """
        w_d1 = np.hstack([self.data1_.patches[x].w for x in indices.astype('int')])
        w_r1 = np.hstack([self.rand1_.patches[x].w for x in indices.astype('int')])
        w_d2 = np.hstack([self.data2_.patches[x].w for x in indices.astype('int')])
        w_r2 = np.hstack([self.rand2_.patches[x].w for x in indices.astype('int')])
        
        if corr == 'auto':
            self.normDD = np.sum(w_d1)**2 - np.sum(w_d1**2)
            self.normDR = np.sum(w_d1) * np.sum(w_r1)
            self.normDR2 = np.sum(w_d2) * np.sum(w_r1)
            self.normRR = np.sum(w_r1)**2 - np.sum(w_r1**2)
            
        elif corr == 'cross':
            self.normDD = np.sum(w_d2) * np.sum(w_d1)
            self.normDR = np.sum(w_d2) * np.sum(w_r1)
            self.normDR2 = np.sum(w_d2) * np.sum(w_r1)
            if self.cross_RR == False:
                self.normRR = np.sum(w_r1)**2 - np.sum(w_r1**2)
            else:
                self.normRR = np.sum(w_r1) * np.sum(w_r2)
                
        elif corr == 'subsample':
            self.normDR = np.sum(w_d2) * np.sum(w_r1)
            self.normDR2 = np.sum(w_d2) * np.sum(w_r1)
            
            ra_d1 = np.hstack([self.data1_.patches[x].ra for x in indices.astype('int')])
            dec_d1 = np.hstack([self.data1_.patches[x].dec for x in indices.astype('int')])
            ra_d2 = np.hstack([self.data2_.patches[x].ra for x in indices.astype('int')])
            dec_d2 = np.hstack([self.data2_.patches[x].dec for x in indices.astype('int')])
            
            posC_hash = ra_d1 + 1j * dec_d1
            pos_hash = ra_d2 + 1j * dec_d2
            
            _, xind, yind = np.intersect1d(posC_hash, pos_hash, return_indices=True)
            w_int1 = w_d1[xind]
            w_int2 = w_d2[yind]
                              
            self.normDD = np.sum(w_d1) * np.sum(w_d2) - np.sum(w_int1 * w_int2)
            
            if self.cross_RR == False:
                self.normRR = (np.sum(w_r1)**2 - np.sum(w_r1**2))
            else:
                ra_r1 = np.hstack([self.rand1_.patches[x].ra for x in indices.astype('int')])
                dec_r1 = np.hstack([self.rand1_.patches[x].dec for x in indices.astype('int')])
                ra_r2 = np.hstack([self.rand2_.patches[x].ra for x in indices.astype('int')])
                dec_r2 = np.hstack([self.rand2_.patches[x].dec for x in indices.astype('int')])
                
                posC_hash = ra_r1 + 1j * dec_r1
                pos_hash = ra_r2 + 1j * dec_r2
                
                _, xind, yind = np.intersect1d(posC_hash, pos_hash, return_indices=True)
                w_int1 = w_r1[xind]
                w_int2 = w_r2[yind]
                
                self.normRR = np.sum(w_r1) * np.sum(w_r2) - np.sum(w_int1 * w_int2)


    def compute_correlation_multipoles(self, xi, sab, ells=(0, 2, 4), rp_cut=None):
        """
        Compute correlation function multipoles for (s,μ) binning.
        If oversampling is used, properly rebins using RR pair counts as weights.
        """
        import numpy as _np
        import math as _math
    
        mu_centers = (self.bins2[:-1] + self.bins2[1:]) / 2.0
        dmu = np.diff(self.bins2)
        s_centers_fine = self._get_bin_centers(self.bins1)
        
        n_jack = xi.shape[0]
        n_s_fine = len(s_centers_fine)
        n_mu = len(mu_centers)
    
        use_rebinning = hasattr(self, 'bins1_coarse') and len(self.bins1) != len(self.bins1_coarse)
        
        if use_rebinning:
            s_centers_coarse = self._get_bin_centers(self.bins1_coarse)
            n_s_coarse = len(s_centers_coarse)
            fine_to_coarse = self._get_fine_to_coarse_mapping(self.bins1, self.bins1_coarse)
            
            RR_sum = np.sum(self.RR_, axis=(0, 1))
            RR_s = np.sum(RR_sum, axis=1)
        else:
            n_s_coarse = n_s_fine
            fine_to_coarse = None
    
        mu_grid = mu_centers[np.newaxis, :]
        s_grid_fine = s_centers_fine[:, np.newaxis]
        rp_grid = s_grid_fine * _np.sqrt(_np.maximum(0.0, 1.0 - mu_grid**2))
    
        if rp_cut is None:
            rp_mask = _np.ones_like(rp_grid, dtype=float)
        else:
            rp_mask = (rp_grid >= rp_cut).astype(float)
    
        Lfuncs_sab0 = {0: L00, 2: L20, 4: L40}
        Lfuncs_sab2 = {2: L22, 4: L42}
    
        xiell_list = []
    
        for ell in ells:
            if sab > ell:
                print(f"  Warning: sab={sab} > ell={ell}, skipping multipole ℓ={ell}")
                continue
    
            if sab == 0:
                if ell not in Lfuncs_sab0:
                    raise ValueError(f"Unsupported ell={ell} for sab=0")
                P_ell = Lfuncs_sab0[ell](mu_centers)
            elif sab == 2:
                if ell not in Lfuncs_sab2:
                    raise ValueError(f"Unsupported ell={ell} for sab=2")
                P_ell = Lfuncs_sab2[ell](mu_centers)
            else:
                raise ValueError("sab must be 0 or 2")
    
            if sab == 0:
                factor = (2 * ell + 1) / 2.0
            else:
                factor = ((2 * ell + 1) / 2.0) * (_math.factorial(ell - sab) / _math.factorial(ell + sab))
    
            P_ell_b = P_ell[np.newaxis, np.newaxis, :]
            dmu_b = dmu[np.newaxis, np.newaxis, :]
            mask_b = rp_mask[np.newaxis, :, :]
    
            multipole_fine = factor * _np.sum(xi * P_ell_b * dmu_b * mask_b, axis=2)
    
            if use_rebinning:
                multipole_coarse = self._rebin_to_coarse_weighted(
                    multipole_fine, fine_to_coarse, n_s_coarse, RR_s
                )
                xiell_list.append(multipole_coarse)
                print(f"  ✓ ξ_{ell} computed and rebinned with RR weights: {n_s_fine} -> {n_s_coarse} bins")
            else:
                xiell_list.append(multipole_fine)
                print(f"  ✓ ξ_{ell} computed: shape {multipole_fine.shape} (rp_cut={rp_cut})")
    
        xi_multipoles = _np.column_stack(xiell_list)
        return xi_multipoles
    
    
    def _rebin_to_coarse_weighted(self, data_fine, fine_to_coarse, n_coarse, weights):
        """
        Rebin fine data to coarse bins using RR pair counts as weights.
        """
        n_jack = data_fine.shape[0]
        data_coarse = np.zeros((n_jack, n_coarse))
        
        for i_coarse in range(n_coarse):
            mask = (fine_to_coarse == i_coarse)
            if np.any(mask):
                w = weights[mask]
                w_sum = np.sum(w)
                if w_sum > 0:
                    data_coarse[:, i_coarse] = np.sum(
                        data_fine[:, mask] * w[np.newaxis, :], axis=1
                    ) / w_sum
                else:
                    data_coarse[:, i_coarse] = np.mean(data_fine[:, mask], axis=1)
        
        return data_coarse

    def _get_bin_centers(self, bins):
        """Get bin centers respecting log vs linear spacing."""
        if is_log_spaced(bins):
            return np.sqrt(bins[1:] * bins[:-1])
        else:
            return (bins[1:] + bins[:-1]) / 2.0

    def _get_fine_to_coarse_mapping(self, bins_fine, bins_coarse):
        """
        Create mapping from fine bin indices to coarse bin indices.
        """
        s_centers_fine = self._get_bin_centers(bins_fine)
        n_fine = len(s_centers_fine)
        mapping = np.zeros(n_fine, dtype=int)

        for i, s in enumerate(s_centers_fine):
            for j in range(len(bins_coarse) - 1):
                if bins_coarse[j] <= s < bins_coarse[j + 1]:
                    mapping[i] = j
                    break
            else:
                mapping[i] = len(bins_coarse) - 2

        return mapping

    def _rebin_to_coarse(self, data_fine, fine_to_coarse, n_coarse):
        """
        Average fine-binned data to coarse bins.
        """
        n_jack = data_fine.shape[0]
        data_coarse = np.zeros((n_jack, n_coarse))

        for i_coarse in range(n_coarse):
            mask = (fine_to_coarse == i_coarse)
            if np.any(mask):
                data_coarse[:, i_coarse] = np.mean(data_fine[:, mask], axis=1)

        return data_coarse
    
    
    def get_measurements(self, rp_cut=None, correction_cov=None):
        """
        Main method to compute correlation function and covariance.
        
        Pairs are computed only once on the first call. Subsequent calls
        with different correction_cov values will re-combine the same
        pairs without recomputing them.
        
        Parameters
        ----------
        rp_cut : float or None
            Transverse separation cut for multipole projection.
        correction_cov : bool or None
            If None, uses self.correction_cov (set at init).
            If True, applies Mohammad+21 α correction.
            If False, standard jackknife (α=1).
            
        Returns
        -------
        rp : array
            Bin centers.
        xi_mean : array
            Mean correlation function.
        Cov : 2D array
            Jackknife covariance matrix.
        """
        # Compute pairs only once
        if not self._pairs_computed:
            self.compute_all_pairs()
            self._pairs_computed = True
        
        # Reset totals flag so they are recomputed fresh
        # (totals themselves don't depend on alpha, but we ensure they exist)
        self._ensure_totals()
        
        # Combine pairs based on computation type
        # (Mohammad correction is applied inside these methods via correction_cov)
        if self.computation in ['WGG', 'xil']:
            self.combine_all_clustering(correction_cov=correction_cov)
        elif self.computation in ['WGP', 'xi2p']:
            self.combine_all_wgp(correction_cov=correction_cov)
        elif self.computation in ['WPP', 'WCC', 'W_+']:
            self.combine_all_wpp(correction_cov=correction_cov)
        
        # Project 2D measurements
        if self.twoD is True:
            if self.computation in ['WGG', 'WGP', 'WPP', 'WCC', 'W_+']:
                xi = np.sum(self.xi * self.du, axis=2)
                self.xi = xi
                if self.computation == 'W_+':
                    xi2 = np.sum(self.xi2 * self.du, axis=2)
                    xi = np.column_stack([xi, xi2])
                    self.xi = xi
                    
            elif self.computation == 'xil':
                xi = self.compute_correlation_multipoles(self.xi, sab=0, ells=(0,), rp_cut=rp_cut)
                self.xi = xi
                
            elif self.computation == 'xi2p':
                xi = self.compute_correlation_multipoles(self.xi, sab=2, ells=(2,), rp_cut=rp_cut)
                self.xi = xi
            
            # Compute covariance
            self.xi_mean = np.mean(xi, axis=0)
            xi_centered = xi - self.xi_mean
            Cov = (self.Ns - self.Nd) / (self.Nd * self.comb) * np.dot(xi_centered.T, xi_centered)
            rp = self.get_meanr()
            
        else:
            self.xi_mean = np.mean(self.xi, axis=0)
            xi_centered = self.xi - self.xi_mean
            Cov = (self.Ns - self.Nd) / (self.Nd * self.comb) * np.dot(xi_centered.T, xi_centered)
            rp = self.get_meanr()
            
        return rp, self.xi_mean, Cov

    
    def get_meanr(self):
        """Return bin centers for the OUTPUT (coarse) bins."""
        bins = self.bins1_coarse if hasattr(self, 'bins1_coarse') else self.bins1

        if is_log_spaced(bins):
            s = np.sqrt(bins[1:] * bins[:-1])
        else:
            s = (bins[1:] + bins[:-1]) / 2.0
        return s
    
    def set_random_pairs(self, corr):
        self.RR_ = corr.RR_
        
    
    def combine_measurements(self, corr):
        self.xi_tot = np.column_stack((self.xi, corr.xi))
        self.xi_mean = np.mean(self.xi_tot, axis=0)
        xi = self.xi_tot - self.xi_mean
        Cov = (self.Ns - self.Nd) / (self.Nd * self.comb) * np.dot(xi.T, xi)
        return Cov