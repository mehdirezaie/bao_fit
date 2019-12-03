"""
 to run the code simply:
   python get_survey_vols.py ../../inputs/quicksurvey-elg.fits 64
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import sys
sys.path.append('/Users/mehdi/Dropbox/github/DESILSS')
from clustering import hpixsum # to get the healpix map
from cosmo import cosmology    # to compute the comoving distance r(z)

import fitsio as ft
import numpy  as np
import healpy as hp

infile = sys.argv[1]
nside = int(sys.argv[2])


galaxy = ft.read(infile, lower=True)
#print(galaxy.dtype.names)

# selecting ELGs with stype of GALAXY
# find the healpix map, and the number of non-empty pixels to get footprint area
#
galcat = galaxy  #[galaxy['stype']==b'GALAXY']
galhpx = hpixsum(nside, galcat['ra'], galcat['dec'])
area   = hp.nside2pixarea(nside, degrees=True) / 3282.80635 * np.sum(galhpx != 0)
print('area = {} str'.format(area))


#
# find the comoving volume
#
omega_c = .3075
universe = cosmology(omega_c, 1.-omega_c, h=.696)
bine = np.array([.5, .75, 1., 1.25, 1.5])
vols = []
for z in bine:
  vol_i = universe.CMVOL(z) # get the comoving vol. @ redshift z
  vols.append(vol_i)

#
# find the volume in each shell and multiply by footprint area
#
volse = np.array(vols) * area / (4.* np.pi)
vols_mpc3  = np.diff(volse) * universe.h**3

for i,b_i in enumerate(0.5*(bine[1:]+bine[:-1])):
    print('z={}, volume={} Mpc/h^3'.format(b_i, vols_mpc3[i]))


