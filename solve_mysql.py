#!/usr/bin/python

"""
This script fits the observed histogram of eccentricities and orbital
periods with the selection-corrected theoretical histogram based on the
Besancon model. It is based on the mysql backend for model lookup.
"""

import argparse
import sys
import numpy as np
import numpy.random as rng
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import operator
import MySQLdb as mdb

from scipy.optimize import newton
import time

rng.seed(0)


def calculate_radius(M, logg):
    """
    Compute radius from mass and logg: (R/RSun) = sqrt(M/MSun*gSun/10**logg).
    Mass is in Solar masses, logg is in cgs units, the computed radius is in Solar radii.
    """
    return (27435.153*M/10**logg)**0.5

def calculate_sma(M1, M2, P):
    """
    Computes the semi-major axis of the binary (in solar radii).
    The masses M1 and M2 are in solar masses, the period is in days.
    """

    return 4.20661*((M1+M2)*P**2)**(1./3)

def calculate_pot1(r, D, q, F, l, n):
    """
    Computes surface potential of the primary star.
    """

    # If P0=inf, then r=0 and that causes division-by-0 warning to pop
    # up. This is a workaround for that scenario:
    if r == 0:
        r = 1e-6

    return 1./r + q*( (D**2+r**2-2*r*l*D)**-0.5 - r*l/D**2 ) + 0.5*F**2*(1.+q)*r**2*(1-n**2)

def calculate_pot2(r, D, q, F, l, n):
    """
    Computes surface potential of the secondary star.
    """
    q = 1./q
    pot = calculate_pot1(r, D, q, F, l, n)
    return pot/q + 0.5*(q-1)/q        

def conjunction_separation(a, e, w):
    """
    Calculates instantaneous separation at superior and inferior conjunctions.
    """

    dp = a*(1-e*e)/(1+e*np.sin(w))
    ds = a*(1-e*e)/(1-e*np.sin(w))
    return (dp, ds)

def dOmegadx (x, y, z, D, q, F):
    return -x*(x**2+y**2+z**2)**-1.5 - q*(x-D)*((x-D)**2+y**2+z**2)**-1.5 - q/D**2 + F**2*(1+q)*x

def d2Omegadx2 (x, y, z, D, q, F):
    return (2.*x**2-y**2-z**2)/(x**2+y**2+z**2)**2.5 + q*(2.*(x-D)**2-y**2-z**2)/((x-D)**2+y**2+z**2)**2.5 + F**2*(1+q)

def calculate_critpot(q, F, e):
    """
    Computes critical surface potentials.
    """
    D = 1.-e
    xL1 = newton(dOmegadx, D/2, d2Omegadx2, (0, 0, D, q, F))
    L1 = calculate_pot1(xL1, D, q, F, 1, 0)
    
    if q > 1:
        xL2 = newton(dOmegadx, -0.1, d2Omegadx2, (0, 0, D, q, F))
        L2 = calculate_pot1(abs(xL2), D, q, F, -1.0, 0)
    else:
        q = 1./q
        xL2 = newton(dOmegadx, -0.1, d2Omegadx2, (0, 0, D, q, F))
        L2 = calculate_pot1(abs(xL2), D, q, F, -1.0, 0)
        L2 = L2/q + 0.5*(q-1)/q
    
    return L1, L2

def distsq(d1, d2):
    return (d1.ra-d2.ra)**2+(d1.dec-d2.dec)**2

def draw_ecc(P0, method='envelope', A=3.5, B=3.0, C=0.23, E=0.98):
    """
    Some form of eccentricity distribution derived from the Kepler EB sample.
    """

    if method == 'stupid':
        if P0 < 0.18:
            return 0.0
        elif P0 < 5:
            return min(st.expon.rvs(0, 0.05), 0.9)
        elif P0 < 15:
            return min(st.expon.rvs(0, 0.12), 0.9)
        else:
            return 0.9*rng.random()

    if method == 'envelope':
        # emax(P0) = E - A*exp(-(B*P0)**C)
        emax = E - A*np.exp(-(B*P0)**C)
        
        # dN/de(a, c) = uniform or exponentiated Weibull
        #             = a*c*[1-exp(-x*c)]**(a-1)*exp(-x*c)*x**(c-1)
        e = rng.random()
        if e < emax:
            return e
        else:
            return st.exponweib.rvs(3.6163625281792133, 0.42393548261761904, 0, 0.0016752233584087976)

def count_eccs(sample, Pdist, thresh=0.025, A=3.5, B=3.0, C=0.23, E=0.98):
    """
    Counts the number of EBs with eccentricities smaller than @thresh,
    between @thresh and the envelope, and above the envelope.
    
    @sample: a sample of (logP, ecc) pairs
    @Pdist = (Prange, Phist)
    """

    emax = E - A*np.exp(-(B*10**sample[0,:])**C)
    emax[emax < thresh] = thresh

    #~ plt.plot(sample[0,:], sample[1,:], 'bo')
    #~ plt.plot(sample[0,:], emax, 'r.')
    #~ plt.show()

    idx = np.digitize(sample[0,:], Pdist[0,:])
    
    return np.array([
        (len(sample[0,:][(idx == i) & (sample[1,:] < thresh)]), 
         len(sample[0,:][(idx == i) & (sample[1,:] >= thresh) & (sample[1,:] < emax)]),
         len(sample[0,:][(idx == i) & (sample[1,:] >= emax)]))
         for i in range(len(Pdist[0,:]))])

def draw_per0():
    """
    Draw argument of periastron from a uniform distribution:
    """
    return 2*np.pi*rng.random()

def draw_incl():
    """
    Draw the inclination from a uniform distribution in cos(i).
    """
    return np.arccos(rng.random())

def draw_cosi():
    """
    Draw cos(incl) from a uniform distribution.
    """
    return rng.random()

def draw_period(P0hist, P0ranges):
    """
    Draw the orbital period from the passed distribution.
    """
    idx = np.random.choice(range(len(P0hist)), p=P0hist)
    logP0 = P0ranges[idx] + (P0ranges[idx+1]-P0ranges[idx])*rng.random()

    return 10**logP0

def draw_from_distribution(dist):
    """
    Draw from the given discrete distribution histogram. The value will
    be continuous, drawn uniformly from the chosen histogram bin.
    """

    ranges, hist = dist
    binidx = np.random.choice(range(len(hist)), p=hist)
    return ranges[binidx] + rng.random()*(ranges[binidx+1]-ranges[binidx])

def draw_meanan():
    """
    Draw mean anomaly from a uniform distribution.
    """
    return 2*np.pi*rng.random()

def join_mags(mag1, mag2):
    """
    Add two magnitudes.
    """
    # m1-m2 = -5/2 log(f1/f2)
    # f1/f0 = 10**[-0.4(m1-m0)]

    return 14.0-2.5*np.log10(10**(-0.4*(mag1-14.0))+10**(-0.4*(mag2-14.0)))

def qdist_raghavan():
    # Raghavan et al. (2010):
    qrange = np.linspace(0, 1, 21)
    qhist = np.linspace(0.05, 0.05, 20)
    qhist[ 0] = 0.000   # 0.00-0.05
    qhist[ 1] = 0.005   # 0.05-0.10
    qhist[ 2] = 0.005   # 0.10-0.15
    qhist[ 3] = 0.030   # 0.15-0.20
    qhist[19] = 0.100   # 0.95-1.00
    qhist = qhist/qhist.sum()
    return (qrange, qhist)

def mdist_raghavan():
    # Raghavan et al. (2010): $56\% \pm 2\%$ single, $33\% \pm 2\%$ binary, $8\% \pm 1\%$ triple systems and $3\% \pm 1\%$ multis.
    return {'S': 0.56, 'B': 0.33, 'T': 0.08, 'M': 0.03}

def join_Teffs(Teffs, Rs):
    return (np.sum(Rs**2*Teffs**4)/np.sum(Rs**2))**0.25

def join_loggs(loggs, Mbols):
    Ls = 10**(-0.4*Mbols)
    return np.sum(Ls*loggs)/np.sum(Ls)

class Star:
    """
    This class stores a single entry from the Besancon table.
    """
    
    def __init__(self, c):
        self.type     = 1

        self.Rmag     = c[ 0]
        self.absmagV  = c[ 9]
        self.lumclass = c[10]
        self.steltype = c[11]
        self.Teff     = c[12]
        self.logg     = c[13]
        self.age      = c[14]
        self.mass     = c[15]
        self.Mbol     = c[16]
        self.radius   = c[17]
        #~ self.radius   = calculate_radius(self.mass, self.logg)
        self.met      = c[18]
        self.ra       = c[21]
        self.dec      = c[22]
        self.dist     = c[23]
        self.redden   = c[27]

    def __repr__(self):
        return "Rc=%f  Mv=%f  CL=%d  ST=%f  T=%f  lg=%f  age=%d  M=%f  Mb=%f  R1=%f  MH=%f  RA=%f  Dc=%f  D=%f  Av=%f" % (self.Rmag, self.absmagV, self.lumclass, self.steltype, self.Teff, self.logg, self.age, self.mass, self.Mbol, self.radius, self.met, self.ra, self.dec, self.dist, self.redden)

class Single:
    """
    Class attributes:
    
    @type:     number of stars -- always 1
    @ra:       right ascension
    @dec:      declination
    @age:      age
    @mag:      apparent magnitude of the system
    @distance: distance to the system
    @absmagV:  absolute magnitude in Johnson V band
    @radius:   stellar radius
    @Teff:     effective temperature
    @logg:     effective surface gravity
    """

    def __init__(self, dbcur, age=None, DEBUG=False):
        self.type = 1

        # Draw a random star, constraining the age if requested:
        while True:
            dbcur.execute('select * from field order by rand() limit 1')
            star = Star(dbcur.fetchone())

            if age != None and star.age != age:
                continue
            break

        self.ra       = star.ra
        self.dec      = star.dec
        self.age      = star.age
        self.mag      = star.Rmag
        self.distance = star.dist
        self.absmagV  = star.absmagV
        self.period   = 1e-7
        self.EB       = False
        self.SEB      = False
        self.radius   = star.radius
        self.Teff     = star.Teff
        self.logg     = star.logg

class Binary:
    """
    Class attributes:
    
    @type:     number of stars -- always 2
    @period:   orbital period in days (either passed or drawn)
    @ecc:      orbital eccentricity (drawn according to the period)
    @incl:     orbital inclination (drawn)
    @per0:     argument of periastron (drawn)
    @meanan:   mean anomaly (drawn)
    @F:        synchronicity parameter -- always set to 1
    @physical: does the system pass sanity check (True/False)
    @sma:      semi-major axis (computed)
    @supsep:   superior conjunction separation (computed)
    @infsep:   inferior conjunction separation (computed)
    @q:        mass ratio (computed)
    @r1:       fractional primary star radius (computed)
    @r2:       fractional secondary star radius (computed)
    @pot1:     primary star surface potential (computed)
    @pot2:     secondary star surface potential (computed)
    @mag:      apparent magnitude of the system (computed)
    @absmagV   absolute magnitude of the system (computed)
    @distance: distance to the system (via distance to the primary star)
    @EB:       eclipsing binary flag (True/False)
    @SEB:      singly eclipsing binary flag (True/False)
    @ra:       right ascension
    @dec:      declination
    @age:      age
    @Teff:     effective temperature of the binary
    @logg:     effective surface gravity
    """

    def __init__(self, dbcur, period=None, q=None, age=None, Pdist=None, qdist=None, eccpars=None, check_sanity=True, safety_limit=1000, DEBUG=False):
        self.type = 2

        if Pdist is not None:
            P0ranges, P0hist = Pdist
            # Round the last bin so that the integral is exactly 1 (needed for choice):
            P0hist[-1] = 1-P0hist[:-1].sum()

        self.period = period if period != None else draw_period(P0hist, P0ranges)
        self.q = q if q != None else draw_from_distribution(qdist)
        
        if eccpars is not None:
            A, B, C, E = eccpars
            self.ecc = draw_ecc(self.period, A=A, B=B, C=C, E=E)
        else:
            self.ecc = draw_ecc(self.period)

        self.per0    = draw_per0()
        self.cosi    = draw_cosi()
        self.meanan  = draw_meanan()            # mean anomaly
        self.F       = 1.0
        
        self.physical = False
        safety_counter = 0

        # The first while-loop check whether the drawn binary is physical.
        while True:
            attempt = 0

            # The second while-loop picks a random pair of stars that
            # are coeval, have the prescribed mass ratio within the
            # 5% tolerance and are within 1 arcsec^2.
            
            while True:
                # Draw a primary star randomly, possibly constraining
                # its age:
                while True:
                    # Yep, this might be ugly, but it's actually faster
                    # than filtering the table (as we do below).
                    dbcur.execute('select * from field order by rand() limit 1')
                    primary = Star(dbcur.fetchone())

                    if age != None and primary.age != age:
                        continue
                    break
                
                # Create a pool for allowed secondaries:
                condition  = 'Age = %d ' % (primary.age)
                condition += 'and Mass > %f and Mass < %f ' % (0.95*primary.mass, 1.05*primary.mass)
                condition += 'and RA > %f and RA < %f ' % (primary.ra-0.5, primary.ra+0.5)
                condition += 'and DECL > %f and DECL < %f' % (primary.dec-0.5, primary.dec+0.5)
                dbcur.execute('select * from field where %s' % condition)
                pool = dbcur.fetchall()
                
                # If there is at least one secondary candidate, break out:
                if len(pool) > 1:
                    break
                
                # Otherwise keep an eye on it, it might be an
                # implausible mass ratio.
                attempt += 1
                if attempt > safety_limit:
                    # If q is passed, bail.
                    if q != None:
                        print('Mass ratio %f cannot be created from the Besancon stars. Bailing out.' % q)
                        exit()
                    
                    # The drawn mass ratio is too extreme; pick a new one.
                    if DEBUG:
                        print('# Requested q: %f; max attempts reached; drawing another q.' % (self.q))
                    self.q = draw_from_distribution(qdist)
                    attempt = 0
            
            # The pool contains only candidate secondaries, so we draw
            # it from that pool randomly:
            # j = int(rng.random()*len(pool))
            j = int(rng.random()*len(pool))

            #~ if DEBUG:
                #~ print('# Requested q: %f; drawn q: %f; percent diff: % 2.2f%%; distance: %2.2f arcsec' % (self.q, pool[j].mass/table.data[i].mass, (self.q-pool[j].mass/table.data[i].mass)/self.q*100, ((table.data[i].ra-pool[j].ra)**2+(table.data[i].dec-pool[j].dec)**2)**0.5))

            # primary, secondary = table.data[i], pool[j]
            self.primary, self.secondary = primary, Star(pool[j])

            # Check #1: does either of the stars overflow L2, and
            # do the stars fit into the binary:
            self.sma = calculate_sma(self.primary.mass, self.secondary.mass, self.period)
            self.pot1 = calculate_pot1(self.primary.radius/self.sma,   1-self.ecc, self.q, self.F, 0, 1)
            self.pot2 = calculate_pot2(self.secondary.radius/self.sma, 1-self.ecc, self.q, self.F, 0, 1)
            cp1, cp2 = calculate_critpot(self.q, self.F, self.ecc)

            if safety_counter < safety_limit and (self.primary.radius + self.secondary.radius > self.sma*(1-self.ecc) or self.pot1 < cp2 or self.pot2 < cp2):
                safety_counter += 1
                continue
            elif safety_counter == safety_limit:
                    # No star from the Besancon model will fit into this binary.
                    if period != None:
                        print('Period %f cannot be created from the Besancon stars. Bailing out.' % period)
                        exit()
                    
                    # The drawn period/eccentricity combination is too extreme. Pick a new one.
                    if DEBUG:
                        print('# Max attempts reached for P=%f, e=%f; drawing again.' % (self.period, self.ecc))
                    self.period = draw_period(P0hist, P0ranges)
                    self.ecc    = draw_ecc(self.period)
                    safety_counter = 0
                    continue

            # Checks survived!
            break

        # Compute instantaneous separation at both conjunctions:
        self.supsep, self.infsep = conjunction_separation(self.sma, self.ecc, self.per0)
        
        # We assume the distance to the binary is the distance to the first drawn star:
        self.distance = self.primary.dist
        
        # The same with coordinates and age:
        self.ra = self.primary.ra
        self.dec = self.primary.dec
        self.age = self.primary.age
        
        # When joining magnitudes, we need to move the secondary to the distance of the primary;
        # also note that self.distance is in kpc, so we need to multiply by 1000 (and then
        # divide by 10):
        
        self.mag = join_mags(self.primary.Rmag, self.secondary.absmagV+5.*np.log10(self.distance*100.))
        self.absmagV = join_mags(self.primary.absmagV, self.secondary.absmagV)
        
        # Effective temperature and logg of the binary:
        self.Teff = join_Teffs(np.array((self.primary.Teff, self.secondary.Teff)), np.array((self.primary.radius, self.secondary.radius)))
        self.logg = join_loggs(np.array((self.primary.logg, self.secondary.logg)), np.array((self.primary.Mbol, self.secondary.Mbol)))
        
        # We need to check if we have an eclipse at superior and/or at inferior conjunction:
        supEB = (self.primary.radius+self.secondary.radius > abs(self.supsep*self.cosi))
        infEB = (self.primary.radius+self.secondary.radius > abs(self.infsep*self.cosi))
        self.EB = supEB or infEB
        
        # Do we have just a single eclipse?
        self.SEB = (supEB != infEB)

    def __repr__(self):
        return "%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %11.5f %5.1f  %d  %d" % (self.ra, self.dec, self.period, self.ecc, self.per0, self.cosi, self.sma, self.mag, self.distance, self.Teff, self.age, self.EB, self.SEB)

class Triple:
    """
    Class attributes:
    
    @type:     number of stars -- always 3
    @ra:       right ascension
    @dec:      declination
    @age:      age
    @distance: distance to the system
    @mag:      apparent magnitude of the system
    @period:   period of the binary
    @EB:       whether the binary is eclipsing
    @SEB:      whether the binary is singly eclipsing
    @ecc:      eccentricity of the binary
    @Teff:     effective temperature of the triple
    @logg:     effective surface gravity
    """

    def __init__(self, dbcur, period=None, q=None, age=None, Pdist=None, qdist=None, eccpars=None, check_sanity=True, safety_limit=1000, DEBUG=False):
        self.type = 3

        # We generate a triple by generating a single star and a binary.
        # We constrain the age of the single star to the age of the binary.
        
        binary = Binary(dbcur, period, q, age, Pdist, qdist, eccpars, check_sanity, safety_limit, DEBUG)
        single = Single(dbcur, age=binary.age, DEBUG=DEBUG)

        self.ra       = binary.ra
        self.dec      = binary.dec
        self.age      = binary.age
        self.distance = binary.distance
        self.mag      = join_mags(binary.mag, single.absmagV+5.*np.log10(self.distance*100.))
        self.period   = binary.period
        self.EB       = binary.EB
        self.SEB      = binary.SEB
        self.ecc      = binary.ecc

        # Effective temperature and logg of the triple:
        self.Teff = join_Teffs(np.array((binary.primary.Teff, binary.secondary.Teff, single.Teff)), np.array((binary.primary.radius, binary.secondary.radius, single.radius)))
        self.logg = join_loggs(np.array((binary.primary.logg, binary.secondary.logg)), np.array((binary.primary.Mbol, binary.secondary.Mbol)))

class Multiple:
    """
    Class attributes:
    
    @type:     number of stars -- always 4
    @ra:       right ascension
    @dec:      declination
    @age:      age
    @mag:      apparent magnitude of the system
    @distance: distance to the system
    @Teff:     effective temperature of the multiple
    @logg:     effective surface gravity
    """

    def __init__(self, dbcur, period=None, q=None, age=None, Pdist=None, qdist=None, eccpars=None, check_sanity=True, safety_limit=1000, DEBUG=False):
        self.type = 4

        # We generate a multiple by generating two binary stars (i.e. a
        # hierarchical quadruple). We constrain the ages of the binaries
        # to be the same. We should probably constrain ra and dec as
        # well, and store /both/ periods, but that would change the logic
        # of the code significantly for minimal practical benefit.
        
        b1 = Binary(dbcur, period, q, age, Pdist, qdist, eccpars, check_sanity, safety_limit, DEBUG)
        b2 = Binary(dbcur, period, q, b1.age, Pdist, qdist, eccpars, check_sanity, safety_limit, DEBUG)

        self.ra       = b1.ra
        self.dec      = b1.dec
        self.age      = b1.age
        self.distance = b1.distance
        self.mag      = join_mags(b1.mag, b2.absmagV+5.*np.log10(self.distance*100.))
        self.period   = max(b1.period, b2.period)
        self.EB       = b1.EB or b2.EB
        self.SEB      = b1.SEB or b2.SEB
        self.ecc      = b1.ecc if b1.EB else b2.ecc

        # Effective temperature and logg of the triple:
        self.Teff = join_Teffs(np.array((b1.primary.Teff, b1.secondary.Teff, b2.primary.Teff, b2.secondary.Teff)), np.array((b1.primary.radius, b1.secondary.radius, b2.primary.radius, b2.secondary.radius)))
        self.logg = join_loggs(np.array((b1.primary.logg, b1.secondary.logg, b2.primary.logg, b2.secondary.logg)), np.array((b1.primary.Mbol, b1.secondary.Mbol, b2.primary.Mbol, b2.secondary.Mbol)))

class KepFOV:
    def __init__(self):
        # Kepler FOV parameters:
        self.fov = np.loadtxt('kepfov.data').reshape((84, 4, 2))

        self.outline = np.array([
            self.fov[55][0], self.fov[14][3], self.fov[19][0], self.fov[3][0],
            self.fov[10][3], self.fov[27][0], self.fov[31][0], self.fov[70][3],
            self.fov[67][0], self.fov[83][0], self.fov[74][3], self.fov[59][0]])

        self.n = len(self.outline)

        #~ if not simplified:
            #~ import K2fov.fov as fov
            #~ from K2fov.K2onSilicon import angSepVincenty as sphere_dist
            #~ fra, fdec, froll = 290.6688, 44.4952, 303
            #~ froll = fov.getFovAngleFromSpacecraftRoll(froll)
            #~ self.fov = fov.KeplerFov(fra, fdec, froll)

    def within_box(self, ra, dec):
        return (279.60813 < ra < 301.85564) and (36.523277 < dec < 52.481925)

    def within_outline(self, ra, dec):
        inside = False

        p1x, p1y = self.outline[0]
        for i in range(self.n+1):
            p2x, p2y = self.outline[i % self.n]
            if dec > min(p1y, p2y):
                if dec <= max(p1y, p2y):
                    if ra <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (dec-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or ra <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def on_silicon(self, ra, dec):
        inside = False

        for ccd in range(84):
            p1x, p1y = self.fov[ccd][0]
            for i in range(5):
                p2x, p2y = self.fov[ccd][i % 4]
                if dec > min(p1y, p2y):
                    if dec <= max(p1y, p2y):
                        if ra <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (dec-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                            if p1x == p2x or ra <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                return True

        return False

    #~ def on_silicon_detailed(self, ra, dec):
        #~ """
        #~ The original RA, Dec, and roll (all degrees) of Kepler was:
        #~ 
        #~ 290.6688, 44.4952, 110
        #~ 
        #~ The pointing moved slightly (at the arcsec level) throughout
        #~ the mission. The roll is the angle that gets the spacecraft Y &
        #~ Z axis pointed at the Kepler FOV. There is an additional hidden
        #~ 13-deg roll that is built into the photometer corresponding to
        #~ the roll angle of the focal plane wrt the spacecraft axes and a
        #~ 180-deg due to the reflection in the spherical mirror. So, I
        #~ think the total roll angle that you want to use for comparison
        #~ with the K2 roll numbers specified in the Campaign descriptions
        #~ is:  110 + 13 + 180 = 303-deg (and don't forget about the
        #~ seasonal 90-deg rolls that get added in. The summer (season 0)
        #~ is the one that corresponds to this roll angle.
        #~ """
        #~ 
        #~ try:
            #~ dist = sphere_dist(self.fov.ra0_deg, self.fov.dec0_deg, ra, dec)
            #~ if dist >= 90.:
                #~ return False
            #~ ch = self.fov.pickAChannel(ra, dec)
            #~ ch, col, row = self.fov.getChannelColRow(ra, dec)
            #~ 
            #~ return True
        #~ except ValueError:
            #~ return False


class Observation:
    def __init__(self, targets):
        # The passed field is a list of stars generated from the Besancon
        # model. This class applies instrumental selection effects.
        self.targets = targets
        
        # Read in target selection ratios. These are the ratios between
        # Batalha et al. (2010)'s Table 2 and the equivalent table
        # produced by the forward model. The fractions tell us the
        # suppression factor as a function of mag, Teff, logg to get
        # the equivalent selection of Kepler targets.
        self.fractions = np.loadtxt('fractions.tab')[:,1:]
        self.fractions[self.fractions > 1.0] = 1.0
        
        # Duty cycle correction takes a long time to compute, so instead
        # of computing it here, we just read in the results of the
        # standalone computation.
        self.dc_p0, self.dc_prob = np.loadtxt('duty_cycle.data', unpack=True)
        self.dc_prob /= self.dc_prob[0]
        
        # SNR correction is a bit more manageable, so we do it here, on
        # the fly.
        period, pdepth, sigma = np.loadtxt("kepEBs.csv", delimiter=",", unpack=True, usecols=(1, 3, 7))
        logPobs = np.log10(period)
        
        # First let's take all well sampled SNRs (say, logP < 1) and
        # figure out what the "ground" distribution looks like.
        SNR_flat = pdepth[(pdepth > 0) & (sigma > 0) & (logPobs < 1)]/sigma[(pdepth > 0) & (sigma > 0) & (logPobs < 1)]
        SNR_hist, SNR_range = np.histogram(SNR_flat, bins=100)
        
        # Next let's compute the distributions of all SNRs per logP bin:
        logP = np.linspace(-1, 3, 100)
        SNR_per_bin = [pdepth[(pdepth > 0) & (sigma > 0) & (logPobs >= logP[i]) & (logPobs < logP[i+1])]/sigma[(pdepth > 0) & (sigma > 0) & (logPobs >= logP[i]) & (logPobs < logP[i+1])] for i in range(len(logP)-1)]
        
        # Now let's fit a straight line to the minimum log(SNR) on the
        # 1 < logP < 3 range.
        SNR_min = np.array([SNR.min() for SNR in SNR_per_bin])
        SNR_min_for_fit = np.log10(SNR_min[logP[:-1] > 1])
        SNR_logP_for_fit = logP[logP[:-1] > 1]
        p, v = np.polyfit(SNR_logP_for_fit, SNR_min_for_fit, 1, cov=True)
        
        # This line is what determines what part of the original
        # S/N population we lose because of the increased minimum S/N.
        self.snr_baseline = SNR_flat
        self.snr_coeffs = p

    def eta_dc(self, period):
        """
        For the passed period in days, return the probability that we
        detect at least two eclipses due to duty cycle.
        """

        return np.interp(period, self.dc_p0, self.dc_prob)

    def eta_snr(self, period):
        """
        For the passed period in days, return the probability that we
        detect at least two eclipses due to signal-to-noise ratio.
        """

        snrmin = 10**(self.snr_coeffs[0]*np.log10(period) + self.snr_coeffs[1])
        if not hasattr(snrmin, '__len__'):
            return float(len(self.snr_baseline[self.snr_baseline > snrmin]))/len(self.snr_baseline)
        else:
            return np.array( [float(len(self.snr_baseline[self.snr_baseline > x]))/len(self.snr_baseline) for x in snrmin] )

    def selected(self, target, DEBUG=False):
        """
        For the passed mag, teff and logg, return the probability that
        the object will be on the target list.
        """

        if target.mag > 16: return False
        
        col = min(max(int((11000.-target.Teff)/1000), 0), 8)
        if target.logg < 3.5:
            col += 8
        row = min(max(int(target.mag-6.0), 0), 10)
        if (row == 10) or (col == 16) or (col == 8 and target.logg >= 3.5):
            return False
        
        if DEBUG:
            print('# %6.0f %8.2f %7.2f %4d %4d %6.3f' % (target.Teff, target.mag, target.logg, col, row, self.fractions[row,col]))
        
        return rng.random() < self.fractions[row,col]

        # 383 EBs were known before Kepler's first light:
        # - 59 from Simbad
        # - 127 from ASAS
        # - 7 from HET
        # - 190 from Vulcan
        
    def observe(self, fov=None):
        for target in self.targets:
            # If on-silicon test is requested, perform it:
            if fov is not None:
                target.on_silicon = False
                if fov.within_box(ra, dec):
                    if fov.within_outline(ra, dec):
                        if fov.on_silicon(ra, dec):
                            target.on_silicon = True
                if target.on_silicon == False:
                    target.is_target = False
                    target.detected = False
                    continue
            else:
                target.on_silicon = True

            # Is the target on the target list:
            target.is_target = self.selected(target)

            # Given the period of the target, compute the detection probability:
            eta = self.eta_dc(target.period) * self.eta_snr(target.period)
            
            # Roll a dice to see if that target is going to be observed:
            prob = rng.random()
            
            if prob <= eta:
                target.detected = True
            else:
                target.detected = False

    def observe_one(self, target, fov=None):
        if fov is not None:
            target.on_silicon = False
            if fov.within_box(ra, dec):
                if fov.within_outline(ra, dec):
                    if fov.on_silicon(ra, dec):
                        target.on_silicon = True
            if target.on_silicon == False:
                target.is_target = False
                target.detected = False
                return
        else:
            target.on_silicon = True

        # Is the target on the target list:
        target.is_target = self.selected(target)

        # Given the period of the target, compute the detection probability:
        eta = self.eta_dc(target.period) * self.eta_snr(target.period)
        
        # Roll a dice to see if that target is going to be observed:
        prob = rng.random()
        
        if prob <= eta:
            target.detected = True
        else:
            target.detected = False

def simulate_field(dbcur, argv, mdist, Pdist, qdist, eccpars, DEBUG=True):
    """
    Pdist = (P0ranges, P0hist)
    """
    field = []
    if DEBUG:
        print('# requested sample size: %d' % argv.sample_size)
    
    if argv.maxEBs != 0:
        if DEBUG:
            print('# maximum number of EBs set at: %d' % (argv.maxEBs))
        
        Snum, Bnum, Tnum, Mnum, EBnum = 0, 0, 0, 0, 0
        
        # We need to observe this as we create them so that we know
        # how many EBs we have created.
        run = Observation(None)

        while EBnum != argv.maxEBs:
            # Let's roll a dice to see which type of system we will
            # create. We need to do this so that we can count the number
            # of EBs that we create.
            roll = rng.random()
            if roll < mdist['S']:
                field.append(Single(dbcur))
                Snum += 1
            elif roll < mdist['S'] + mdist['B']:
                field.append(Binary(dbcur, Pdist=Pdist, qdist=qdist, eccpars=eccpars, check_sanity=True, safety_limit=100))
                Bnum += 1
            elif roll < mdist['S'] + mdist['B'] + mdist['T']:
                field.append(Triple(dbcur, Pdist=Pdist, qdist=qdist, eccpars=eccpars, check_sanity=True, safety_limit=100))
                Tnum += 1
            else:
                field.append(Multiple(dbcur, Pdist=Pdist, qdist=qdist, eccpars=eccpars, check_sanity=True, safety_limit=100))
                Mnum += 1

            # Observe this target:
            run.observe_one(field[-1])

            # Is it an eclipsing binary?
            if field[-1].EB and field[-1].on_silicon and field[-1].is_target:
                EBnum += 1

        if DEBUG:
            print('# %d single stars created.' % (Snum))
            print('# %d binaries created.' % (Bnum))
            print('# %d triples created.' % (Tnum))
            print('# %d multiples created.' % (Mnum))
    else:
        Bnum = int(mdist['B']*argv.sample_size)
        Tnum = int(mdist['T']*argv.sample_size)
        Mnum = int(mdist['M']*argv.sample_size)
        Snum = argv.sample_size-Bnum-Tnum-Mnum

        # Generate single stars:
        for i in range(Snum):
            field.append(Single(dbcur))
        print('# %d single stars created.' % (Snum))
        
        # Generate binary stars:
        for i in range(Bnum):
            field.append(Binary(dbcur, Pdist=Pdist, qdist=qdist, check_sanity=True, safety_limit=100))
        print('# %d binaries created.' % (Bnum))
        
        # Generate triple stars:
        for i in range(Tnum):
            field.append(Triple(dbcur, Pdist=Pdist, qdist=qdist, check_sanity=True, safety_limit=100))
        print('# %d triples created.' % (Tnum))
        
        # Generate multiple stars:
        for i in range(Mnum):
            field.append(Multiple(dbcur, Pdist=Pdist, qdist=qdist, check_sanity=True, safety_limit=100))
        print('# %d multiples created.' % (Mnum))
    
        run = Observation(field)
        strt = time.time()
        run.observe()
        print '# Observed in %.3fs' % (time.time() - strt)

    return field

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Find the underlying binary period and eccentricity distributions.')
    parser.add_argument('-o', '--observe',      action='store_true',  help='generate a synthetic data-set of the Kepler field\
                                                                            (default: False)')
    parser.add_argument('-c', '--count',        action='store_true',  help='use a previously determined underlying period\
                                                                            distribution to count SEBs and EBs (default: False)')
    parser.add_argument('-s', '--solve',        action='store_true',  help='run the forward model and compare to the observed\
                                                                            period and eccentricity distributions (default: False)')
    parser.add_argument('-t', '--table',        metavar='mysqltab',   help='name of the mysql Besancon table (default: kepfield_besancon)', type=str,   default='kepfield_besancon')
    parser.add_argument('-b', '--bins',         metavar='num',        help='number of histogram bins (default: 20)',                   type=int,   default=20)
    parser.add_argument('-x', '--xi',           metavar='val',        help='step size for differential corrections (default: 0.05)',   type=float, default=0.05)
    parser.add_argument('-q', '--qdist',        metavar='dist',       help='underlying mass ratio distribution (default: raghavan)',   type=str,   choices=['raghavan'], default='raghavan')
    parser.add_argument('-m', '--mdist',        metavar='mdist',      help='underlying multiplicity distribution (default: raghavan)', type=str,   choices=['raghavan'], default='raghavan')
    parser.add_argument('-P', '--Pdist',        metavar='pdist',      help='underlying multiplicity distribution (default: uniform)',  type=str,   choices=['uniform', 'ulogP'], default='uniform')
    parser.add_argument(      '--lpexcess',     metavar='val',        help='fraction of long period EBs (default: 0.65)',              type=float, default=0.65)
    parser.add_argument(      '--lpbin',        action='store_true',  help='include a bin for long period EBs (default: False)')
    parser.add_argument(      '--ulogP',        metavar='file',       help='filename of the underlying period distribution\
                                                                            (default: ulogP.dist)',                                    type=str,   default='ulogP.dist')
    parser.add_argument(      '--maxstars',     metavar='num',        help='maximum number of stars to be read in from the\
                                                                            Besancon table (default: all)',                            type=int,   default=0)
    parser.add_argument(      '--maxEBs',       metavar='num',        help='stop when the passed number of EBs has been created\
                                                                            (default: no limit)',                                      type=int,   default=0)
    parser.add_argument(      '--sample-size',  metavar='num',        help='number of objects to be generated (default: 200000)',      type=int,   default=200000)
    parser.add_argument(      '--only-single',  action='store_true',  help='generate only single stars (default: False)')
    parser.add_argument(      '--on-silicon',   action='store_true',  help='generate only targets on silicon (default: False)')
    argv = parser.parse_args()

    # Initialize Kepler FOV:
    kepfov = KepFOV()

    # Initialize the Besancon mysql database:
    con = mdb.connect('localhost', 'andrej', 'besancon', argv.table)
    dbcur = con.cursor()
    dbcur.execute('select R from field')
    Nstars = dbcur.rowcount
    print("# %d entries read in from the mysql table %s." % (Nstars, argv.table))

    # Mass ratio distribution:
    if argv.qdist == 'raghavan':
        print('# using Raghavan et al. (2010) for mass ratio distribution.')
        qdist = qdist_raghavan()

    if argv.count:
        # Read in the previously computed underlying period distribution:
        P0ranges, P0hist, P0histerr = np.loadtxt(argv.ulogP, unpack=True)
        P0ranges = np.append(P0ranges-(P0ranges[1]-P0ranges[0])/2, [P0ranges[-1]+(P0ranges[1]-P0ranges[0])/2])

        if argv.lpbin:
            # Add the last bin for the Long Period EBs (LPEBs):
            P0ranges = np.append(P0ranges, [np.inf])
            P0hist = np.append(P0hist*(1-argv.lpexcess), argv.lpexcess)

        # Number of binary stars in the sample:
        if argv.lpbin:
            Bnum = int(0.33*argv.sample_size)
        else:
            Bnum = int((1-argv.lpexcess)*0.33*argv.sample_size)

        print('# Number of binaries to be generated: %d' % (Bnum))

        total_EBs = 0
        total_SEBs = 0
        for i in range(Bnum):
            b = Binary(dbcur, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100)

            total_EBs += b.EB
            total_SEBs += b.SEB
            print b, " %2.2f%%" % ((float(i)+1)/Bnum*100)

        print("# Total EBs:  %d/%d (%2.2f%%)" % (total_EBs, Bnum, 100*float(total_EBs)/Bnum))
        print("# Total SEBs: %d/%d (%2.2f%%)" % (total_SEBs, Bnum, 100*float(total_SEBs)/Bnum))
        exit()

    if argv.observe:
        # Initialize multiplicity distribution:
        if argv.only_single == True:
            mdist = {'S': 1.0, 'B': 0.0, 'T': 0.0, 'M': 0.0}
        elif argv.mdist == 'raghavan':
            mdist = mdist_raghavan()
        else:
            print('Unsupported multiplicity distribution, aborting.')
            exit()
        
        print('# multiplicity distribution: %2.2f single, %2.2f binary, %2.2f triple, %2.2f multi systems' % (mdist['S'], mdist['B'], mdist['T'], mdist['M']))
        
        # Read in the previously computed underlying period distribution:
        P0ranges, P0hist, P0histerr = np.loadtxt(argv.ulogP, unpack=True)
        P0ranges = np.append(P0ranges-(P0ranges[1]-P0ranges[0])/2, [P0ranges[-1]+(P0ranges[1]-P0ranges[0])/2])
        print('# underlying binary period distribution loaded from %s.' % (argv.ulogP))
        
        if argv.lpbin:
            # Add the last bin for the Long Period EBs (LPEBs):
            print('# adding long-period binary and multiple star bin.')
            P0ranges = np.append(P0ranges, [np.inf])
            P0hist = np.append(P0hist*(1-argv.lpexcess), argv.lpexcess)
            bins += 1
        else:
            # Otherwise correct for the long period excess:
            print('# correcting occurrence rates by long period excess factor %3.3f.' % (argv.lpexcess))
            mdist['B'] *= (1-argv.lpexcess)
            mdist['T'] *= (1-argv.lpexcess)
            mdist['M'] *= (1-argv.lpexcess)
            mdist['S'] = 1.-mdist['B']-mdist['T']-mdist['M']
        
        # Build a synthetic sample of the Kepler field.
        field = []
        print('# requested sample size: %d' % argv.sample_size)
        
        if argv.maxEBs != 0:
            print('# maximum number of EBs set at: %d' % (argv.maxEBs))

            Snum, Bnum, Tnum, Mnum, EBnum = 0, 0, 0, 0, 0

            # We need to observe this as we create them so that we know
            # how many EBs we have created.
            run = Observation(None)

            while EBnum != argv.maxEBs:
                # Let's roll a dice to see which type of system we will
                # create. We need to do this so that we can count the number
                # of EBs that we create.
                roll = rng.random()
                if roll < mdist['S']:
                    field.append(Single(dbcur))
                    Snum += 1
                elif roll < mdist['S'] + mdist['B']:
                    field.append(Binary(dbcur, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
                    Bnum += 1
                elif roll < mdist['S'] + mdist['B'] + mdist['T']:
                    field.append(Triple(dbcur, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
                    Tnum += 1
                else:
                    field.append(Multiple(dbcur, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
                    Mnum += 1

                # Observe this target:
                run.observe_one(field[-1])

                # Is it an eclipsing binary?
                if field[-1].EB and field[-1].on_silicon and field[-1].is_target:
                    EBnum += 1

            print('# %d single stars created.' % (Snum))
            print('# %d binaries created.' % (Bnum))
            print('# %d triples created.' % (Tnum))
            print('# %d multiples created.' % (Mnum))
        else:
            Bnum = int(mdist['B']*argv.sample_size)
            Tnum = int(mdist['T']*argv.sample_size)
            Mnum = int(mdist['M']*argv.sample_size)
            Snum = argv.sample_size-Bnum-Tnum-Mnum

            # Generate single stars:
            for i in range(Snum):
                field.append(Single(dbcur))
            print('# %d single stars created.' % (Snum))
            
            # Generate binary stars:
            for i in range(Bnum):
                field.append(Binary(dbcur, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
            print('# %d binaries created.' % (Bnum))
            
            # Generate triple stars:
            for i in range(Tnum):
                field.append(Triple(dbcur, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
            print('# %d triples created.' % (Tnum))
            
            # Generate multiple stars:
            for i in range(Mnum):
                field.append(Multiple(dbcur, Pdist=(P0ranges, P0hist), qdist=qdist, check_sanity=True, safety_limit=100))
            print('# %d multiples created.' % (Mnum))
        
            run = Observation(field)
            strt = time.time()
            run.observe()
            print '# Observed in %.3fs' % (time.time() - strt)
        
        print('# Type    R.A.         Dec.       Period  OnSilicon? Detected? Selected? EB?      Teff    logg     Kp')
        for target in field:
             print('%5d  %10.6f  %10.6f  %10.6f   %5s     %5s    %5s    %5s    %5.0f    %0.2f   %5.1f' % (target.type, target.ra, target.dec, target.period, target.on_silicon, target.detected, target.is_target, target.EB, target.Teff, target.logg, target.mag))
        
        print('# Total targets on silicon: %d' % (len([t for t in field if t.on_silicon])))
        print('# Total targets selected: %d' % (len([t for t in field if t.is_target])))
        print('# Total EBs: %d' % (len([t for t in field if t.on_silicon and t.is_target and t.EB])))
        print('# Total SEBs: %d' % (len([t for t in field if t.on_silicon and t.is_target and t.SEB])))

        exit()

    if argv.solve:
        # Read in observed periods:
        catKIC, catP0 = np.loadtxt('kepEBs.csv', delimiter=',', usecols=(0, 1), unpack=True)
        print('# %d systems with measured orbital periods read in.' % (len(catKIC)))
        
        # Read in observed periods and eccentricities:
        eccKIC, eccP0, obsEB_ecc = np.loadtxt('ecc.final_with_p0.res', usecols=(0, 1, 2), unpack=True)
        print('# %d systems with measured eccentricities read in.' % (len(eccKIC)))

        # Initialize multiplicity distribution:
        if argv.mdist == 'raghavan':
            mdist = mdist_raghavan()
        print('# multiplicity distribution: %2.2f single, %2.2f binary, %2.2f triple, %2.2f multi systems' % (mdist['S'], mdist['B'], mdist['T'], mdist['M']))

        # Initialize the starting P0 histogram.
        bins = argv.bins
        P0hist, P0ranges = np.histogram(np.log10(catP0), bins=bins)

        if argv.Pdist == 'uniform':
            synP0hist = np.array([len(catP0)/bins]*(bins-len(catP0)%bins)+[len(catP0)/bins+1]*(len(catP0)%bins))
            synP0hist = np.array([float(v)/len(catP0) for v in synP0hist])
            print('# uniform underlying log10(P0) histogram with %d bins created.' % (bins))
        elif argv.Pdist == 'ulogP':
            P0ranges, synP0hist = np.loadtxt(argv.ulogP, usecols=(0, 1), unpack=True)
            P0ranges = np.append(P0ranges-(P0ranges[1]-P0ranges[0])/2, [P0ranges[-1]+(P0ranges[1]-P0ranges[0])/2])
            print('# initial underlying binary period distribution loaded from %s.' % (argv.ulogP))
        else:
            print("can't ever get here, right?")
            exit()
        
        if argv.lpbin:
            # Add the last bin for the Long Period EBs (LPEBs):
            print('# adding long-period binary and multiple star bin.')
            P0ranges = np.append(P0ranges, [np.inf])
            synP0hist = np.append(synP0hist*(1-argv.lpexcess), argv.lpexcess)
            bins += 1
        else:
            # Otherwise correct for the long period excess:
            print('# correcting occurrence rates by long period excess factor %3.3f.' % (argv.lpexcess))
            mdist['B'] *= (1-argv.lpexcess)
            mdist['T'] *= (1-argv.lpexcess)
            mdist['M'] *= (1-argv.lpexcess)
            mdist['S'] = 1.-mdist['B']-mdist['T']-mdist['M']
                
        eccpars = [3.5, 3.0, 0.23, 0.98]
        print('# initial eccentricity envelope parameters set: %s' % eccpars)
        
        # This is where the loop needs to begin.
        while True:
            # Build a synthetic sample of the Kepler field.
            field = simulate_field(dbcur, argv, mdist, (P0ranges, synP0hist), qdist, eccpars, DEBUG=True)

            # Simulated EBs comprise our comparison sample.
            simEBs  = [t for t in field if t.on_silicon and t.is_target and t.EB]
            simDEBs = [t for t in field if t.on_silicon and t.is_target and t.EB and not t.SEB] # only doubly-eclipsing EBs should be in this sample
            simEB_P0  = np.array([eb.period for eb in simEBs])
            simEB_ecc = np.array([eb.ecc    for eb in simDEBs])
            
            simEB_hist, simEB_ranges = np.histogram(np.log10(simEB_P0), bins=P0ranges)
            
            print('# Total targets on silicon: %d' % (len([t for t in field if t.on_silicon])))
            print('# Total targets selected: %d' % (len([t for t in field if t.is_target])))
            print('# Total EBs: %d' % (len(simEBs)))
            print('# Total SEBs: %d' % (len([t for t in field if t.on_silicon and t.is_target and t.SEB])))
            
            print('# Comparison:')
            print('# PERIODS:')
            print('# %12s %12s %12s' % ('observed:', 'simulated:', 'difference:'))
            for i in range(bins):
                print('# %12.6f %12.6f %12.6f' % (float(P0hist[i])/P0hist.sum(), float(simEB_hist[i])/simEB_hist.sum(), float(P0hist[i])/P0hist.sum()-float(simEB_hist[i])/simEB_hist.sum()))
            
            print('# NUMBERS:')
            print('# EB fraction observed:  %12.6f%%' % (2775./201775*100))
            print('# EB fraction simulated: %12.6f%%' % (100*float(len(simEBs))/(len([t for t in field if t.on_silicon]))))

            sim_ecc_hist, sim_ecc_range = np.histogram(simEB_ecc, bins=np.linspace(0, 1, 10))
            obs_ecc_hist, obs_ecc_range = np.histogram(obsEB_ecc, bins=np.linspace(0, 1, 10))

            print('# ECCENTRICITIES:')
            print('# %12s %12s %12s' % ('observed:', 'simulated:', 'difference:'))
            for i in range(len(sim_ecc_hist)):
                print('# %12.6f %12.6f %12.6f' % (float(obs_ecc_hist[i])/obs_ecc_hist.sum(), float(sim_ecc_hist[i])/sim_ecc_hist.sum(), float(obs_ecc_hist[i])/obs_ecc_hist.sum()-float(sim_ecc_hist[i])/sim_ecc_hist.sum()))

            logL = -0.5*((P0hist.astype(float)/P0hist.sum()-simEB_hist.astype(float)/simEB_hist.sum())**2).sum() + 1000*(2775./201775 - float(len(simEBs))/(len([t for t in field if t.on_silicon])))**2 + ((obs_ecc_hist.astype(float)/obs_ecc_hist.sum()-sim_ecc_hist.astype(float)/sim_ecc_hist.sum())**2).sum()

            print('# logL = %f' % (logL))
            print('# logL = %f = -0.5*(%f + %f + %f)' % (logL, ((P0hist.astype(float)/P0hist.sum()-simEB_hist.astype(float)/simEB_hist.sum())**2).sum(), 1000*(2775./201775 - float(len(simEBs))/(len([t for t in field if t.on_silicon])))**2, ((obs_ecc_hist.astype(float)/obs_ecc_hist.sum()-sim_ecc_hist.astype(float)/sim_ecc_hist.sum())**2).sum()))

            break

        exit()

    # Initialize the ranges for various histograms:
    #~ qbins = np.linspace(0, 1, 20)
    #~ Rsumbins = [x for x in np.linspace(0,1,29)]+[100]

    # Reduce the font size for plots:
    #~ mpl.rcParams.update({'font.size': 7})
    
    #~ for cnt in range(1,201):    
        #~ P0dist = st.rv_discrete(name='discrete', values=(np.arange(bins), vP0syn))

        #~ P0sel, eccsel, qsel, Rsumsel, Rratsel, sinisel = [], [], [], [], [], []
        #~ numBs, numEBs = 0, 0
        
        #~ while numEBs != len(P0obs):
            #~ rP0 = P0dist.rvs()
            #~ P0 = 10**st.uniform.rvs(rP0obs[rP0], binw)
            #~ b = Binary(table.data, mode=MODE, period=P0, check_sanity=True, safety_limit=10000)

            #~ while not b.physical:
                #~ rP0 = P0dist.rvs()
                #~ P0 = 10**st.uniform.rvs(rP0obs[rP0], binw)
                #~ b = Binary(table.data, mode=MODE, period=P0, check_sanity=True, safety_limit=10000)

            #~ numBs += 1

            #~ if b.EB:
                #~ numEBs += 1
                #~ P0sel.append(np.log10(b.period))
                #~ eccsel.append(b.ecc)
                #~ qsel.append(b.q if b.q <= 1 else 1./b.q)
                #~ Rsumsel.append(b.r1+b.r2)
                #~ Rratsel.append((b.r2/b.r1) if (b.r2/b.r1) <= 1 else (b.r1/b.r2))
                #~ sinisel.append(np.sin(b.incl))

            #~ print ("# run %03d: numBs = %d, numEBs = %d" % (cnt, numBs, numEBs))

        #~ vP0sel, rP0sel = np.histogram(P0sel, bins=rP0obs)
        #~ vP0sel = [float(v)/len(P0obs) for v in vP0sel]
        
        #~ delta = vP0sel-vP0obs
        #~ cf = (delta**2).sum()
        #~ print cnt, cf

        #~ font = {'family': 'serif', 'variant': 'normal', 'weight': 'normal', 'size': 16}
        #~ mpl.rc('font', **font)
            
        #~ if EXTRA == 'onlyP':
            #~ plt.close()
            #~ fig = plt.figure(1)
            #~ fig.set_size_inches(3.6,7.2)
            #~ fig.patch.set_alpha(0.0)
            #~ plt.suptitle("i=%03d, %d bins, xi=%2.2f, cf=%f" % (cnt, bins, xi, cf), fontsize=12)
            #~ plt.subplot(311)
            #~ plt.ylabel("dN/dlogP")
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0,0.14)
            #~ plt.bar(rP0sel[:-1], vP0syn, binw)
            #~ plt.subplot(312)
            #~ plt.ylabel("Nsyn")
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0, 200)
            #~ plt.hist(P0sel, bins=rP0sel)
            #~ plt.subplot(313)
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0, 200)
            #~ plt.xlabel("logP")
            #~ plt.ylabel("Nobs")
            #~ plt.hist(P0obs, bins=rP0obs)
            #~ plt.subplots_adjust(left=0.23, right=0.98, top=0.93, bottom=0.1)
            #~ plt.savefig("img%03d.png" % cnt, dpi=100)

        #~ else:
            #~ plt.clf()
            #~ fig = plt.figure(1)
            #~ fig.patch.set_alpha(0.0)
            
            #~ plt.suptitle("Iteration %03d, %d bins, bin width=%3.3f, xi=%2.2f, cf=%f" % (cnt, bins, binw, xi, cf), fontsize=12)
            #~ plt.subplot(331)
            #~ plt.ylabel("dN/dlogP")
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0,0.14)
            #~ plt.bar(rP0sel[:-1], vP0syn, binw)
            #~ plt.subplot(334)
            #~ plt.ylabel("Nsyn")
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0, 200)
            #~ plt.hist(P0sel, bins=rP0sel)
            #~ plt.subplot(337)
            #~ plt.xlim(-1.2, 3.03)
            #~ plt.ylim(0, 200)
            #~ plt.xlabel("logP")
            #~ plt.ylabel("Nobs")
            #~ plt.hist(P0obs, bins=rP0obs)
            #~ plt.subplot(332)
            #~ plt.xlabel("logP")
            #~ plt.ylabel("ecc_syn")
            #~ plt.plot(P0obs, eccobs, 'rx', markersize=0.5)
            #~ plt.plot(P0sel, eccsel, 'b.', markersize=0.5)
            #~ plt.subplot(335)
            #~ plt.xlim(0, 1)
            #~ plt.ylim(0, 600)
            #~ plt.xlabel("Eccentricity")
            #~ plt.ylabel("Nsyn")
            #~ plt.hist(eccsel, bins=20)
            #~ plt.subplot(338)
            #~ plt.xlim(0, 1)
            #~ plt.ylim(0, 600)
            #~ plt.xlabel("Mass ratio")
            #~ plt.ylabel("Nsyn")
            #~ plt.hist(qsel, qbins)
            #~ plt.subplot(333)
            #~ plt.xlabel("(R1+R2)/a")
            #~ plt.ylabel("Nsyn")
            #~ plt.xlim(0, 1.033)
            #~ plt.hist(Rsumsel, bins=Rsumbins)
            #~ plt.subplot(336)
            #~ plt.xlabel("R2/R1")
            #~ plt.ylabel("Nsyn")
            #~ plt.hist(Rratsel, bins=30)
            #~ plt.subplot(339)
            #~ plt.xlabel("sin(i)")
            #~ plt.ylabel("Nsyn")
            #~ plt.hist(sinisel, bins=30)
            #~ plt.savefig("img%03d.png" % cnt, dpi=200)

        # Correct the input histogram:
        #~ for i in range(len(vP0syn)):
            #~ vP0syn[i] -= xi*delta[i]
        #~ vP0syn /= vP0syn.sum()
