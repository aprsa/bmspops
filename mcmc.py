import solve_mysql as solve
import emcee
from emcee.utils import MPIPool
import numpy as np
import MySQLdb as mdb

# Initialize the Besancon mysql database:
con = mdb.connect('localhost', 'andrej', 'besancon', 'kepfield_besancon')
dbcur = con.cursor()

class Argv:
    def __init__(self):
        self.maxstars = 0
        self.maxEBs = 10
        self.sample_size = 500000
        self.lpexcess = 0.6
        

def lnprob(x, boundaries, P0ranges, P0hist, obsEB_ecc, mdist, qdist, argv):
    # Check to see that all values are within the allowed limits:
    if not np.all([boundaries[i][0] < x[i] < boundaries[i][1] for i in range(len(boundaries))]):
        return -np.inf

    synP0hist = x[:20]
    lpexcess = x[20]
    argv.lpexcess = x[20]
    eccpars = x[21:]

    ndist = mdist.copy()
    ndist['B'] *= (1-argv.lpexcess)
    ndist['T'] *= (1-argv.lpexcess)
    ndist['M'] *= (1-argv.lpexcess)
    ndist['S'] = 1.-ndist['B']-ndist['T']-ndist['M']

    # x[] is an array of sampled values from parameter priors:
    field = solve.simulate_field(dbcur, argv, ndist, (P0ranges, synP0hist), qdist, eccpars, DEBUG=False)
    
    # Simulated EBs comprise our comparison sample.
    simEBs  = [t for t in field if t.on_silicon and t.is_target and t.EB]
    simDEBs = [t for t in field if t.on_silicon and t.is_target and t.EB and not t.SEB] # only doubly-eclipsing EBs should be in this sample
    simEB_P0  = np.array([eb.period for eb in simEBs])
    simEB_ecc = np.array([eb.ecc    for eb in simDEBs])
    
    simEB_hist, simEB_ranges = np.histogram(np.log10(simEB_P0), bins=P0ranges)
    sim_ecc_hist, sim_ecc_range = np.histogram(simEB_ecc, bins=np.linspace(0, 1, 10))
    obs_ecc_hist, obs_ecc_range = np.histogram(obsEB_ecc, bins=np.linspace(0, 1, 10))
    simEB_hist = simEB_hist.astype(float)
    sim_ecc_hist = sim_ecc_hist.astype(float)
    obs_ecc_hist = obs_ecc_hist.astype(float)

    logL = -0.5*(
        ((P0hist/P0hist.sum()-simEB_hist/simEB_hist.sum())**2).sum() +
        1000*(2775./201775 - (float(len(simEBs))/(len(field))))**2 +
        ((obs_ecc_hist/obs_ecc_hist.sum()-sim_ecc_hist/sim_ecc_hist.sum())**2).sum())

    return logL

def run(starting_point, boundaries, bins, state, nwalkers, niter):
    ndim = len(starting_point)

    if state is not None:
        p0 = np.loadtxt(state)[:, 1:-1]
    else:
        p0 = np.array([(0.99+0.02*np.random.rand(len(starting_point)))*starting_point for i in xrange(nwalkers)])
        #~ p0 = np.array([[p[0] + (p[1]-p[0])*np.random.rand() for p in priors] for i in xrange(nwalkers)])
    # Observations:
    catKIC, catP0 = np.loadtxt('kepEBs.csv', delimiter=',', usecols=(0, 1), unpack=True)
    eccKIC, eccP0, obsEB_ecc = np.loadtxt('ecc.final_with_p0.res', usecols=(0, 1, 2), unpack=True)
    P0hist, P0ranges = np.histogram(np.log10(catP0), bins=bins)
    P0hist = P0hist.astype(float)

    # Simulator parameters:
    argv = Argv()
    mdist = solve.mdist_raghavan()
    qdist = solve.qdist_raghavan()

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        exit()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[boundaries, P0ranges, P0hist, obsEB_ecc, mdist, qdist, argv], pool=pool)

    for result in sampler.sample(p0, iterations=niter, storechain=False):
        position = result[0]
        f = open('params.mcmc', "a")
        for k in range(position.shape[0]):
            f.write("%d %s %f\n" % (k, " ".join(['%.12f' % i for i in position[k]]), result[1][k]))
        f.close()

    pool.close()

if __name__ == '__main__':
    # BASIC SETUP:
    chain_file = 'params.mcmc'
    nwalkers = 256
    niters = 1000
    state = None
    bins = 20

    # Load up the underlying period distribution from DC as a starting point.
    ulogPr, ulogP, ulogPerr = np.loadtxt('ulogP.dist', unpack=True)
    if bins != 30:
        P0s = np.loadtxt('kepEBs.csv', delimiter=',', usecols=(1,))
        uP0hist, uP0ranges = np.histogram(np.log10(P0s), bins=bins)
        ulogP = np.interp(uP0ranges[:-1]+0.5*(uP0ranges[1]-uP0ranges[0]), ulogPr, ulogP)

    boundaries = [(0, 1)]*20 + [(0.7, 1.0)] + [(2.0, 6.0), (1.0, 5.0), (0.15, 0.35), (0.9, 1.0)]
    starting_point = np.append(ulogP, np.array([0.85, 3.5, 3.0, 0.23, 0.98]))

    run(starting_point, boundaries, bins, state, nwalkers, niters)
