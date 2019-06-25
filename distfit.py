#!/usr/bin/env python
#title           :distribution_check.py
#description     :Checks a sample against 80 distributions by applying the Kolmogorov-Smirnov test.
#author          :Andre Dietrich
#email           :dietrich@ivs.cs.uni-magdeburg.de
#date            :07.10.1014
#version         :0.2
#usage           :python distribution_check.py -f filename -v
#python_version  :2.*
#########################################################################################
from math import isnan
import scipy.stats
import warnings
# just for surpressing warnings
warnings.simplefilter('ignore')

#########################################################################################
# this part is only used to read in the file ...
# format should be :
# 0.00192904472351
# 0.0030369758606
# 0.00188088417053
# 0.00222492218018
# 0.00447607040405
# 0.00301194190979
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", help="file with measurement data", type="string", metavar="FILE")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="print all results immediately (default=False)", default=False)
parser.add_option("-t", "--top", dest="top", help="define amount of printed results (default=10)", type="int", default=10)
parser.add_option("-p", "--plot", dest="plot", help="plot the best result with matplotlib (default=False)", action="store_true", default=False)

(options, args) = parser.parse_args()
if hasattr(options, "filename"):
    print "reading data in file ", options.filename, "...",
    f = open(options.filename)
    data = [float(value) for value in f.readlines()]
    f.close()
    print "done"
else:
    print "generating random values for halflogistic ...",
    data = scipy.stats.halflogistic(1,1).rvs(500)
    print "done"
########################################################################################
  
# list of all available distributions
cdfs = [
    "norm",            #Normal (Gaussian)
    "alpha",           #Alpha
    "anglit",          #Anglit
    "arcsine",         #Arcsine
    "beta",            #Beta
    "betaprime",       #Beta Prime
    "bradford",        #Bradford
    "burr",            #Burr
    "cauchy",          #Cauchy
    "chi",             #Chi
    "chi2",            #Chi-squared
    "cosine",          #Cosine
    "dgamma",          #Double Gamma
    "dweibull",        #Double Weibull
    "erlang",          #Erlang
    "expon",           #Exponential
    "exponweib",       #Exponentiated Weibull
    "exponpow",        #Exponential Power
    "fatiguelife",     #Fatigue Life (Birnbaum-Sanders)
    "foldcauchy",      #Folded Cauchy
    "f",               #F (Snecdor F)
    "fisk",            #Fisk
    "foldnorm",        #Folded Normal
    "frechet_r",       #Frechet Right Sided, Extreme Value Type II
    "frechet_l",       #Frechet Left Sided, Weibull_max
    "gamma",           #Gamma
    "gausshyper",      #Gauss Hypergeometric
    "genexpon",        #Generalized Exponential
    "genextreme",      #Generalized Extreme Value
    "gengamma",        #Generalized gamma
    "genlogistic",     #Generalized Logistic
    "genpareto",       #Generalized Pareto
    "genhalflogistic", #Generalized Half Logistic
    "gilbrat",         #Gilbrat
    "gompertz",        #Gompertz (Truncated Gumbel)
    "gumbel_l",        #Left Sided Gumbel, etc.
    "gumbel_r",        #Right Sided Gumbel
    "halfcauchy",      #Half Cauchy
    "halflogistic",    #Half Logistic
    "halfnorm",        #Half Normal
    "hypsecant",       #Hyperbolic Secant
    "invgamma",        #Inverse Gamma
    "invgauss",        #Inverse Normal
    "invweibull",      #Inverse Weibull
    "johnsonsb",       #Johnson SB
    "johnsonsu",       #Johnson SU
    "laplace",         #Laplace
    "logistic",        #Logistic
    "loggamma",        #Log-Gamma
    "loglaplace",      #Log-Laplace (Log Double Exponential)
    "lognorm",        #Log-Normal
    "lomax",           #Lomax (Pareto of the second kind)
    "maxwell",         #Maxwell
    "mielke",          #Mielke's Beta-Kappa
    "nakagami",        #Nakagami
    "ncx2",            #Non-central chi-squared
    "ncf",             #Non-central F
    "nct",             #Non-central Student's T
    "pareto",          #Pareto
    "powerlaw",        #Power-function
    "powerlognorm",    #Power log normal
    "powernorm",       #Power normal
    "rdist",           #R distribution
    "reciprocal",      #Reciprocal
    "rayleigh",        #Rayleigh
    "rice",            #Rice
    "recipinvgauss",   #Reciprocal Inverse Gaussian
    "semicircular",    #Semicircular
    "t",               #Student's T
    "triang",          #Triangular
    "truncexpon",      #Truncated Exponential
    "truncnorm",       #Truncated Normal
    "tukeylambda",     #Tukey-Lambda
    "uniform",         #Uniform
    "vonmises",        #Von-Mises (Circular)
    "wald",            #Wald
    "weibull_min",     #Minimum Weibull (see Frechet)
    "weibull_max",     #Maximum Weibull (see Frechet)
    "wrapcauchy",      #Wrapped Cauchy
    "ksone",           #Kolmogorov-Smirnov one-sided (no stats)
    "kstwobign"]       #Kolmogorov-Smirnov two-sided test for Large N


result = []
for cdf in cdfs:

    #fit our data set against every probability distribution
    parameters = eval("scipy.stats."+cdf+".fit(data, floc=0)");
 
    #Applying the Kolmogorov-Smirnof two sided test
    D, p = scipy.stats.kstest(data, cdf, args=parameters);
        
    #print the results
    if options.verbose:
        print cdf.ljust(16) + ("p: "+str(p)).ljust(25)+"D: "+str(D);

    result.append( (cdf, p, D) )
     
print "-------------------------------------------------------------------"
print "Top", options.top
print "-------------------------------------------------------------------"

best = sorted(result, key=lambda elem: -1 if isnan(elem[1]) else elem[1], reverse=True)
for t in range(options.top):
    print str(t+1).ljust(4), best[t][0].ljust(16), "\tp: ", best[t][1],"\tD: ", best[t][2]

if options.plot:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # plot data
    plt.hist(data, normed=True, bins=np.linspace(0, 0.1, 30), alpha=0.3)
    
    # plot fitted probability
    for t in range(options.top):
        params = eval("scipy.stats."+best[t][0]+".fit(data)")
        fct = eval("scipy.stats."+best[t][0]+".freeze"+str(params))
        #x = np.linspace(fct.ppf(0.001), fct.ppf(0.999), 500)
        x = np.linspace(0, 0.1, 100)
        plt.plot(x, fct.pdf(x), lw=3, label=best[t][0])
    plt.legend(loc='best', frameon=False)
    plt.title("Top "+str(options.top)+" Results")
    plt.show()

