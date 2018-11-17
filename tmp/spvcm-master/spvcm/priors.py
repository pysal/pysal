
import scipy.stats as stats

def Constant():
    """
    A constructor for a rescaled flat log probability density function.
    This will simply return 0 always. 
    """
    def constant(*arg, **args):
        return 0
    return constant

def constant(*args, **kwargs):
    return 0

def Beta(shapea, shapeb, bounds=(-1,1)):
    """
    A constructor for a rescaled beta log proability density function 
    to fit between a given bounds. By default, this covers -1,1.

    Parameters
    -----------
    shapea  :   float
            first shape parameter for the beta distribution
    shapeb  :   float
            second shape parameter for the beta distribution
    bounds  :   tuple of floats
            left and right boundary of the truncated log prior distribution
    """
    a = bounds[0]
    b = bounds[1] - a
    return stats.beta(a=shapea, b=shapeb,loc=a,scale=b).logpdf

beta22 = Beta(2,2)
beta105 = Beta(10,5)
beta510 = Beta(5,10)

def Truncnorm(mean, dev, bounds=(-1,1)):
    """
    A rescaled truncated normal distribution to fit between a given bounds.
    By default, this covers -1,1
    Parameters
    ----------
    mean    :   float
                mean of truncated normal
    dev     :   float
                standard deviation of the truncated normal
    bounds  :   tuple of floats
                left and right bounds of the truncated normal distribution
    """
    clipa, clipb = bounds
    a,b = (clipa - mean)/float(dev), (clipb - mean)/float(dev)

    return stats.truncnorm(loc=mean, scale=dev, a=a,b=b).logpdf

truncnorm_std = Truncnorm(0,1)
truncnorm_positive = Truncnorm(0,.5,bounds=(0,1))
truncnorm_narrow = Truncnorm(0,.25)

