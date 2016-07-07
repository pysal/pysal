from scipy.sparse import kron
from pysal import weights as w

def ODW(Wo, Wd, transform='r'):
    """
    Constructs an o*d by o*d origin-destination style spatial weight for o*d
    flows using standard spatial weights on o origins and d destinations. Input
    spatial weights must be binary or able to be sutiably transformed to binary.

    Parameters
    ----------
    Wo          : W object for origin locations
                  o x o spatial weight object amongst o origins
    Wd          : W object for destination locations
                  d x d spatial weight object amongst d destinations
    transform   : Transformation for standardization of final OD spatial weight; default
                  is 'r' for row standardized
    Returns
    -------
    W           : W object for flows
                 o*d x o*d spatial weight object amongst o*d flows between o
                 origins and d destinations
    """
    if Wo.transform is not 'b':
        try:
    	    Wo.tranform = 'b'
        except:
            raise AttributeError('Wo is not binary and cannot be transformed to '
                    'binary. Wo must be binary or suitably transformed to binary.')
    if Wd.transform is not 'b':
        try:
    	    Wd.tranform = 'b'
        except:
            raise AttributeError('Wd is not binary and cannot be transformed to '
                   'binary. Wd must be binary or suitably transformed to binary.')
    Wo = Wo.sparse
    Wd = Wd.sparse
    Ww = kron(Wo, Wd)
    Ww = w.WSP2W(w.WSP(Ww))
    Ww.transform = transform
    return Ww
