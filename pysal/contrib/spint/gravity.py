# coding=utf-8
"""
MLE calibration for Wilson (1967) family of gravity models

References
----------

Fotheringham, A. S. and O'Kelly, M. E. (1989). Spatial Interaction Models: Formulations
 and Applications. London: Kluwer Academic Publishers.

Williams, P. A. and A. S. Fotheringham (1984), The Calibration of Spatial Interaction
 Models by Maximum Likelihood Estimation with Program SIMODEL, Geographic Monograph
 Series, 7, Department of Geography, Indiana University.

Wilson, A. G. (1967). A statistical theory of spatial distribution models.
 Transportation Research, 1, 253–269.


"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import gravity_stats as stats


class Unconstrained:
    """
    Unconstrained gravity model estimated using MLE

    Parameters
    ----------
    data            : str
                      pandas d frame
    origins         : str
                      column name for origin names
    destinations    : str
                      column name for destination names
    flows           : str
                      column name for observed flows
    o_factors       : list of strings
                      column name for each origin attribute
    d_factors       : list of strings
                      column name for each destination attribute
    cost            : str
                      column name for distance or cost values
    cost_func       : str
                      either 'exp' for exponential or 'pow' for power distance function
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      model input data
    o               : pandas series
                      origins
    d               : pandas series
                      destinations
    f               : pandas series
                      observed flows
    of              : dict
                      origin factors
    df              : dict
                      destination factors
    c               : pandas series
                      cost or distance variable
    cf              : str
                      either 'exp' for exponential or 'pow' for power cost function
    opt_log         : dict
                      record of minimization of o_function
    ip              : dict
                      initial parameter estimate values
    p               : dict
                      parameters estimates
    ests            : list of floats
                      estimated values for calibrated model

    Example
    -------

    import numpy as np
    import pandas as pd
    import gravity as grav
    >>> f = np.array([56, 100.8, 173.6, 235.2, 87.36,
                    28., 100.8, 69.44, 235.2, 145.6,
                    22., 26.4, 136.4, 123.2, 343.2,
                    14., 75.6, 130.2, 70.56, 163.8,
                    22, 59.4,  204.6,  110.88,  171.6])
    >>> V = np.repeat(np.array([56, 56, 44, 42, 66]), 5)
    >>> o = np.repeat(np.array(range(1, 6)), 5)
    >>> W = np.tile(np.array([10, 18, 62, 84, 78]), 5)
    >>> d = np.tile(np.array(range(1, 6)), 5)
    >>> dij = np.array([10, 10, 20, 20, 50,
                        20, 10, 50, 20, 30,
                        20, 30, 20, 30, 10,
                        30, 10, 20, 50, 20,
                        30, 20, 20, 50, 30])
    >>> dt = pd.DataFrame({'origins': self.o,
                            'destinations': self.d,
                            'V': V,
                            'W': W,
                            'Dij': dij,
                            'flows': f})
    >>> model = grav.Unconstrained(dt, 'origins', 'destinations', 'flows',
                                    ['V'], ['W'], 'Dij', 'pow')
    >>> print model.p
    {'beta': -1.0, 'W': 1.0, 'V': 1.0}

    """

    def __init__(self, data, origins, destinations, flows, o_factors, d_factors, cost, cost_func, filter_intra=True):
        if filter_intra:
            self.dt = data[data[origins] != data[destinations]].reset_index(level=0, drop=True)
        else:
            self.dt = data
        self.o = self.dt[origins].astype(str)
        self.d = self.dt[destinations].astype(str)
        self.f = self.dt[flows]
        self.c = self.dt[cost]
        self.of = dict(zip(o_factors, [self.dt[x] for x in o_factors]))
        self.df = dict(zip(d_factors, [self.dt[x] for x in d_factors]))
        self.cf = cost_func
        self.opt_log = {}
        self.ip = {'beta': 0}
        self.p = {'beta': 0}

        if self.cf == 'exp':
            k = self.c
        elif self.cf == 'pow':
            k = np.log(self.c)
        else:
            raise ValueError('variable cost_func must either be "exp" or "pow"')

        for fx in self.of:
            k += np.log(self.of[fx])
            self.ip[fx] = 1
            self.p[fx] = 1
        for fx in self.df:
            k += np.log(self.df[fx])
            self.ip[fx] = 1
            self.p[fx] = 1

        o_info = np.sum(self.f*k)

        self.p, self.ests = calibrate(self, o_info, k)
        errors = output(self)
        self.dt['aboslute_error'] = errors[0]
        self.dt['percent_error'] = errors[1]
        stats = statistics(self)
        self.system_stats = stats[0]
        self.entropy_stats = stats[1]
        self.fit_stats = stats[2]
        self.parameter_stats = stats[3]

    def calc_dcy(self, c, cf, p):
        if cf == 'exp':
            dcy = np.exp(c*p['beta'])
        elif cf == 'pow':
            dcy = c**p['beta']
        return dcy

    def estimate_flows(self, c, cf, of, df, p):
        """
        estimate predicted flows multiplying individual model terms
        """
        dcy = self.calc_dcy(c, cf, p)
        ests = dcy
        for fx in of:
            ests *= (of[fx]**p[fx])

        for fx in df:
            ests *= (df[fx]**p[fx])
        return ests

    def estimate_cum(self, ests, k):
        """
        calculate sum of all estimated flows and log of known 'information'
        being estimated (log likelihood)
        """
        return np.sum(ests*k)

    def build_LL_function(self, gm, LL_fx, of, df, f, cf, beta=False):
        """
        builds model-specifc components of the LL function being evaluated
        """
        dcy = self.calc_dcy(self.c, cf, gm.p)
        fxs = dict(of.items() + df.items())
        fs = 1
        for fx in fxs:
            fs *= fxs[fx]**gm.p[fx]
        if beta:
            if cf == 'exp':
                return np.sum(fs*dcy*LL_fx) - np.sum(f*LL_fx)
            else:
                return np.sum(fs*dcy*np.log(LL_fx)) - np.sum(f*np.log(LL_fx))
        else:
            return np.sum(fs*dcy*np.log(LL_fx)) - np.sum(f*np.log(LL_fx))


class ProductionConstrained(Unconstrained):
    """
    Production-constrained gravity model estimated using MLE

    Parameters
    ----------
    data            : str
                      pandas d frame
    origins         : str
                      column name for origin names
    destinations    : str
                      column name for destination names
    flows           : str
                      column name for observed flows
    d_factors       : list of strings
                      column name for each destination attribute
    cost            : str
                      column name for distance or cost values
    cost_func       : str
                      either 'exp' for exponential or 'pow' for power distance function
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      model input data
    o               : pandas series
                      origins
    d               : pandas series
                      destinations
    f               : pandas series
                      observed flows
    of              : empty dict
                      origin factors
    df              : dict
                      destination factors
    c               : pandas series
                      cost or distance variable
    cf              : str
                      either 'exp' for exponential or 'pow' for power cost function
    opt_log         : dict
                      record of minimization of o_function
    ip              : dict
                      initial parameter estimate values
    p               : dict
                      parameters estimates
    ests            : list of floats
                      estimated values for calibrated model

    Example
    -------

    import numpy as np
    import pandas pd
    import gravity as grav
    >>> f = np.array([0, 6469, 7629, 20036, 4690,
                        6194, 11688, 2243, 8857, 7248,
                        3559, 9221, 10099, 22866, 3388,
                        9986, 46618, 11639, 1380, 5261,
                        5985, 6731, 2704, 12250, 16132])
    >>> o = np.repeat(1, 25)
    >>> d = np.array(range(1, 26))
    >>> dij = np.array([0, 576, 946, 597, 373,
                        559, 707, 1208, 602, 692,
                        681, 1934, 332, 595, 906,
                        425, 755, 672, 1587, 526,
                        484, 2141, 2182, 410, 540])
    >>> pop = np.array([1596000, 2071000, 3376000, 6978000, 1345000,
                        2064000, 2378000, 1239000, 4435000, 1999000,
                        1274000, 7042000, 834000, 1268000, 1965000,
                        1046000, 12131000, 4824000, 969000, 2401000,
                        2410000, 2847000, 1425000, 1089000, 2909000])
    >>> dt = pd.DataFrame({'origins': self.o,
                            'destinations': self.d,
                            'pop': self.pop,
                            'Dij': self.dij,
                            'flows': self.f})
    >>> model = grav.ProductionConstrained(self.dt, 'origins', 'destinations', 'flows',
            ['pop'], 'Dij', 'pow')
    {'beta': -0.7365098, 'pop': 0.7818262}

    """
    def __init__(self, data, origins, destinations, flows, d_factors, cost, cost_func, filter_intra=True):
        if filter_intra:
            self.dt = data[data[origins] != data[destinations]].reset_index(level=0, drop=True)
        else:
            self.dt = data
        self.o = self.dt[origins].astype(str)
        self.d = self.dt[destinations].astype(str)
        self.f = self.dt[flows]
        self.c = self.dt[cost]
        self.of = {}
        self.df = dict(zip(d_factors, [self.dt[x] for x in d_factors]))
        self.cf = cost_func
        self.opt_log = {}
        self.ip = {'beta': 0}
        self.p = {'beta': 0}
        self.dt['Oi'] = total_flows(self.dt, flows, self.o)

        if self.cf == 'exp':
            k = self.c
        elif self.cf == 'pow':
            k = np.log(self.c)
        else:
            raise ValueError('variable cost_func must either be "exp" or "pow"')

        for fx in self.df:
            k += np.log(self.df[fx])
            self.ip[fx] = 1
            self.p[fx] = 1

        o_info = np.sum(self.f*k)

        self.p, self.ests = calibrate(self, o_info, k)
        errors = output(self)
        self.dt['aboslute_error'] = errors[0]
        self.dt['percent_error'] = errors[1]
        stats = statistics(self)
        self.system_stats = stats[0]
        self.entropy_stats = stats[1]
        self.fit_stats = stats[2]
        self.parameter_stats = stats[3]

    def estimate_flows(self, c, cf, of, df, p):
        """
        estimate predicted flows multiplying individual model terms
        """
        dcy = self.calc_dcy(c, cf, p)
        self.dt['Ai'] = self.calc_Ai(self.dt, self.o, df, p)
        ests = self.dt['Oi']*self.dt['Ai']*dcy
        for fx in df:
            ests *= (df[fx]**p[fx])
        return ests

    def calc_Ai(self, dt, o, df, p, dc=False):
        """
        calculate Ai balancing factor
        """
        Ai = self.calc_dcy(self.c, self.cf, p)

        if df:
            for fx in df:
                Ai *= df[fx]**p[fx]

        if not dc:
            dt['Ai'] = Ai
        else:
            dt['Ai'] = Ai*dt['Bj']*dt['Dj']
        Ai = (dt.groupby(o).aggregate({'Ai': np.sum}))
        Ai['Ai'] = 1/Ai['Ai']
        Ai = Ai.ix[pd.match(o, Ai.index), 'Ai']
        return Ai.reset_index(level=0, drop=True)

    def build_LL_function(self, gm, LL_fx, of, df, f, cf, beta=False):
        """
        builds model-specifc components of the LL function being evaluated
        """
        self.dt['Ai'] = self.calc_Ai(self.dt, self.o, df, self.p)
        dcy = self.calc_dcy(self.c, cf, gm.p)
        fxs = dict(of.items() + df.items())
        fs = 1
        for fx in fxs:
            fs *= fxs[fx]**gm.p[fx]
        Ai = self.dt['Ai']
        Oi = self.dt['Oi']
        if beta:
            if cf == 'exp':
                return np.sum(Ai*Oi*fs*dcy*LL_fx) - np.sum(f*LL_fx)
            else:
                return np.sum(Ai*Oi*fs*dcy*np.log(LL_fx)) - np.sum(f*np.log(LL_fx))
        else:
            return np.sum(Ai*Oi*fs*dcy*np.log(LL_fx)) - np.sum(f*np.log(LL_fx))


class AttractionConstrained(Unconstrained):
    """
    Attraction-constrained gravity model estimated using MLE

    Parameters
    ----------
    data            : str
                      pandas d frame
    origins         : str
                      column name for origin names
    destinations    : str
                      column name for destination names
    flows           : str
                      column name for observed flows
    o_factors       : list of strings
                      column name for each origin attribute
    cost            : str
                      column name for distance or cost values
    cost_func       : str
                      either 'exp' for exponential or 'pow' for power distance function
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      model input data
    o               : pandas series
                      origins
    d               : pandas series
                      destinations
    f               : pandas series
                      observed flows
    of              : dict
                      origin factors
    df              : None
                      destination factors
    c               : pandas series
                      cost or distance variable
    cf              : str
                      either 'exp' for exponential or 'pow' for power cost function
    opt_log         : dict
                      record of minimization of o_function
    ip              : dict
                      initial parameter estimate values
    p               : dict
                      parameters estimates
    ests            : list of floats
                      estimated values for calibrated model

    Example
    -------

    import numpy as np
    import pandas as pd
    import gravity as grav
    >>> f = np.array([56, 100.8, 173.6, 235.2, 87.36,
                        28., 100.8, 69.44, 235.2, 145.6,
                        22., 26.4, 136.4, 123.2, 343.2,
                        14., 75.6, 130.2, 70.56, 163.8,
                        22, 59.4,  204.6,  110.88,  171.6])
    >>> V = np.repeat(np.array([56, 56, 44, 42, 66]), 5)
    >>> o = np.repeat(np.array(range(1, 6)), 5)
    >>> W = np.tile(np.array([10, 18, 62, 84, 78]), 5)
    >>> d = np.tile(np.array(range(1, 6)), 5)
    >>> dij = np.array([10, 10, 20, 20, 50,
                        20, 10, 50, 20, 30,
                        20, 30, 20, 30, 10,
                        30, 10, 20, 50, 20,
                        30, 20, 20, 50, 30])
    >>> dt = pd.DataFrame({'origins': self.o,
                            'destinations': self.d,
                            'V': self.V,
                            'Dij': self.dij,
                            'flows': self.f})
    >>> model = grav.AttractionConstrained(self.dt, 'origins', 'destinations', 'flows',
                                            ['V'], 'Dij', 'pow')
    >>> print model.p
    {'beta': -1.0, 'V': 1.0}

    """

    def __init__(self, data, origins, destinations, flows, o_factors, cost, cost_func, filter_intra=True):
        if filter_intra:
            self.dt = data[data[origins] != data[destinations]].reset_index(level=0, drop=True)
        else:
            self.dt = data
        self.o = self.dt[origins].astype(str)
        self.d = self.dt[destinations].astype(str)
        self.f = self.dt[flows]
        self.c = self.dt[cost]
        self.of = dict(zip(o_factors, [self.dt[x] for x in o_factors]))
        self.df = {}
        self.cf = cost_func
        self.opt_log = {}
        self.ip = {'beta': 0}
        self.p = {'beta': 0}
        self.dt['Dj'] = total_flows(self.dt, flows, self.d)

        if self.cf == 'exp':
            k = self.c
        elif self.cf == 'pow':
            k = np.log(self.c)
        else:
            raise ValueError('variable cost_func must either be "exp" or "pow"')

        for fx in self.of:
            k += np.log(self.of[fx])
            self.ip[fx] = 1
            self.p[fx] = 1

        o_info = np.sum(self.f*k)

        self.p, self.ests = calibrate(self, o_info, k)
        errors = output(self)
        self.dt['aboslute_error'] = errors[0]
        self.dt['percent_error'] = errors[1]
        stats = statistics(self)
        self.system_stats = stats[0]
        self.entropy_stats = stats[1]
        self.fit_stats = stats[2]
        self.parameter_stats = stats[3]

    def estimate_flows(self, c, cf, of, df, p):
        """
        estimate predicted flows multiplying individual model terms
        """
        dcy = self.calc_dcy(c, cf, p)
        self.dt['Bj'] = self.calc_Bj(self.dt, self.d, of, p)
        ests = self.dt['Dj']*self.dt['Bj']*dcy
        for fx in of:
            ests *= (of[fx]**p[fx])
        return ests

    def calc_Bj(self, dt, d, of, p, dc=False):
        """
        calculate Bj balancing factor
        """
        Bj = self.calc_dcy(self.c, self.cf, p)

        if of:
            for fx in of:
                Bj *= of[fx]**p[fx]
        if not dc:
            dt['Bj'] = Bj
        else:
            dt['Bj'] = Bj*dt['Ai']*dt['Oi']
        Bj = (dt.groupby(d).aggregate({'Bj': np.sum}))
        Bj['Bj'] = 1/Bj['Bj']
        Bj = Bj.ix[pd.match(d, Bj.index), 'Bj']
        return Bj.reset_index(level=0, drop=True)

    def build_LL_function(self, gm, LL_fx, of, df, f, cf, beta=False):
        """
        builds model-specifc components of the LL function being evaluated
        """
        self.dt['Bj'] = self.calc_Bj(self.dt, self.d, of, self.p)
        dcy = self.calc_dcy(self.c, cf, gm.p)
        fxs = dict(of.items() + df.items())
        fs = 1
        for fx in fxs:
            fs *= fxs[fx]**gm.p[fx]
        Bj = self.dt['Bj']
        Dj = self.dt['Dj']
        if beta:
            if cf == 'exp':
                return np.sum(Bj*Dj*fs*dcy*LL_fx) - np.sum(f*LL_fx)
            else:
                return np.sum(Bj*Dj*fs*dcy*np.log(LL_fx)) - np.sum(f*np.log(LL_fx))
        else:
            return np.sum(Bj*Dj*fs*dcy*np.log(LL_fx)) - np.sum(f*np.log(LL_fx))


class DoublyConstrained(ProductionConstrained, AttractionConstrained):
    """
    Doubly-constrained gravity model estimated using MLE

    Parameters
    ----------
    data            : str
                      pandas d frame
    origins         : str
                      column name for origin names
    destinations    : str
                      column name for destination names
    flows           : str
                      column name for observed flows
                      column name for each destination attribute
    cost            : str
                      column name for distance or cost values
    cost_func       : str
                      either 'exp' for exponential or 'pow' for power distance function
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      model input data
    o               : pandas series
                      origins
    d               : pandas series
                      destinations
    f               : pandas series
                      observed flows
    of              : None
                      origin factors
    df              : None
                      destination factors
    c               : pandas series
                      cost or distance variable
    cf              : str
                      either 'exp' for exponential or 'pow' for power cost function
    opt_log         : dict
                      record of minimization of o_function
    ip              : dict
                      initial parameter estimate values
    p               : dict
                      parameters estimates
    ests            : list of floats
                      estimated values for calibrated model

    Example
    -------
    import numpy as np
    import pandas as pd
    import gravity as grav
    >>> f = np.array([0, 180048, 79223, 26887, 198144, 17995, 35563, 30528, 110792,
                        283049, 0, 300345, 67280, 718673, 55094, 93434, 87987, 268458,
                        87267, 237229, 0, 281791, 551483, 230788, 178517, 172711, 394481,
                        29877, 60681, 286580, 0, 143860, 49892, 185618, 181868, 274629,
                        130830, 382565, 346407, 92308, 0, 252189, 192223, 89389, 279739,
                        21434, 53772, 287340, 49828, 316650, 0, 141679, 27409, 87938,
                        30287, 64645, 161645, 144980, 199466, 121366, 0, 134229, 289880,
                        21450, 43749, 97808, 113683, 89806, 25574, 158006, 0, 437255,
                        72114, 133122, 229764, 165405, 266305, 66324, 252039, 342948, 0])
    >>> o = np.repeat(np.array(range(1, 10)), 9)
    >>> d = np.tile(np.array(range(1, 10)), 9)
    >>> dij = np.array([0, 219, 1009, 1514, 974, 1268, 1795, 2420, 3174,
                        219, 0, 831, 1336, 755, 1049, 1576, 2242, 2996,
                        1009, 831, 0, 505, 1019, 662, 933, 1451, 2205,
                        1514, 1336, 505, 0, 1370, 888, 654, 946, 1700,
                        974, 755, 1019, 1370, 0, 482, 1144, 2278, 2862,
                        1268, 1049, 662, 888, 482, 0, 662, 1795, 2380,
                        1795, 1576, 933, 654, 1144, 662, 0, 1287, 1779,
                        2420, 2242, 1451, 946, 2278, 1795, 1287, 0, 754,
                        3147, 2996, 2205, 1700, 2862, 2380, 1779, 754, 0])
    >>> dt = pd.DataFrame({'Origin': self.o,
                            'Destination': self.d,
                            'flows': self.f,
                            'Dij': self.dij})
    >>> model = grav.DoublyConstrained(self.dt, 'Origin', 'Destination', 'flows', 'Dij', 'exp')
    >>> print model.p
    {'beta': -0.0007369}

    """

    def __init__(self, data, origins, destinations, flows, cost, cost_func, filter_intra=True):
        if filter_intra:
            self.dt = data[data[origins] != data[destinations]].reset_index(level=0, drop=True)
        else:
            self.dt = data
        self.o = self.dt[origins].astype(str)
        self.d = self.dt[destinations].astype(str)
        self.f = self.dt[flows]
        self.c = self.dt[cost]
        self.of = {}
        self.df = {}
        self.cf = cost_func
        self.opt_log = {}
        self.ip = {'beta': 0}
        self.p = {'beta': 0}
        self.dt['Bj'] = 1.0
        self.dt['Ai'] = 1.0
        self.dt['OldAi'] = 10.000000000
        self.dt['OldBj'] = 10.000000000
        self.dt['diff'] = abs((self.dt['OldAi'] - self.dt['Ai'])/self.dt['OldAi'])
        self.dt['Oi'] = total_flows(self.dt, flows, self.o)
        self.dt['Dj'] = total_flows(self.dt, flows, self.d)

        if self.cf == 'exp':
            k = self.c
        elif self.cf == 'pow':
            k = np.log(self.c)
        else:
            raise ValueError('variable cost_func must either be "exp" or "pow"')

        o_info = np.sum(self.f*k)

        self.p, self.ests = calibrate(self, o_info, k)
        errors = output(self)
        self.dt['aboslute_error'] = errors[0]
        self.dt['percent_error'] = errors[1]
        stats = statistics(self)
        self.system_stats = stats[0]
        self.entropy_stats = stats[1]
        self.fit_stats = stats[2]
        self.parameter_stats = stats[3]

    def estimate_flows(self, c, cf, of, df, p):
        """
        estimate predicted flows multiplying individual model terms
        """
        dcy = self.calc_dcy(c, cf, p)
        self.dt['Ai'], self.dt['Bj'] = self.balance_factors(self.dt, self.o, self.d, of, df, p)
        ests = self.dt['Dj']*self.dt['Bj']*self.dt['Oi']*self.dt['Ai']*dcy
        return ests

    def build_LL_function(self, gm, LL_fx, of, df, f, cf, beta=False):
        """
        builds model-specifc components of the LL function being evaluated
        """
        self.dt['Ai'], self.dt['Bj'] = self.balance_factors(self.dt, self.o, self.d, of, df, gm.p)
        dcy = self.calc_dcy(self.c, cf, gm.p)
        Ai = self.dt['Ai']
        Oi = self.dt['Oi']
        Bj = self.dt['Bj']
        Dj = self.dt['Dj']
        if beta:
            if cf == 'exp':
                return np.sum(Ai*Oi*Bj*Dj*dcy*LL_fx) - np.sum(f*LL_fx)
            else:
                return np.sum(dcy*np.log(LL_fx)) - np.sum(f*np.log(LL_fx))
        else:
            return np.sum(dcy*np.log(LL_fx)) - np.sum(f*np.log(LL_fx))

    def balance_factors(self, dt, o, d, of, df, p):
        """
        calculate balancing factors and balance the balancing factors
        if doubly-constrained model
        """
        its = 0
        cnvg = 1
        while cnvg > .0001:
            its += 1
            dt['Ai'] = self.calc_Ai(dt, o, df, p, dc=True)
            if its == 1:
                dt['OldAi'] = dt['Ai']
            else:
                dt['diff'] = abs((dt['OldAi'] - dt['Ai'])/dt['OldAi'])
                dt['OldAi'] = dt['Ai']
            dt['Bj'] = self.calc_Bj(dt, d, of, p, dc=True)
            if its == 1:
                dt['OldBj'] = dt['Bj']
            else:
                dt['diff'] = abs((dt['OldBj'] - dt['Bj'])/dt['OldBj'])
                dt['OldBj'] = dt['Bj']
            cnvg = np.sum(dt['diff'])
        return dt['Ai'], dt['Bj']


def total_flows(dt, f, locs):
    """
    sum rows or columns to derive total inflows or total outflows
    """

    totals = dt.groupby(locs).aggregate({f: np.sum})
    return totals.ix[pd.match(locs, totals.index.astype(str))].reset_index()[f]


def o_function(pv, gm, cf, of, df):
    """
    evaluates log-likelihood functions for each parameter being estimated.
    Used in optimization/calibration and statistics.

    prepares and builds the constant terms across a model and passes it
    to build_LL_function to finish
    """
    for x, px in enumerate(gm.p):
        gm.p[px] = pv[x]
    funcs = []
    fxs = dict(of.items() + df.items())
    for fx in gm.p:
        if fx == 'beta':
            funcs.append(gm.build_LL_function(gm, gm.c, of, df, gm.f, gm.cf, beta=True))
        else:
            funcs.append(gm.build_LL_function(gm, fxs[fx], of, df, gm.f, gm.cf))
    return funcs


def calibrate(gm, o_info, k):
    """
    run the main routine which estimates parameters using mle
    """
    ests = gm.estimate_flows(gm.c, gm.cf, gm.of, gm.df, gm.p)
    e_info = gm.estimate_cum(ests, k)
    its = 0
    diff = abs(e_info - o_info)
    try:
        while diff > .1:
            gm.pv = gm.p.values()
            gm.opt_log[its] = [gm.pv, diff]
            gm.pv = fsolve(o_function, gm.pv, (gm, gm.cf, gm.of, gm.df))
            ests = gm.estimate_flows(gm.c, gm.cf, gm.of, gm.df, gm.p)
            e_info = gm.estimate_cum(ests, k)
            its += 1
            diff = abs(e_info - o_info)
            if its > 25:
                print 'Convergence criterion not met, solution may not be optimal'
                break
        gm.opt_log[its] = [gm.pv, diff]
        gm.p = dict(zip(gm.p.keys(), [round(x, 7) for x in gm.pv]))
        return gm.p, ests
    except:
        raise RuntimeError('Optimization could not be carried out, check input validity')
        return gm.p, ests


def output(gm):
    """
    prepare output
    """
    abs_err = gm.ests - gm.f
    perc_err = (abs_err/gm.f) * 100
    return abs_err, perc_err


def statistics(gm):
    """
    calculate statistics
    """
    if 'Ai' in gm.dt.columns:
        Ai = gm.dt.Ai.values.copy()
    if 'Bj' in gm.dt.columns:
        Bj = gm.dt.Bj.values.copy()
    ests = gm.ests.values.copy()
    p = gm.p.copy()

    system_stats = stats.sys_stats(gm)
    entropy_stats = stats.ent_stats(gm)
    fit_stats = stats.fit_stats(gm)
    parameter_statistics = stats.param_stats(gm)

    if 'Ai' in gm.dt.columns:
        gm.dt.Ai = Ai
    if 'Bj' in gm.dt.columns:
        gm.dt.Bj = Bj
    gm.ests = ests
    gm.dt.ests = ests
    gm.p = p

    return system_stats, entropy_stats, fit_stats, parameter_statistics