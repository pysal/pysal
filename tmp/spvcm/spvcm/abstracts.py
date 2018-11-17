import warnings
from datetime import datetime as dt
import numpy as np
import copy
import multiprocessing as mp
import pandas as pd
import os


from .sqlite import head_to_sql, start_sql
from .plotting import plot_trace
from collections import OrderedDict
try:
    from tqdm import tqdm
    import six
    if not six.PY3:
        range = xrange
except ImportError:
    from .utils import thru_op
    tqdm = thru_op

__all__ = ['Sampler_Mixin', 'Hashmap', 'Trace']

######################
# SAMPLER MECHANISMS #
######################


class Sampler_Mixin(object):
    """
    A Mixin class designed to facilitate code reuse. This should be the parent class of anything that uses the sampling framework in this package.
    """
    def __init__(self):
        super(Sampler_Mixin, self).__init__()

    def sample(self, n_samples, n_jobs=1):
        """
        Sample from the joint posterior distribution defined by all of the
        parameters in the gibbs sampler.

        Parameters
        ----------
        n_samples   :   int
                        number of samples from the joint posterior density to take
        n_jobs      :   int
                        number of parallel chains to run.

        Returns
        -------
        Implicitly updates all values in place, returns None
        """
        if n_jobs > 1:
           self._parallel_sample(n_samples, n_jobs)
           return
        elif isinstance(self.state, list):
            self._parallel_sample(n_samples, n_jobs=len(self.state))
            return
        _start = dt.now()
        try:
            for _ in tqdm(range(n_samples)):
                if (self._verbose > 1) and (n_samples % 100 == 0):
                    print('{} Draws to go'.format(n_samples))
                self.draw()
        except KeyboardInterrupt:
            warnings.warn('Sampling interrupted, drew {} samples'.format(self.cycles))
        finally:
            _stop = dt.now()
            if not hasattr(self, 'total_sample_time'):
                self.total_sample_time = _stop - _start
            else:
                self.total_sample_time += _stop - _start

    def draw(self):
        """
        Take exactly one sample from the joint posterior distribution.
        """
        if self.cycles == 0:
            self._finalize()
        self._iteration()
        self.cycles += 1
        for param in self.traced_params:
            self.trace.chains[0][param].append(self.state[param])
        if self.database is not None:
            head_to_sql(self, self._cur, self._cxn)
            for param in self.traced_params:
                self.trace.chains[0][param] = [self.trace[param,-1]]

    def _parallel_sample(self, n_samples, n_jobs):
        """
        Run n_jobs parallel samples of a given model. 
        Not intended to be called directly, and should be called by model.sample.
        """
        models = [copy.deepcopy(self) for _ in range(n_jobs)]
        for i, model in enumerate(models):
            if isinstance(model.state, list):
                models[i].state = copy.deepcopy(self.state[i])
            if hasattr(model, 'configs'):
                if isinstance(model.configs, list):
                    models[i].configs = copy.deepcopy(self.configs[i])
            if self.database is not None:
                models[i].database = self.database + str(i)
            models[i].trace = Trace(**{k:[] for k in model.trace.varnames})
            if self.cycles == 0:
                models[i]._fuzz_starting_values()
        n_samples = [n_samples] * n_jobs
        _start = dt.now()
        seed = np.random.randint(0,10000, size=n_jobs).tolist()
        P = mp.Pool(n_jobs)
        results = P.map(_reflexive_sample, zip(models, n_samples, seed))
        P.close()
        _stop = dt.now()
        if self.cycles > 0:
            new_traces = []
            for i, model in enumerate(results):
                # model.trace.chains is always single-chain, since we've broken everything into single chains
                new_traces.append(Hashmap(**{k:param + model.trace.chains[0][k]
                                             for k, param in self.trace.chains[i].items()}))
            new_trace = Trace(*new_traces)
        else:
            new_trace = Trace(*[model.trace.chains[0] for model in results])
        self.trace = new_trace
        self.state = [model.state for model in results]
        self.cycles += n_samples[0]
        self.configs = [model.configs for model in results]
        if hasattr(self, 'total_sample_time'):
            self.total_sample_time += _stop - _start
        else:
            self.total_sample_time = _stop - _start

    def _fuzz_starting_values(self, state=None):
        """
        Function to overdisperse starting values used in the package.
        """
        st = self.state
        if hasattr(st, 'Betas'):
            st.Betas += np.random.normal(0,5, size=st.Betas.shape)
        if hasattr(st, 'Alphas'):
            st.Alphas += np.random.normal(0,5,size=st.Alphas.shape)
        if hasattr(st, 'Sigma2'):
            st.Sigma2 += np.random.uniform(0,5)
        if hasattr(st, 'Tau2'):
            st.Tau2 += np.random.uniform(0,5)
        if hasattr(st, 'Lambda'):
            st.Lambda += np.random.uniform(-.25,.25)
        if hasattr(st, 'Rho'):
            st.Rho += np.random.uniform(-.25,.25)

    def _finalize(self, **args):
        """
        Abstract function to ensure inheritors define a finalze method. This method should compute all derived quantities used in the _iteration() function that would change if the user changed priors, starting values, or other information. This is to ensure that if the user initializes the sampler with n_samples=0 and then changes the state, the derived quantites used in sampling are correct.
        """
        raise NotImplementedError

    def _setup_priors(self, **args):
        """
        Abstract function to ensure inheritors define a _setup_priors method. This method should assign into the state all of the correct priors for all parameters in the model.
        """
        raise NotImplementedError

    def _setup_truncation(self, **args):
        """
        Abstract function to ensure inheritors define a _setup_truncation method. This method should truncate parameter space to a given arbitrary bounds.
        """
        raise NotImplementedError

    def _setup_starting_values(self, **args):
        """
        Abstract function to ensure that inheritors define a _setup_starting_values method. This method should assign the correct values for each of the parameters into model.state.
        """
        raise NotImplementedError

    @property
    def database(self):
        """
        the database used for the model.
        """
        return getattr(self, '_db', None)

    @database.setter
    def database(self, filename):
        self._cxn, self._cur = start_sql(self, tracename=filename)
        self._db = filename
        from .sqlite import trace_from_sql
        def load_sqlite():
            return trace_from_sql(filename)
        self.trace.load_sqlite = load_from_sqlite

def _reflexive_sample(tup):
    """
    a helper function sample a bunch of models in parallel.

    Tuple must be:

    model : model object
    n_samples : int number of samples
    seed : seed to use for the sampler
    """
    model, n_samples, seed = tup
    np.random.seed(seed)
    model.sample(n_samples=n_samples)
    return model

def _noop(*args, **kwargs):
    pass

#######################
# MAPS AND CONTAINERS #
#######################

class Hashmap(dict):
    """
    A dictionary with dot access on attributes
    """
    def __init__(self, **kw):
        super(Hashmap, self).__init__(**kw)
        if kw != dict():
            for k in kw:
                self[k] = kw[k]

    def __getattr__(self, attr):
        try:
            r = self[attr]
        except KeyError:
            try:
                r = getattr(super(Hashmap, self), attr)
            except AttributeError:
                raise AttributeError("'{}' object has no attribute '{}'"
                                     .format(self.__class__, attr))
        return r

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Hashmap, self).__setitem__(key,value)
        self.__dict__.update({key:value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Hashmap, self).__delitem__(key)
        del self.__dict__[key]

class Trace(object):
    """
    Object to contain results from sampling.

    Arguments
    ---------
    chains  :   a chain or comma-separated sequence of chains
                a chain is a dict-like collection, where keys are the parameter name and the values are the values of the chain.
    kwargs  :   a dictionary splatted into keyword arguments
                the name of the argument is taken to the be the parameter name, and the value is taken to be a chain of that parameter.

    Examples
    ---------
    >>> Trace(a=[1,2,3], b=[4,2,5], c=[1,9,23]) #Trace with one chain
    >>> Trace([{'a':[1,2,3], 'b':[4,2,5], 'c':[1,9,23]},
               {'a':[2,5,1], 'b':[2,9,1], 'c':[9,21,1]}]) #Trace with two chains
    """
    def __init__(self, *chains, **kwargs):
        if chains is () and kwargs != dict():
            self.chains = _maybe_hashmap(kwargs)
        if chains is not ():
            self.chains = _maybe_hashmap(*chains)
            if kwargs != dict():
                self.chains.extend(_maybe_hashmap(kwargs))
        self._validate_schema()

    @property
    def varnames(self, chain=None):
        """
        Names of variables contained in the trace.
        """
        try:
            return self._varnames
        except AttributeError:
            try:
                self._validate_schema()
            except KeyError:
                if chain is None:
                    raise Exception('Variable names are heterogeneous in chains and no default index provided.')
                else:
                    warnings.warn('Variable names are heterogeneous in chains!', stacklevel=2)
                    return list(self.chains[chain].keys())
            self._varnames = list(self.chains[0].keys())
            return self._varnames

    def drop(self, varnames, inplace=True):
        """
        Drop a variable from the trace.

        Arguments
        ---------
        varnames    :   list of strings
                        names of parameters to drop from the trace.
        inplace     :   bool
                        whether to return a copy of the trace with parameters removed, or remove them inplace.
        """
        if isinstance(varnames, str):
            varnames = (varnames,)
        if not inplace:
            new = copy.deepcopy(self)
            new.drop(varnames, inplace=True)
            new._varnames = list(new.chains[0].keys())
            return new
        for i, chain in enumerate(self.chains):
            for varname in varnames:
                del self.chains[i][varname]
        self._varnames = list(self.chains[0].keys())

    def _validate_schema(self, chains=None):
        """
        Validates the trace to ensure that the chain is self-consistent.
        """
        if chains is None:
            chains = self.chains
        tracked_in_each = [set(chain.keys()) for chain in chains]
        same_schema = [names == tracked_in_each[0] for names in tracked_in_each]
        try:
            assert all(same_schema)
        except AssertionError:
            bad_chains = [i for i in range(len(chains)) if same_schema[i]]
            KeyError('The parameters tracked in each chain are not the same!'
                     '\nChains {} do not have the same parameters as chain 1!'.format(bad_chains))

    def add_chain(self, chains, validate=True):
        """
        Add chains to a trace object

        Parameters
        ----------
        chains  :   Hashmap or list of hashmaps
                    chains to merge into the trace
        validate:   bool
                    whether or not to validate the schema and reject the chain if it does not match the current trace.
        """
        if not isinstance(chains, (list, tuple)):
            chains = (chains,)
        new_chains = [self.chains]
        for chain in chains:
            if isinstance(chain, Hashmap):
                new_chains.append(chain)
            elif isinstance(chain, Trace):
                new_chains.extend(chain.chains)
            else:
                new_chains.extend(_maybe_hashmap(chain))
        if validate:
            self._validate_schema(chains=new_chains)
        self.chains = new_chains

    def map(self, func, **func_args):
        """
        Map a function over all parameters in a chain.
        Multivariate parameters are reduced to sequences of univariate parameters.

        Usage
        -------
        Intended when full-trace statistics are required. Most often,
        the trace should be sliced directly. For example, to get the mean value of a
        parameter over the last -1000 iterations with a thinning of 2:

        trace[0, 'Betas', -1000::2].mean(axis=0)

        but, to average of the parameter over all recorded chains:

        trace['Betas', -1000::2].mean(axis=0).mean(axis=0)

        since the first reduction provides an array where rows
        are iterations and columns are parameters.

        trace.map(np.mean) yields the mean of each parameter within each chain, and is
        provided to make within-chain reductions easier.

        Arguments
        ---------
        func        :   callable
                        a function that returns a result when provided a flat vector.
        varnames    :   string or list of strings
                        a keyword only argument governing which parameters to map over.
        func_args   :   dictionary/keyword arguments
                        arguments needed to be passed to the reduction
        """
        varnames = func_args.pop('varnames', self.varnames)
        if isinstance(varnames, str):
            varnames = (varnames, )
        all_stats = []
        for i, chain in enumerate(self.chains):
            these_stats=dict()
            for var in varnames:
                data = np.squeeze(self[i,var])
                if data.ndim > 1:
                    n,p = data.shape[0:2]
                    rest = data.shape[2:0]
                    if len(rest) == 0:
                        data = data.T
                    elif len(rest) == 1:
                        data = data.reshape(n,p*rest[0]).T
                    else:
                        raise Exception('Parameter "{}" shape not understood.'                  ' Please extract, shape it, and pass '
                                        ' as its own chain. '.format(var))
                else:
                    data = data.reshape(1,-1)
                stats = [func(datum, **func_args) for datum in data]
                if len(stats) == 1:
                    stats = stats[0]
                these_stats.update({var:stats})
            all_stats.append(these_stats)
        return all_stats

    @property
    def n_chains(self):
        return len(self.chains)

    @property
    def n_iters(self):
        """
        Number of raw iterations stored in the trace.
        """
        lengths = [len(chain[self.varnames[0]]) for chain in self.chains]
        if len(lengths) == 1:
            return lengths[0]
        else:
            return lengths

    def plot(self, burn=0, thin=None, varnames=None,
             kde_kwargs={}, trace_kwargs={}, figure_kwargs={}):
        """
        Make a trace plot paired with a distributional plot.

        Arguments
        -----------
        trace   :   namespace
                    a namespace whose variables are contained in varnames
        burn    :   int
                    the number of iterations to discard from the front of the trace
        thin    :   int
                    the number of iterations to discard between iterations
        varnames :  str or list
                    name or list of names to plot.
        kde_kwargs : dictionary
                     dictionary of aesthetic arguments for the kde plot
        trace_kwargs : dictionary
                       dictinoary of aesthetic arguments for the traceplot

        Returns
        -------
        figure, axis tuple, where axis is (len(varnames), 2)
        """
        f, ax = plot_trace(model=None, trace=self, burn=burn,
                           thin=thin, varnames=varnames,
                      kde_kwargs=kde_kwargs, trace_kwargs=trace_kwargs,
                      figure_kwargs=figure_kwargs)
        return f,ax

    def summarize(self, level=0):
        """
        Compute a summary of the trace. See Also: diagnostics.summary

        Arguments
        ------------
        level   :   int
                    0 for a summary by chain or 1 if the summary should be computed by pooling over chains.
        """
        from .diagnostics import summarize
        return summarize(trace=self, level=level)

    def __getitem__(self, key):
        """
        Getting an item from a trace can be done using at most three indices, where:

        1 index
        --------
            str/list of str: names of variates in all chains to grab. Returns list of Hashmaps
            slice/int: iterations to grab from all chains. Returns list of Hashmaps, sliced to the specification

        2 index
        -------
            (str/list of str, slice/int): first term is name(s) of variates in all chains to grab,
                                          second term specifies the slice each chain.
                                          returns: list of hashmaps with keys of first term and entries sliced by the second term.
            (slice/int, str/list of str): first term specifies which chains to retrieve,
                                          second term is name(s) of variates in those chains
                                          returns: list of hashmaps containing all iterations
            (slice/int, slice/int): first term specifies which chains to retrieve,
                                    second term specifies the slice of each chain.
                                    returns: list of hashmaps with entries sliced by the second term
        3 index
        --------
            (slice/int, str/list of str, slice/int) : first term specifies which chains to retrieve,
                                                      second term is the name(s) of variates in those chains,
                                                      third term is the iteration slicing.
                                                      returns: list of hashmaps keyed on second term, with entries sliced by the third term
        """
        if isinstance(key, str): #user wants only one name from the trace
            if self.n_chains  > 1:
                result = ([chain[key] for chain in self.chains])
            else:
                result = (self.chains[0][key])
        elif isinstance(key, (slice, int)): #user wants all draws past a certain index
            if self.n_chains > 1:
                return [Hashmap(**{k:v[key] for k,v in chain.items()}) for chain in self.chains]
            else:
                return Hashmap(**{k:v[key] for k,v in self.chains[0].items()})
        elif isinstance(key, list) and all([isinstance(val, str) for val in key]): #list of atts over all iters and all chains
                if self.n_chains > 1:
                    return [Hashmap(**{k:chain[k] for k in key}) for chain in self.chains]
                else:
                    return Hashmap(**{k:self.chains[0][k] for k in key})
        elif isinstance(key, tuple): #complex slicing
            if len(key) == 1:
                return self[key[0]] #ignore empty blocks
            if len(key) == 2:
                head, tail = key
                if isinstance(head, str): #all chains, one var, some iters
                    if self.n_chains > 1:
                        result = ([_ifilter(tail, chain[head]) for chain in self.chains])
                    else:
                        result = (_ifilter(tail, self.chains[0][head]))
                elif isinstance(head, list) and all([isinstance(v, str) for v in head]): #all chains, some vars, some iters
                    if self.n_chains > 1:
                        return [Hashmap(**{name:_ifilter(tail, chain[name]) for name in head})
                                   for chain in self.chains]
                    else:
                        chain = self.chains[0]
                        return Hashmap(**{name:_ifilter(tail, chain[name]) for name in head})
                elif isinstance(tail, str):
                    target_chains = _ifilter(head, self.chains)
                    if isinstance(target_chains, Hashmap):
                        target_chains = [target_chains]
                    if len(target_chains) > 1:
                        result = ([chain[tail] for chain in target_chains])
                    elif len(target_chains) == 1:
                        result = (target_chains[0][tail])
                    else:
                        raise IndexError('The supplied chain index {} does not'
                                        ' match any chains in trace.chains'.format(head))
                elif isinstance(tail, list) and all([isinstance(v, str) for v in tail]):
                    target_chains = _ifilter(head, self.chains)
                    if isinstance(target_chains, Hashmap):
                        target_chains = [target_chains]
                    if len(target_chains) > 1:
                        return [Hashmap(**{k:chain[k] for k in tail}) for chain in target_chains]
                    elif len(target_chains) == 1:
                        return Hashmap(**{k:target_chains[0][k] for k in tail})
                    else:
                        raise IndexError('The supplied chain index {} does not'
                                         ' match any chains in trace.chains'.format(head))
                else:
                    target_chains = _ifilter(head, self.chains)
                    if isinstance(target_chains, Hashmap):
                        target_chains = [target_chains]
                    out = [Hashmap(**{k:_ifilter(tail, val) for k,val in chain.items()})
                            for chain in target_chains]
                    if len(out) == 1:
                        return out[0]
                    else:
                        return out
            elif len(key) == 3:
                chidx, varnames, iters = key
                if isinstance(chidx, int):
                    if np.abs(chidx) > self.n_chains:
                        raise IndexError('The supplied chain index {} does not'
                                         ' match any chains in trace.chains'.format(chidx))
                if varnames == slice(None, None, None):
                    varnames = self.varnames
                chains = _ifilter(chidx, self.chains)
                if isinstance(chains, Hashmap):
                    chains = [chains]
                nchains = len(chains)
                if isinstance(varnames, str):
                    varnames = [varnames]
                if varnames is slice(None, None, None):
                    varnames = self.varnames
                if len(varnames) == 1:
                    if nchains > 1:
                        result = ([_ifilter(iters, chain[varnames[0]]) for chain in chains])
                    else:
                        result = (_ifilter(iters, chains[0][varnames[0]]))
                else:
                    if nchains > 1:
                        return [Hashmap(**{varname:_ifilter(iters, chain[varname])
                                        for varname in varnames})
                                for chain in chains]
                    else:
                        return Hashmap(**{varname:_ifilter(iters, chains[0][varname]) for varname in varnames})
        else:
            raise IndexError('index not understood')

        result = np.asarray(result)
        if result.shape == ():
            result = result.tolist()
        elif result.shape in [(1,1), (1,)]:
            result = result[0]
        return result

    ##############
    # Comparison #
    ##############

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            a = [ch1==ch2 for ch1,ch2 in zip(other.chains, self.chains)]
            return all(a)

    def _allclose(self, other, **allclose_kw):
        try:
            self._assert_allclose(other, **allclose_kw)
        except AssertionError:
            return False
        return True

    def _assert_allclose(self, other, **allclose_kw):
        ignore_shape = allclose_kw.pop('ignore_shape', False)
        squeeze = allclose_kw.pop('squeeze', True)
        try:
            assert set(self.varnames) == set(other.varnames)
        except AssertionError:
            raise AssertionError('Variable names are different!\n'
                                 'self: {}\nother:{}'.format(
                                     self.varnames, other.varnames))
        assert isinstance(other, type(self))
        for ch1, ch2 in zip(self.chains, other.chains):
            for k,v in ch1.items():
                allclose_kw['err_msg'] = 'Failed on {}'.format(k)
                if ignore_shape:
                    A = [np.asarray(item).flatten() for item in v]
                    B = [np.asarray(item).flatten() for item in ch2[k]]
                elif squeeze:
                    A = [np.squeeze(item) for item in v]
                    B = [np.squeeze(item) for item in ch2[k]]
                else:
                    A = v
                    B = ch2[k]
                np.testing.assert_allclose(A,B,**allclose_kw)


    ###################
    # IO and Exchange #
    ###################

    def to_df(self):
        """
        Convert the trace object to a Pandas Dataframe.

        Returns
        -------
        a dataframe where each column is a parameter. Multivariate parameters are vectorized and stuffed into a column.
        """
        dfs = []
        outnames = self.varnames
        to_split = [name for name in outnames if np.asarray(self[0,name,0]).size > 1]
        for chain in self.chains:
            out = OrderedDict(list(chain.items()))
            for split in to_split:
                records = np.asarray(copy.deepcopy(chain[split]))
                if len(records.shape) == 1:
                    records = records.reshape(-1,1)
                n,k = records.shape[0:2]
                rest = records.shape[2:]
                if len(rest) == 0:
                    pass
                elif len(rest) == 1:
                    records = records.reshape(n,int(k*rest[0]))
                else:
                    raise Exception("Parameter '{}' has too many dimensions"
                                    " to flatten able to be flattend?"               .format(split))
                records = OrderedDict([(split+'_'+str(i),record.T.tolist())
                                        for i,record in enumerate(records.T)])
                out.update(records)
                del out[split]
            df = pd.DataFrame().from_dict(out)
            dfs.append(df)
        if len(dfs) == 1:
            return dfs[0]
        else:
            return dfs

    def to_csv(self, filename, **pandas_kwargs):
        """
        Write trace out to file, going through Trace.to_df()

        If there are multiple chains in this trace, this will write
        them each out to 'filename_number.csv', where `number` is the
            number of the trace.

        Arguments
        ---------
        filename    :   string
                        name of file to write the trace to.
        pandas_kwargs:  keyword arguments
                        arguments to pass to the pandas to_csv function.
        """
        if 'index' not in pandas_kwargs:
            pandas_kwargs['index'] = False
        dfs = self.to_df()
        if isinstance(dfs, list):
            name, ext = os.path.splitext(filename)
            for i, df in enumerate(dfs):
                df.to_csv(name + '_' + str(i) + ext, **pandas_kwargs)
        else:
            dfs.to_csv(filename, **pandas_kwargs)

    @classmethod
    def from_df(cls, dfs, varnames=None, combine_suffix='_'):
        """
        Convert a dataframe into a trace object.

        Arguments
        ----------
        dfs     :   dataframe or list of dataframes
                    pandas dataframes to convert into a trace. Each dataframe is assumed to be a single chain.
        varnames:   string or list of strings
                    names to use instead of the names in the dataframe. If none, the column
                    names are split using `combine_suffix`, and the unique things before the suffix are used as parameter names.
        """
        if not isinstance(dfs, (tuple, list)):
            dfs = (dfs,)
        if len(dfs) > 1:
            traces = ([cls.from_df(df, varnames=varnames,
                        combine_suffix=combine_suffix) for df in dfs])
            return cls(*[trace.chains[0] for trace in traces])
        else:
            df = dfs[0]
        if varnames is None:
            varnames = df.columns
        unique_stems = set()
        for col in varnames:
            suffix_split = col.split(combine_suffix)
            if suffix_split[0] == col:
                unique_stems.update([col])
            else:
                unique_stems.update(['_'.join(suffix_split[:-1])])
        out = dict()
        for stem in unique_stems:
            cols = []
            for var in df.columns:
                if var == stem:
                    cols.append(var)
                elif '_'.join(var.split('_')[:-1]) == stem:
                    cols.append(var)
            if len(cols) == 1:
                targets = df[cols].values.flatten().tolist()
            else:
                # ensure the tail ordinate sorts the columns, not string order
                # '1','11','2' will corrupt the trace
                order = [int(st.split(combine_suffix)[-1]) for st in cols]
                cols = np.asarray(cols)[np.argsort(order)]
                targets = [vec for vec in df[cols].values]
            out.update({stem:targets})
        return cls(**out)

    @classmethod
    def from_pymc3(cls, pymc3trace):
        """
        Convert a PyMC3 trace to a spvcm trace
        """
        try:
            from pymc3 import trace_to_dataframe
        except ImportError:
            raise ImportError("The 'trace_to_dataframe' function in "
                              "pymc3 is used for this feature. Pymc3 "
                              "failed to import.")
        return cls.from_df(mc.trace_to_dataframe(pymc3trace))

    @classmethod
    def from_csv(cls, filename=None, multi=False,
                      varnames=None, combine_suffix='_', **pandas_kwargs):
        """
        Read a CSV into a trace object, by way of `Trace.from_df()`

        Arguments
        ----------
        filename    :   string
                        string containing the name of the file to read.
        multi       :   bool
                        flag denoting whether the trace being read is a multitrace or not. If so, the filename is understood to be the prefix of many files that end in `filename_#.csv`
        varnames    :   string or list of strings
                        custom names to use for the trace. If not provided, combine suffix is used to identify the unique prefixes in the csvs.
        pandas_kawrgs:  keyword arguments
                        keyword arguments to pass to the pandas functions.
        """
        if multi:
            filepath = os.path.dirname(os.path.abspath(filename))
            filestem = os.path.basename(filename)
            targets = [f for f in os.listdir(filepath)
                         if f.startswith(filestem)]
            ordinates = [int(os.path.splitext(fname)[0].split(combine_suffix)[-1])
                         for fname in targets]
            # preserve the order of the trailing ordinates
            targets = np.asarray(targets)[np.argsort(ordinates)].tolist()
            traces = ([cls.from_csv(filename=os.path.join(filepath, f)
                                    ,multi=False) for f in targets])
            if traces == []:
                raise IOError("No such file or directory: " +
                                        filepath + filestem)

            return cls(*[trace.chains[0] for trace in traces])
        else:
            df = pd.read_csv(filename, **pandas_kwargs)
            return cls.from_df(df, varnames=varnames,
                               combine_suffix=combine_suffix)


####################
# HELPER FUNCTIONS #
####################

def _ifilter(filt,iterable):
    """
    Filter an iterable by whether or not each item is in the filt
    """
    try:
        return iterable[filt]
    except:
        if isinstance(filt, (int, float)):
            filt = [filt]
        return [val for i,val in enumerate(iterable) if i in filt]

def _maybe_hashmap(*collections):
    """
    Attempt to coerce a collection into a Hashmap. Otherwise, leave it alone.
    """
    out = []
    for collection in collections:
        if isinstance(collection, Hashmap):
            out.append(collection)
        else:
            out.append(Hashmap(**collection))
    return out

def _copy_hashmaps(*hashmaps):
    """
    Create deep copies of the hashmaps passed to the function.
    """
    return [Hashmap(**{k:copy.deepcopy(v) for k,v in hashmap.items()})
            for hashmap in hashmaps]
