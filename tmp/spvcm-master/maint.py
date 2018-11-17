import os
import warnings

def rebuild_test_data(force=False):
    warnings.warn('Make sure you really want to do this. Pass the force=True option'
                  " if you're sure", stacklevel=2)
    if not force:
        return
    else:
        warnings.warn('Proceeding to overwrite tests...')
    from spvcm.both_levels.tests import make_data as bltests
    from spvcm.upper_level.tests import make_data as ultests
    from spvcm.lower_level.tests import make_data as lltests
    from spvcm.hierarchical.msvc.tests import make_data as msvctests
    from spvcm.hierarchical.svc.tests import make_data as svctests
    from spvcm.tests import make_data as overall
    for module in [bltests, ultests, lltests, overall, msvctests, svctests]:
        module.build()

if __name__ == '__main__':
    warnings.warn('cowardly refusing to wipe out tests. If you want to update test data,'
                  'start up an interpreter, import the maint module, '
                  'and run `rebuild_test_data`', stacklevel=2)
