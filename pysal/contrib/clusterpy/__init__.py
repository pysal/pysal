try:
    import clusterpy
except ImportError:
    print 'pysal.contrib.clusterpy requires clusterpy'
    print 'clusterpy not found.'
else:
    from clusterpy_ext import Layer
    from clusterpy_ext import loadArcData
    from clusterpy_ext import importCsvData
    from clusterpy_ext import addRook2Layer
    from clusterpy_ext import addQueen2Layer
    from clusterpy_ext import addArray2Layer
    from clusterpy_ext import addW2Layer
