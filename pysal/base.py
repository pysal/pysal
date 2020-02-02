"""
Base information for pysal meta package
"""


federation_hierarchy = {
    'explore': ['esda', 'giddy', 'segregation',
                'pointpats', 'inequality',
                'segregation', 'spaghetti'],
    'model': ['spreg', 'spglm', 'tobler', 'spint',
              'mgwr', 'spvcm'],
    'viz': ['splot', 'mapclassify'],
    'lib': ['libpysal']
}

memberships = {'libpysal': 'lib',
               'esda': 'explore',
               'giddy': 'explore',
               'segregation': 'explore',
               'pointpats': 'explore',
               'inequality': 'explore',
               'spaghetti': 'explore',
               'spreg': 'model',
               'spglm': 'model',
               'spint': 'model',
               'tobler': 'model',
               'spvcm': 'model',
               'splot': 'viz',
               'mapclassify': 'viz'
}
