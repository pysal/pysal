import os as _os

def available():
    dcty = _os.path.dirname(_os.path.abspath(__file__))
    files = _os.listdir(dcty)

    unique_names = {_os.path.splitext(f)[0]:[] for f in files 
                    if not _os.path.splitext(f)[0].startswith('__init__')}

    for f in files:
        filename = _os.path.splitext(f)[0]
        if filename not in unique_names:
            continue
        unique_names[filename].append(f)
    
    print('The following examples are stored in the '
          'examples directory\n{}'.format(dcty))
    print(unique_names)
