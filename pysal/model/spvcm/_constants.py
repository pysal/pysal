import sys
import types
import os
RTOL = 1e-3
ATOL = 1e-3
TEST_SEED = 310516

PY2 = sys.version_info[0] == 2

if PY2:
    CLASSTYPES = (types.ClassType, type)
else:
    CLASSTYPES = (type,)

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
