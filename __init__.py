"""Here we put subpackages. For example, without running this code you would have to run `from speech2phone import pca`.
After running this code, we can call `import speech2phone; speech2phone.pca.foo()`. This difference is why you can call
`import numpy as np; np.linalg.solve()` just fine, but `import scipy as sp; sp.linalg.solve()` doesn't work. You have to
call `from scipy import linalg`.
"""

# TL;DR: Do this for every folder, along with __init__.py

from . import experiments
from . import preprocessing
from . import temp_jared
from . import temp_kyle
from . import temp_seong
from . import visualizations
