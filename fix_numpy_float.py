import numpy as np
if not hasattr(np, 'float'):
    np.float = float

if not hasattr(np, 'isarray'):
    def isarray(x):
        return isinstance(x, np.ndarray)
    np.isarray = isarray