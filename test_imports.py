import sys
print("Python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
except ImportError as e:
    print("torch: NOT INSTALLED -", e)
try:
    import numpy
    print("numpy:", numpy.__version__)
except ImportError as e:
    print("numpy: NOT INSTALLED -", e)
try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("matplotlib: NOT INSTALLED -", e)
