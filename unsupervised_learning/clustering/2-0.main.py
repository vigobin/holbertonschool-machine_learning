#!/usr/bin/env python3

import numpy as np
variance = __import__('2-variance').variance

if __name__ == "__main__":
    C = np.random.randn(100, 3)
    print(variance('hello', C))
    print(variance(np.array([1, 2, 3, 4, 5]), C))
    print(variance(np.array([[[1, 2, 3, 4, 5]]]), C))
