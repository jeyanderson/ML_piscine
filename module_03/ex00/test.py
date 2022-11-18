#!/usr/bin/env python3
import numpy as np
from sigmoid import sigmoid_
import matplotlib.pyplot as plt

# Example 1:
x = np.array(-4)
print(repr(sigmoid_(x)))
# Output:
# array([[0.01798620996209156]])

# Example 2:
x = np.array(2)
print(repr(sigmoid_(x)))
# Output:
# array([[0.8807970779778823]])

# Example 3:
x = np.array([[-4], [2], [0]])
print(repr(sigmoid_(x)))
# Output:
# array([[0.01798620996209156], [0.8807970779778823], [0.5]])

continuous_x=np.linspace(-20,20,100)
plt.plot(continuous_x,sigmoid_(continuous_x))
plt.show()