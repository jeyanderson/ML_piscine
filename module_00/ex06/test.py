

#!/usr/bin/env python3
import numpy as np
from prediction import predict_
from loss import loss_, loss_elem_
x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
# Example 1:
print(repr(loss_elem_(y1, y_hat1)))
# Output:
# array([[0.], [1], [4], [9], [16]])
# Example 2:
print(loss_(y1, y_hat1))
# Output:
# 3.0