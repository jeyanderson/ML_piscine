#!/usr/bin/env python3
import numpy
from z_score import zscore
# Example 1:
X = numpy.array([[0], [15], [-9], [7], [12], [3], [-21]])
print(repr(zscore(X)))
# Output:
'''
array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559,
0.17240647, -1.89647119])
'''
# Example 2:
Y = numpy.array([[2], [14], [-13], [5], [12], [4], [-19]])
print(repr(zscore(Y)))
# Output:
'''
array([0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659,
0.28795027, -1.72770165])
'''