#!/usr/bin/env python3
from matrix import Matrix, Vector

sep = '\n' + '-' * 80 + '\n'
m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape, end=sep)
# Output:
# (3, 2)
print(repr(m1.T()), end=sep)
# Output:
# Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.T().shape, end=sep)
# Output:
# (2, 3)
m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.shape, end=sep)
# Output:
# (2, 3)
print(repr(m1.T()), end=sep)
# Output:
# Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.T().shape, end=sep)
# Output:
# (3, 2)
m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0]])
print(repr(m1 * m2), end=sep)
# Output:
# Matrix([[28., 34.], [56., 68.]])
m1 = Matrix([[0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0]])
v1 = Vector([[1], [2], [3]])
print(repr(m1 * v1), end=sep)
# Output:
# Matrix([[8], [16]])
# Or: Vector([[8], [16]])
v2 = Vector([[2], [4], [8]])
print(repr(v1 + v2), end=sep)
# Output:
# Vector([[3],[6],[11]])
m3 = Matrix((3, 2))
m4 = Matrix([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])
print(repr(m3 + m4), end=sep)
print(repr(m3 + 9), end=sep)
print(repr(m3 - m4), end=sep)
print(repr(m3 - 9), end=sep)
print(repr(v1 - v2), end=sep)
print(repr(v1 - 2), end=sep)
print(repr(m4 / 10), end=sep)
print(repr(v2 / 2), end=sep)
print(repr(5 / m3), end=sep)
print(repr(5 / v2), end=sep)
print(repr(10 / m4), end=sep)
print(repr(m4 / m3), end=sep)
print(repr(v1 / v2), end=sep)
print(repr(m4 / 0), end=sep)
print(repr(m4 * 2.5), end=sep)
print(repr(2.5 * m4), end=sep)
print(repr(v1 * 2.5), end=sep)
m5 = Matrix([[9, 8, 7]])
m6 = Matrix([[0.1], [0.2], [0.3]])
print(repr(m5 * m6), end=sep)
print(repr(m6 * m5), end=sep)
print(repr(v1 * m5), end=sep)
v3 = Vector([[6., 5, 4]])
print(repr(v1 * v3), end=sep)
print(repr(v3 * v1), end=sep)
print(v2.dot(v1), end=sep)
print(v3.dot(v3), end=sep)
print(repr(v3.T()), end=sep)
print(repr(v2.T()), end=sep)