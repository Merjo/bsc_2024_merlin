import numpy as np

f = 3217.15114
y = np.array([0.73, 0.94, 0.81, 0.77])
t = np.array([0, -123.93244, 65.24117, -637.06388])
c = np.array([0.47, 0.10, 0.13, 0.3])
i_0 = np.array([771.43142, 164.28865, 213.56079, 492.51950])
s_prepare = np.array([0.09496, 0.20757, 0.21605, 0.30787])
lt = np.array([0.00005, 0.00099, 0.00092, 0.00029])

m = (f * y[0] * (1 - lt[0]) * c / c[0]) / ((1 - lt) * f * y)
b = ((((t[0] * (1 - lt[0]) - s_prepare[0]) * c / c[0]) + s_prepare) / ((1 - lt) * f * y)) - (t / (f * y))
b_test_3 = ((((t[0] * (1 - lt[0]) - s_prepare[0]) * c / c[0]) + s_prepare) - (t * (1 - lt))) / (f * y * (1 - lt))
b_test_zaehler = ((((t[0] * (1 - lt[0]) - s_prepare[0]) * c / c[0]) + s_prepare) - (t * (1 - lt)))
b_test_nenner = (f * y * (1 - lt))
b_test_4 = b_test_zaehler / b_test_nenner

x = np.ones_like(y)
x[0] = (1 - np.sum(b)) / np.sum(m)
for i in range(1, 4):
    x[i] = x[0] * m[i] + b[i]

x_2 = (1 - np.sum(b)) / np.sum(m) * m + b

test_x = np.sum(x)
i_t = f * x * y + t
s_c_t = i_t * (1 - lt) - s_prepare
test_c = s_c_t / np.sum(s_c_t)
test_d = test_c - c

a = 0

x = np.ones_like(y) * 0.3

i_t_0 = f * x[0] * y[0] + t[0]
s_c_0 = i_t_0 * (1 - lt[0]) - s_prepare[0]

for i in range(1, 4):
    a = s_c_0 * c[i] / c[0]
    b_t = a + s_prepare[i]
    c_t = b_t / (1 - lt[i])
    d = c_t - t[i]
    x[i] = d / (f * y[i])

i_t = f * x * y + t
s_c_t = i_t * (1 - lt) - s_prepare
test_c = s_c_t / np.sum(s_c_t)
test_d = test_c - c

a = 0

l_2 = np.einsum('g,d->gd', 1 - lt, c)
s_2 = np.einsum('g,d->gd', s_prepare, c)
m_2 = np.einsum('g,gd,d,dg->gd', y, l_2, 1 / y, 1 / l_2)
tl_2 = t * l_2.transpose()
b_2 = (tl_2 - tl_2.transpose() - s_2 + s_2.transpose()) / (f * y.transpose() * l_2.transpose())
b_2_zaehler = ((t * (1 - lt) - s_prepare) * c.transpose() / c + s_prepare.transpose()) - (
        t.transpose() * (1 - lt.transpose()))
b_2_nenner = (f * y * (1 - lt))

b_test_zaehler = ((((t[0] * (1 - lt[0]) - s_prepare[0]) * c / c[0]) + s_prepare) - (t * (1 - lt)))
b_test_nenner = (f * y * (1 - lt))
x = (1 - np.sum(b_2, axis=1)) / np.sum(m_2, axis=1)

test_1_2 = np.sum(x, axis=0)
i_t_2 = f * x * y + t
s_c_t_2 = i_t_2 * (1 - lt) - s_prepare
test_c_2 = s_c_t_2 / np.sum(s_c_t_2)

a = 0
