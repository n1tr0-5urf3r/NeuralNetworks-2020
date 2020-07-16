import numpy as np

x = np.array((-1.25, 0, 1.25, 2.75))
y = np.array((12.02, 2.25, 4.98, 8.27))
w = np.array((-2, -0.25, 1.5, 3))
h = np.zeros((x.shape[0], w.shape[0]))

for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        h[i][j] = np.exp(-((x[i]-w[j])**2)/2)

pseudoinverse = np.linalg.pinv(h)
c = np.dot(pseudoinverse, y)

sum = 0
for i in range(c.shape[0]):
    sum += c[i]*np.exp(-((4-w[i])**2)/2)

def px(x): return (x-1.5)**2 * (1-x) + 4*x

print(sum)
print(px(4))
