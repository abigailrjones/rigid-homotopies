import matplotlib.pyplot as plt

# test (homogeneous) polynomials
ff = lambda x, y, z: x**2 + y**2 - z**2
g = lambda x, y, z: x**2 + y**2 - 4*z**2

with open('out.txt') as f:
    zeros = [zero.split(' ') for zero in f.read().splitlines()]

vals = {ff: [], g: []}

for j in range(len(zeros)):
    zero = [complex(z) for z in zeros[j]]
    for func in [ff,g]:
        vals[func].append(abs(func(*zero)))

for func,label in zip([ff,g],['f','g']):
    plt.plot(vals[func],label=label)

plt.legend()
plt.ylabel('Magnitude of func(zero)')
plt.xlabel('Number of steps')
plt.show()
# zeros = [complex(z) for z in zeros]

# print(complex(zeros[0].split(' ')[0]))

