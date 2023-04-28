import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

def V(x, V0=50, a=1):
    if np.iterable(x):
        return np.array([V(xi, V0, a) for xi in x])
    elif np.abs(x) < a/2:
        return 0
    else:
        return V0
    
def f(x, y, p):
    E=p[0]
    return np.vstack((y[1], -2*(E - V(x))*y[0]))

def bc(ya, yb, p):
    return np.array([ya[0], yb[0], ya[1] - 0.001])

x = np.linspace(-1.5, 1.5, 11)\

y_i = np.zeros((2, x.size))
y_i[0,4] = 0.1

def fk(E, V0=50, a=1, n=1):
    """Returns the error in the equality:
        k2 = k1 * tan(k1 * a / 2) for odd n (even parity solutions); or
       -k2 = k1 * cot(k1 * a / 2) for even n (odd parity solutions),
        where k1 = sqrt(2 * E) and k2 = sqrt(2 * (V0 - E)) for E < V0.
    """
    k1 = np.sqrt(2 * E)
    k2 = np.sqrt(2 * (V0 - E))
    if n % 2:
        return k2 - k1 * np.tan(k1 * a / 2)
    else:
        return k2 + k1 / np.tan(k1 * a / 2)


def Eanalytic(V0=50, a=1, pts=100):
    """Finds the roots of the fk between 0 and V0 for odd and even n."""
    Ei = np.linspace(0.0, V0, pts)
    roots = []
    for n in [1, 2]:
        for i in range(pts - 1):
            soln = root_scalar(fk, args=(V0, a, n), x0=Ei[i], x1=Ei[i + 1])
            if soln.converged and np.around(soln.root, 9) not in roots:
                roots.append(np.around(soln.root, 9))
    return np.sort(roots)

Elist = Eanalytic()

solns = [solve_bvp(f, bc, x, y_i, p=[Ei]) for Ei in Elist]

np.array([soln.p[0] - Ei for soln, Ei in zip(solns, Elist)])

x_plot = np.linspace(x.min(), x.max(), 100)

""" 1D Finite Square Well """
def finsqwell():
    plt.plot(x_plot, V(x_plot), drawstyle='steps-mid', c='k', alpha=0.5)
    for n, soln in enumerate(solns):
        y_plot = soln.sol(x_plot)[0]
        l = plt.plot(x_plot, 4 * y_plot / y_plot.max() + soln.p[0], label = fr"$E_{n}$")
        plt.axhline(soln.p[0], xmin=0.25, xmax=0.75, ls='--', c=l[0].get_color())
    plt.axis(xmin=-1, xmax=1)
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\psi(x)$')
    plt.title("Finite Square Well")
    plt.show()

    plt.plot(x_plot, V(x_plot), drawstyle='steps-mid', c='k', alpha=0.5)
    for n, soln in enumerate(solns):
        y_plot = soln.sol(x_plot)[0]
        l = plt.plot(x_plot, 8 * y_plot**2 / y_plot.max()**2 + soln.p[0], label = fr"$E_{n}$")
        plt.axhline(soln.p[0], xmin=0.25, xmax=0.75, ls='--', c=l[0].get_color())
    plt.axis(xmin=-1, xmax=1)
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\psi(x)$')
    plt.title("Finite Square Well Probability Density")
    plt.show()

""" 1D Infinite Square Well """

def E_inf(n, a=1):
    """n-th energy eigenvalue of the 1D infinite square well potential"""
    return n**2 * np.pi**2 / (2 * a**2)

def psi_inf(x, n, a=1):
    """n-th energy eigenstate of the 1D infinite square well potential"""
    if np.iterable(x):
        return np.array([psi_inf(xi, n, a) for xi in x])
    elif np.abs(x) < a/2:
        phase = n * np.pi * x / a
        return np.cos(phase) if n % 2 else np.sin(phase)
    else:
        return 0
    
def infsqwell():
    plt.plot(x_plot, V(x_plot, V0=100), drawstyle="steps-mid", c="gray")
    for n in np.arange(len(solns)) + 1:
        y_plot = psi_inf(x_plot, n)
        l = plt.plot(x_plot, 4 * y_plot + E_inf(n), label = fr"$E_{n}$")
        col = l[0].get_color()
        plt.axhline(E_inf(n), xmin=0.25, xmax=0.75, ls="--", c=col)
    plt.axis(xmin=-1, xmax=1, ymin=-2, ymax=90)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\psi(x)$")
    plt.title("Infinite Square Well")
    plt.show()

    plt.plot(x_plot, V(x_plot, V0=100), drawstyle="steps-mid", c="gray")
    for n in np.arange(len(solns)) + 1:
        y_plot = psi_inf(x_plot, n)
        l = plt.plot(x_plot, 8 * y_plot ** 2 + E_inf(n), label = fr"$E_{n}$")
        col = l[0].get_color()
        plt.axhline(E_inf(n), xmin=0.25, xmax=0.75, ls="--", c=col)
    plt.axis(xmin=-1, xmax=1, ymin=-2, ymax=90)
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\psi(x)$")
    plt.title("Infinite Square Well Probability Density")
    plt.show()

dx = np.diff(x_plot)[0]
norms = np.array([(soln.sol(x_plot)**2 * dx).sum() for soln in solns])
ci = np.array([5.-0.2j, 3.-0.2j, 0.2+0.7j, 0.01-0.005j])
ci /= (np.abs(ci)**2).sum()
dt = 1. / (2 * np.pi * 50)

def superposition(t, c):
    psi = 0.
    for i, soln in enumerate(solns):
        psi += c[i] * np.exp(-1j*soln.p[0]*t) * soln.sol(x_plot)[0]
    return psi

fig, ax = plt.subplots()

ax.set_xlim((-1.5, 1.5))
ax.set_ylim((0, 0.0012))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$|\psi(x)|^2$')

ax.plot(x_plot, V(x_plot, V0=0.001), drawstyle="steps-mid", c="gray")
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    line.set_data(x_plot, np.abs(superposition(i * dt, ci / norms))**2)
    return (line,)

fps = 25
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=1000./fps, blit=True)
plt.show()

def main():
    return ""

if __name__ == '__main__':
    main()
