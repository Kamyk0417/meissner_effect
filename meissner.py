import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

class Superconductor:
    #parametry nadprzewodnika w kształcie kuli - promień, głębokość wnikania Londonów (jak głęboko może wniknąć pole magn.)
    #krytyczne pole magnetyczne H_c1 (powyżej jego wartości nie ma efektu Meissnera), pole krytyczne H_c2 (dotyczy tylko
    #nadprzewodników II typu, więc domyślnie ma wartość none)
    def __init__(self, radius, lambdaL, H_c1, H_c2=None):
        self.radius = radius
        self.lambdaL = lambdaL
        self.H_c2 = H_c2
        self.H_c1 = H_c1
    
class TypeISc(Superconductor):
    def magn_field(self, R, H_0):

        #rozłożenie równania Londonów (laplasjan(H) = H/lambda^2) na dwa równania pierwszego rzędu
        def london_eq(z, H):
            H1, H2 = H
            dH1_dz = H2
            dH2_dz = H1 / self.lambdaL**2
            return [dH1_dz, dH2_dz]
        
        H_ins = np.zeros_like(R)

        for i,r in enumerate(R):
            r_2 = [self.radius, 0]
            H_init = [H_0, H_0/self.lambdaL]
            sol = solve_ivp(london_eq, r_2, H_init, t_eval=R)
            H_ins = sol.y[0]

        return H_ins


def plot_magn_field(sc, H_init):
    rd = np.linspace(sc.radius, 0, 10)
    H_r = sc.magn_field(rd, H_init)

    alpha = np.linspace(0,np.pi,10, endpoint=False)
    beta = np.linspace(0,2*np.pi,10, endpoint=False)

    alpha,beta = np.meshgrid(alpha,beta)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for i in range(len(rd)):
        x0 = rd[i]*np.sin(alpha)*np.cos(beta)
        y0 = rd[i]*np.sin(alpha)*np.sin(beta)
        z0 = rd[i]*np.cos(alpha)

        #jeżeli pole magnetyczne jest mniejsze od krytycznego pola magnetycznego występuje efekt Meissnera,
        #jeżeli jest większe - przez nadprzewodnik pole przechodzi normalnie (założenie że w kierunku "z")
        if H_init < sc.H_c1:
            H_mag = H_r[i]
            H_x = H_mag*np.sin(alpha)*np.cos(beta)
            H_y = H_mag*np.sin(alpha)*np.sin(beta)
            H_z = H_mag*np.cos(alpha)
        else:
            H_x,H_y, H_z = 0,0, H_init

        ax.quiver(x0,y0,z0, H_x,H_y,H_z, normalize=False, alpha=0.3, length=0.1)
    
    #dodanie sfery reprezentującej nadprzewodnik
    r = sc.radius 
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)

    theta, phi = np.meshgrid(theta, phi)
    x_s = r * np.sin(phi) * np.cos(theta)
    y_s = r * np.sin(phi) * np.sin(theta)
    z_s = r * np.cos(phi)

    ax.plot_surface(x_s, y_s, z_s, color='b', alpha=0.1, edgecolor='r')

    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    ax.axis("off")
    plt.show()


#zadanie parametrów nadprzewodnika i stworzenie takiego obiektu
H_c1 = 51
radius = 5
lambdaL = 0.9
sc1 = TypeISc(radius, lambdaL, H_c1)

#zadanie pola początkowego i narysowanie ostatecznego wykresu
H_0 = 50
plot_magn_field(sc1, H_0)

