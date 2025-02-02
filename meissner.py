import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

class Superconductor:
    #parametry nadprzewodnika w kształcie kuli - promień, głębokość wnikania Londonów (jak głęboko może wniknąć pole magn.)
    #krytyczne pole magnetyczne H_c1 (powyżej jego wartości nie ma efektu Meissnera), pole krytyczne H_c2 (dotyczy tylko
    #nadprzewodników II typu, więc domyślnie ma wartość none)
    def __init__(self, radius, lambdaL, H_c1, H_c2=None):
        self.radius = radius
        self.lambdaL = lambdaL
        self.H_c2 = H_c2
        self.H_c1 = H_c1
    
    #sprawdza czy punkt znajduje się wewnątrz nadprzewodnika 
    def inside(self, r):
        return r < self.radius
    
class TypeISc(Superconductor):
    def magn_field(self, R, Theta, Phi, H_0):
        #składowe pola magnetycznego, podział przestrzeni na punkty 
        H_r = np.zeros_like(R)
        H_theta = np.zeros_like(Theta)
        H_phi = np.zeros_like(Phi)

        #każdy punkt ma przypisane True lub False zależnie czy jest wewnątrz czy nie
        inside = self.inside(R)
        #każdy punkt wewnątrz ma przypisaną odległość od środka
        dist = R[inside]
        
        #rozłożenie równania Londonów (laplasjan(H) = H/lambda^2) na dwa równania pierwszego rzędu
        def london_eq(H, z):
            H1, H2 = H
            dH1_dz = H2
            dH2_dz = H1 / self.lambdaL**2
            return [dH1_dz, dH2_dz]
        
        H_ins = np.zeros_like(dist)

        #przypisuje i-temu indeksowi (i-towej odległości) tę odległość, jeżeli jest mniejsza od promienia to oblicza
        #wartość pola w tym punkcie przy pomocy funkcji odeint oraz przypisuje tę wartość dla i-tego miejsca, tym razem 
        #w tablicy wartości pól a nie odległości
        for i, r in enumerate(dist):
            if r <= self.radius:
                r_2 = [self.radius, r]
                H_init = [H_0, H_0/self.lambdaL]
                sol = odeint(london_eq, H_init, r_2)
                H_ins[i] = sol[-1, 0]

        if H_0 < self.H_c1:
            H_r[inside] = H_ins

        return H_r,H_theta,H_phi

def plot_magn_field(sc, H_0):
    #podział przestrzeni na punkty, podział na "osie" funkcją meshgrid i wywołanie metody magn_field
    rd = np.linspace(0, sc.radius+2, 30)
    alfa = np.linspace(0,2*np.pi,30)
    beta = np.linspace(0,np.pi,30)

    rd,alfa,beta = np.meshgrid(rd,alfa,beta)
    H_r,H_theta,H_phi = sc.magn_field(rd,alfa,beta, H_0)

    #przejście do współrzędnych kartezjańskich
    X = rd * np.sin(alfa) * np.cos(beta)
    Y = rd * np.sin(alfa) * np.sin(beta)
    Z = rd * np.cos(alfa)

    H_x = H_r * np.sin(alfa) * np.cos(beta)
    H_y = H_r * np.sin(alfa) * np.sin(beta)
    H_z = H_r * np.cos(alfa)

    #wykreślenie w 3d wektorów pola
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.quiver(X,Y,Z, H_x,H_y,H_z, normalize=False, length=0.05, alpha=0.3)
    
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
H_c1 = 100
radius = 5
lambdaL = 0.2
sc1 = TypeISc(radius, lambdaL, H_c1)

#zadanie pola początkowego i narysowanie ostatecznego wykresu
H_0 = 50
plot_magn_field(sc1, H_0)
