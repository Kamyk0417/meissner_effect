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
    
    #sprawdza czy punkt (x,y,z) znajduje się wewnątrz nadprzewodnika 
    def inside(self, x, y, z):
        return np.sqrt(x**2+y**2+z**2) < self.radius
    
class TypeISc(Superconductor):
    def magn_field(self, X, Y, Z, H_0):
        #składowe pola magnetycznego (zał. pole ma tylko składową z, mnożenie przez 0 służy tylko temu żeby na wykresie 
        #linie pola na zewnątrz nie zasłaniały nadprzewodnika), podział przestrzeni na punkty 
        H_x = np.zeros_like(X)
        H_y = np.zeros_like(Y)
        H_z = 0*H_0*np.ones_like(Z)

        #każdy punkt ma przypisane True lub False zależnie czy jest wewnątrz czy nie
        inside = self.inside(X,Y,Z)
        #każdy punkt wewnątrz ma przypisaną odległość od środka
        dist = np.sqrt(X[inside]**2+Y[inside]**2+Z[inside]**2)
        
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
            #analityczne rozwiązanie równania laplasjan(H) = H/lambda^2 i rozłożenie wyniku na składowe 
            #H_ins = H_0*np.exp(-(self.radius-dist)/self.lambdaL)
            H_x[inside] = H_ins * (X[inside]/dist)
            H_y[inside] = H_ins * (Y[inside]/dist)
            H_z[inside] = H_ins * (Z[inside]/dist)

        return H_x, H_y, H_z

def plot_magn_field(sc, H_0):
    #podział przestrzeni na punkty, podział na "osie" funkcją meshgrid i wywołanie metody magn_field
    x = np.linspace(-7,7,30)
    y = np.linspace(-7,7,30)
    z = np.linspace(-7,7,30)

    x1,y1,z1 = np.meshgrid(x,y,z)
    H_x,H_y,H_z = sc.magn_field(x1,y1,z1, H_0)

    #wykreślenie w 3d wektorów pola
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.quiver(x1,y1,z1, H_x,H_y,H_z, normalize=False, length=0.05, alpha=0.3)
    
    #przejście do współrzędnych sferycznych, dodanie na wykresie sfery o promieniu r reprezentującej nadprzewodnik
    r = 7 
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)


    theta, phi = np.meshgrid(theta, phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    ax.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='r')

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

