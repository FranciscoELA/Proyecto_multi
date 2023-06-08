import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import contour
from numpy import meshgrid, linspace
from sympy.solvers import solve
from sympy import Symbol

# Excel que contiene los codigos de las estaciones y cordenadas
dfcords = pd.read_excel(r'Coordenadas.xlsx')

# Lista con las horas en el formato que estan en el CSV
lista_estaciones = ['330030', '330113', '330160', '330075', '330071', '330161', '330076', '330112', '330122', '330121',
                    '330111', '330114', '330020', '330019', '330021', '330077', '330118', '330162', '330163',
                    '330081', '330193', '330007', '330006', '320041', '320045', '320056', '320055', '320019', '320063']

# Hacemos las listas en las cuales se ordenaran los datos
lista_temp = []
lista_x = []
lista_y = []

for i in range(len(lista_estaciones)):
    # Se genera un ciclio que recorre todos los CSV
    estacion = pd.read_csv('Datos CSV/' + lista_estaciones[i] + '_202304_Temperatura.csv', sep=';')
    # Saca el ID de la estacion
    id_estacion = estacion.iloc[0, 0]
    # Con el ID sacado se filtra el excel para que solo tenga los datos de la estacion correspondiente
    df = dfcords.loc[dfcords['id'] == id_estacion]
    # Luego agrega las cordenadas correspondientas a cada estacion de esta manera los pares de datos estan ordenados
    lista_x.append(df.iloc[0, 3])
    lista_y.append(df.iloc[0, 4])
    # Luego filtamos por la hora y dia que se desean, asi nos queda solo una fila
    df2 = estacion.loc[estacion['momento'] == '2023-04-14 12:00:00']
    # Finalmente sacamos el dato de ts que se requiere
    dato = df2.iloc[0, 4]
    # Se agrega el dato a la lista
    lista_temp.append(dato)


def polynomial_fit(x, y, z):
    # Construye la matriz de diseño A
    A = np.column_stack((x ** 3, y ** 3, x ** 2, y ** 2, x * y ** 2, x ** 2 * y, x * y, x, y, np.ones_like(x)))

    # Calcula los coeficientes del ajuste utilizando la descomposición QR
    Q, R = np.linalg.qr(A)
    coeffs = np.linalg.lstsq(R, np.dot(Q.T, z), rcond=None)[0]

    # Define la función de mejor ajuste
    def f(x, y):
        return (coeffs[0] * x ** 3 + coeffs[1] * y ** 3 + coeffs[2] * x ** 2 +
                coeffs[3] * y ** 2 + coeffs[4] * x * y ** 2 + coeffs[5] * x ** 2 * y +
                coeffs[6] * x * y + coeffs[7] * x + coeffs[8] * y + coeffs[9])

    return coeffs, f

# Re asignamos el nombre de las listas y las convertimos en arrays
x = np.array(lista_x)
y = np.array(lista_y)
z = np.array(lista_temp)

# Antes de aplicar el ajuste polinomias de minimos cuadrados se asegura que los datos sean del mismo porte
if len(x) == len(y) == len(z):
    coeffs, f = polynomial_fit(x, y, z)
    print("Coeficientes del ajuste:", coeffs)
    print("Función de mejor ajuste:")
    print("T(x, y) =", coeffs[0], "x^3 +", coeffs[1], "y^3 +", coeffs[2], "x^2 +",
          coeffs[3], "y^2 +", coeffs[4], "xy^2 +", coeffs[5], "x^2y +",
          coeffs[6], "xy +", coeffs[7], "x +", coeffs[8], "y +", coeffs[9])
else:
    print("Los arreglos x, y y z deben tener la misma longitud.")

# Preguntas

# Con la funcion sacamos el valor de cada uno de los lugares
Casa_b = f(-3701.48537, -7934.39537)

Milli_p = f(-3735.77083, -7918.21921)

San_b = f(-3731.10463, -7854.77)

Quill = f(-3651.49037, -7915.875)

# vectores para x y y
x = linspace(-3750.118728, -3617.570429, 1000)
y = linspace(-7956.240963, -7809.620071, 1000)
X, Y = meshgrid(x, y)  # construccion de malla

# Luego generamos el grafico de las curvas de nivel a estas altura y le ponemos la leyendas
plot = contour(X, Y, f(X, Y), [San_b, Milli_p, Casa_b, Quill], cmap='Dark2_r')
h1, l1 = plot.legend_elements()
h2, l2 = plot.legend_elements()
h3, l3 = plot.legend_elements()
h4, l4 = plot.legend_elements()
plt.legend([h1[0], h2[1],h3[2], h4[3]], ['San Bernardo', 'Melipilla','Casa Blanca', 'Quillota'])
# Guarda la imagen del plot
plt.savefig('plots')


# Pregunta 2b

# Usamos este codigo para obtener otra ubicacion en que se tenga la misma temperatura
y = Symbol('y')
x = -3680
sol = solve(coeffs[0] * x ** 3 + coeffs[1] * y ** 3 + coeffs[2] * x ** 2 +
            coeffs[3] * y ** 2 + coeffs[4] * x * y ** 2 + coeffs[5] * x ** 2 * y +
            coeffs[6] * x * y + coeffs[7] * x + coeffs[8] * y + coeffs[9] - Quill, y)
print(sol)
