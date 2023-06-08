import pandas as pd
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

# Excel que contiene los codigos de las estaciones y cordenadas
dfcords = pd.read_excel(r'Coordenadas.xlsx')

# Lista con los codigos de las estaciones que se usaran
lista_estaciones = ['330113', '330160', '330071', '330161', '330076', '330112', '330122', '330121',
                    '330111', '330020', '330019', '330077', '330118', '330162', '330021',
                    '330081', '330193', '320041', '320045', '320056', '320055', '320019', '320063']

# Lista con las horas en el formato que estan en el CSV
lista_hrs = ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00', '06:00:00', '07:00:00',
             '08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00', '14:00:00', '15:00:00',
             '16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00']

# Se genera la matrriz en que se pondran todos los datos de cada hora de las 23 estaciones, donde las columnas son las
# horas y las filas son las estaciones
matrix = np.zeros((23, 24))

# Se genera la lista que contendran las cordenadas
lista_x = []
lista_y = []

# Se genera la lista de las cuales se usaran para la pregunta 5
list_names = ["Llay Llay", "Catemu", "San Felipe"]
lista_cords_x = [-3653.1122207, -3639.512457, -3639.111275]
lista_cords_y = [-7878.440411, -7881.742525, -7855.541478]

# Se genera la funcion que extrae los datos del csv
for i in range(len(lista_estaciones)):
    # Hace que "estacion" sea el data frame de una estacion, va cambiando a otra estacion cada ciclio
    estacion = pd.read_csv('Datos Radiacion/' + lista_estaciones[i] + '_202304_RadiacionGlobal.csv', sep=';')
    # obtienen de el data frame cual es el codigo de la estacion que lo identifica
    id_estacion = estacion.iloc[0, 0]
    # Luego se filtra el exel que contiene las cordenadas para que solo contenga la fila de
    # las coordenadas de la estacion pertinente
    df = dfcords.loc[dfcords['id'] == id_estacion]
    # Luego saca la cordenada x e y de la estacion y las agrega a la lista, eso lo hace para que este cada independte
    # alineado con su dato dependiente
    lista_x.append(df.iloc[0, 3])
    lista_y.append(df.iloc[0, 4])

    for j in range(len(lista_hrs)):
        # Luego se genera otro ciclo en el que filta la fila que contiene la hora necesitada y convierte esto en una
        # data frame
        df2 = estacion.loc[estacion['momento'] == '2023-04-14 ' + lista_hrs[j]]
        # Finalmente extrae el dato de radiacion de cada hora en punto y finalmenta los guarde en orden en la matriz
        # definida anteriormente
        dato = df2.iloc[0, 4]
        matrix[i, j] = dato


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


# Convierte las lista en np.arrays para poder trabajarlos mejor
x = np.array(lista_x)
y = np.array(lista_y)

# Genera una matriz en la que despues se pondran los datos de la radicioon total
matrix_places = np.zeros((3, 24))

for i in range(24):
    z = np.array(matrix[:, i])
    # Asegura que los largos de los datos sean iguales antes de usar la funcion
    if len(x) == len(y) == len(z):
        coeffs, f = polynomial_fit(x, y, z)

    else:
        print("Los arreglos x, y y z deben tener la misma longitud.")

    for j in range(3):
        def g(x, y):
            # Damos vueltas la matriz por como funciona la matriz
            return f(y, x)


        def calcular_volumen(a, b, c, d):
            # a y b son las coordenadas del vértice inferior izquierdo y superior derecho de la región cuadrada

            volume, error = dblquad(g, a, b, c, d)
            return volume


        # Ejemplo de uso
        a = lista_cords_x[j] - 0.5  # Coordenada x del vértice inferior izquierdo de la región cuadrada
        b = lista_cords_x[j] + 0.5  # Coordenada x del vértice superior derecho de la región cuadrada
        c = lista_cords_y[j] - 0.5
        d = lista_cords_y[j] + 0.5
        # Calcula el volumen bajo las superficies, esta es la radiacion total
        volumen = calcular_volumen(a, b, c, d)
        # Genera una matriz en la cual la hora son las columnas y las filas son las estaciones
        matrix_places[j][i] = volumen
        print("La radiacion total a las " + lista_hrs[i] + " en " + list_names[j] + " es del: " + str(volumen))

print("--------------------------------------------------------------------------------------------------------------")
# Suma todas los volumenes sacados anteriormente
for i in range(3):
    suma = 0
    for j in range(24):
        suma += matrix_places[i][j]

    print("La radiacion total del dia 14 en " + list_names[i] + " es igual a: " + str(suma))

print("--------------------------------------------------------------------------------------------------------------")
# Genera un ajuste polinomial de los volumenes de cada hora
lista_hr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
x = np.array(lista_hr)
for i in range(3):
    print("Radicacion total del dia en " + list_names[i] + " es:")
    y = np.array(matrix_places[i, :])

    # Calcular ajustes para diferentes grados
    sols = {}

    z = np.polyfit(x, y, 11, full=True)
    sols[11] = z

    # Pintar datos
    plt.plot(x, y, 'o')
    xp = 0
    # Pintar curvas de ajuste
    xp = np.linspace(0, 23, 10000)
    for grado, sol in sols.items():
        coefs, error, *_ = sol
        # Guarda la aproximacion polinomial de las funciones
        p = np.poly1d(coefs)
        # Esto se uso para sacar la imagen de las aproximacion polinomias mostradas en el documento
        plt.plot(xp, p(xp), "-", label="Function " + list_names[i])
        plt.legend()


    def riemann_sum(poly, a, b, num_intervals):
        """Calcula la suma de Riemann de una función de tipo poly1d en el intervalo [a, b].

        Args:
            poly (numpy.poly1d): Función polinómica.
            a (float): Extremo izquierdo del intervalo.
            b (float): Extremo derecho del intervalo.
            num_intervals (int): Número de intervalos para dividir el intervalo [a, b].

        Returns:
            float: Suma de Riemann de la función en el intervalo [a, b].
        """
        interval_width = (b - a) / num_intervals
        x_values = np.linspace(a, b, num_intervals + 1)
        y_values = poly(x_values)

        # Convierte todos los valores negativos causados por la aproximacion en  0 para mejor la aproximacion
        for i in range(len(y_values)):
            if y_values[i] < 0:
                y_values[i] = 0

        riemann_sum = np.sum(y_values[:-1]) * interval_width
        return riemann_sum


    num_intervals = 1000  # Número de intervalos para la aproximación

    # Llamamos la funcion para aproximar el area bajo la curva
    resultado = riemann_sum(p, 0, 23, num_intervals)
    print(resultado)
