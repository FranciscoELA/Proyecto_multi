import sympy as sp

# Se genera una lista con los coeficientes que sacamos del codigo en "Funcion Temperatura"
coeffs = [4.60001333 * 10 ** -6, -7.89735339 * 10 ** -7, -1.14688791 * 10 ** -2,
          -2.26658270 * 10 ** -3, 4.46432541 * 10 ** -6, -7.90077210 * 10 ** -6,
          1.15854967 * 10 ** -2, 9.80575544 * 10 ** -1, 2.08769183,
          -7.93957076 * 10 ** -4]


# Pregunta 2 c
def directional_derivative(f, x, y, a, b, u, v):
    # Definir las variables simbólicas
    x, y = sp.symbols('x y')

    # Calcular la derivada parcial con respecto a x
    df_dx = sp.diff(f, x)

    # Calcular la derivada parcial con respecto a y
    df_dy = sp.diff(f, y)

    # Definir el vector director
    vector_director = sp.Matrix([u, v])

    # Normalizar el vector director
    norm = vector_director.norm()
    normalized_vector = vector_director / norm

    # Calcular el punto de evaluacion
    point = sp.Matrix([a, b])

    # Calcular la derivada direccional
    d_dir = normalized_vector.dot(sp.Matrix([df_dx, df_dy])).subs([(x, a), (y, b)])

    return d_dir


# Funcion
x, y = sp.symbols('x y')
f = coeffs[0] * x ** 3 + coeffs[1] * y ** 3 + coeffs[2] * x ** 2 + coeffs[3] * y ** 2 + coeffs[4] * x * y ** 2 \
    + coeffs[5] * x ** 2 * y + coeffs[6] * x * y + coeffs[7] * x + coeffs[8] * y + coeffs[9]

# Hacia el Norte
a = -3717.189911
b = -7859.954593
u = 0
v = 1
result = directional_derivative(f, x, y, a, b, u, v)
print(f"La derivada direccional de f en el punto ({a}, {b}) con el vector director [{u}, {v}] es: {result}")

# Hacia el Este
a = -3717.189911
b = -7859.954593
u = 1
v = 0
result = directional_derivative(f, x, y, a, b, u, v)
print(f"La derivada direccional de f en el punto ({a}, {b}) con el vector director [{u}, {v}] es: {result}")

# Pregunta 2d
# Hacia vina de Rodelillo
a = -3673.552608
b = -7949.914929
u = (3673.552608 - 3660.682784)
v = (7949.914929 - 7940.995821)
result = directional_derivative(f, x, y, a, b, u, v)
print(f"La derivada direccional de f en el punto ({a}, {b}) con el vector director [{u}, {v}] es: {result}")

# Pregunta 2e

import sympy as sp


def calcular_gradiente(func, variables, punto):
    # Definir las variables simbólicas
    x, y = variables

    # Calcular el gradiente
    gradiente = [sp.diff(func, var) for var in variables]

    # Evaluar el gradiente en el punto especificado
    gradiente_evaluado = [deriv.subs([(x, punto[0]), (y, punto[1])]) for deriv in gradiente]

    return gradiente_evaluado


# Ejemplo de una función de dos variables


# Punto en el cual calcular el gradiente
punto = (-3740.767775, -7871.373228)

# Calcular el gradiente
gradiente = calcular_gradiente(f, (x, y), punto)

print("Gradiente:", gradiente)

print("--------------------------------------------------------------------------------------------------------------")


# Pregunta 2f

def optimize_function_2d(f, x_range, y_range, step_size=0.1, epsilon=1e-6):
    x_start, x_end = x_range
    y_start, y_end = y_range

    def partial_derivative_x(f, x, y, h):
        return (f(x + h, y) - f(x - h, y)) / (2 * h)

    def partial_derivative_y(f, x, y, h):
        return (f(x, y + h) - f(x, y - h)) / (2 * h)

    def hessian_matrix(f, x, y, h):
        # Genera una matriz hessiana
        f_xx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h ** 2
        f_yy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h ** 2
        f_xy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
        return [[f_xx, f_xy], [f_xy, f_yy]]

    def classify_point(f, x, y, h):
        # Clasifica los puntos como maxiomos o minimos usando el metodo  de la matriz hessiana
        hessian = hessian_matrix(f, x, y, h)
        determinant = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0]
        trace = hessian[0][0] + hessian[1][1]
        if determinant > 0 and trace > 0:
            return "Minimum"
        elif determinant > 0 and trace < 0:
            return "Maximum"
        else:
            return "Saddle Point"

    def gradient_descent(f, x_range, y_range, step_size, epsilon):
        x_min, x_max = x_range
        y_min, y_max = y_range

        x_best = None
        y_best = None
        f_best = float('inf')

        x_current = x_min
        while x_current <= x_max:
            y_current = y_min
            while y_current <= y_max:
                f_current = f(x_current, y_current)

                if f_current < f_best:
                    x_best = x_current
                    y_best = y_current
                    f_best = f_current

                y_current += step_size

            x_current += step_size

        return x_best, y_best

    def optimize():
        x_min, x_max = x_range
        y_min, y_max = y_range
        h = 1e-6

        # Optimization within the rectangular region
        x_opt, y_opt = gradient_descent(f, x_range, y_range, step_size, epsilon)
        f_opt = f(x_opt, y_opt)
        points = [(x_opt, y_opt, f_opt, "Global Minimum")]

        # Optimization along the boundary
        for x in [x_min, x_max]:
            for y in [y_min, y_max]:
                if x != x_opt or y != y_opt:
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        point_type = classify_point(f, x, y, h)
                        if point_type == "Minimum":
                            points.append((x, y, f(x, y), "Local Minimum"))
                        elif point_type == "Maximum":
                            points.append((x, y, f(x, y), "Local Maximum"))

        # Maximum global (in rectangular region)
        x_max, y_max = gradient_descent(lambda x, y: -f(x, y), x_range, y_range, step_size, epsilon)
        f_max = f(x_max, y_max)
        points.append((x_max, y_max, f_max, "Global Maximum"))

        return points

    return optimize()


# Ejemplo de uso:
def my_function(x, y):
    return (coeffs[0] * x ** 3 + coeffs[1] * y ** 3 + coeffs[2] * x ** 2 +
            coeffs[3] * y ** 2 + coeffs[4] * x * y ** 2 + coeffs[5] * x ** 2 * y +
            coeffs[6] * x * y + coeffs[7] * x + coeffs[8] * y + coeffs[9])


x_range = (-3750.118728, -3617.570429)
y_range = (-7956.240963, -7809.620071)

optimal_points = optimize_function_2d(my_function, x_range, y_range)
if optimal_points is not None:
    for point in optimal_points:
        x, y, f_val, point_type = point
        print(f"Point: ({x}, {y}), Value: {f_val}, Type: {point_type}")
else:
    print("No se encontraron puntos óptimos.")

print("--------------------------------------------------------------------------------------------------------------")
# Pregunta 2 g

import math

def optimize_function_2d(f, radius, center, step_size=0.1, epsilon=1e-6):
    r = radius
    cx, cy = center

    def gradient_descent(f, radius, center, step_size, epsilon):
        r = radius
        cx, cy = center

        x_min = cx - r
        x_max = cx + r
        y_min = cy - r
        y_max = cy + r

        x_best = None
        y_best = None
        f_best = float('inf')

        x_current = x_min
        while x_current <= x_max:
            y_current = y_min
            while y_current <= y_max:
                distance = math.sqrt((x_current - cx) ** 2 + (y_current - cy) ** 2)
                if distance <= r:
                    f_current = f(x_current, y_current)

                    if f_current < f_best:
                        x_best = x_current
                        y_best = y_current
                        f_best = f_current

                y_current += step_size

            x_current += step_size

        return x_best, y_best

    def optimize():
        r = radius
        cx, cy = center

        # Optimization within the circular region
        x_opt, y_opt = gradient_descent(f, r, (cx, cy), step_size, epsilon)
        f_opt = f(x_opt, y_opt)
        points = [(x_opt, y_opt, f_opt, "Global Minimum")]

        # Maximum global (in circular region)
        x_max, y_max = gradient_descent(lambda x, y: -f(x, y), r, (cx, cy), step_size, epsilon)
        f_max = f(x_max, y_max)
        points.append((x_max, y_max, f_max, "Global Maximum"))

        return points

    return optimize()


# Ejemplo de uso:
def my_function(x, y):
    return (coeffs[0] * x ** 3 + coeffs[1] * y ** 3 + coeffs[2] * x ** 2 +
            coeffs[3] * y ** 2 + coeffs[4] * x * y ** 2 + coeffs[5] * x ** 2 * y +
            coeffs[6] * x * y + coeffs[7] * x + coeffs[8] * y + coeffs[9])


center = (-3643.523717212497, -7868.577510955656)
radius = 13.76255076232253
step_size = 0.1

optimal_points = optimize_function_2d(my_function, radius, center, step_size)
if optimal_points is not None:
    for point in optimal_points:
        x, y, f_val, point_type = point
        print(f"Point: ({x}, {y}), Value: {f_val}, Type: {point_type}")
else:
    print("No se encontraron puntos óptimos.")



