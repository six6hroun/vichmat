import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(x)

def f_rational(x):
    return 1 / (1 + 25 * x**2)

def f_linear(x):
    return x

# Интерполяционный полином Лагранжа
def lagrange(x, y, x_interp):
    n = len(x)
    y_interp = np.zeros_like(x_interp)
    for i in range(n):
        l_i = np.ones_like(x_interp)
        for j in range(n):
            if i != j:
                l_i *= (x_interp - x[j]) / (x[i] - x[j])
        y_interp += y[i] * l_i
    return y_interp

# Равноотстоящие узлы
def ravnostoyashchikh(n):
    return np.linspace(-1, 1, n)

# Узлы Чебешева
def chebyshev(n):
    return np.cos(np.pi * (2 * np.arange(n) + 1) / (2 * n))

# Определение графика
def interpolation(f, n, node_type, title):
    x = np.linspace(-1, 1, 100)
    y = f(x)

    if node_type == 'equidistant':
        nodes = ravnostoyashchikh(n)
    elif node_type == 'chebyshev':
        nodes = chebyshev(n)
    else:
        raise ValueError("Неверный тип узлов")

    y_interp = lagrange(nodes, f(nodes), x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Исходная функция', color='blue')
    plt.plot(x, y_interp, label='Интерполяционный полином', color='red')
    plt.scatter(nodes, f(nodes), color='green', label='Узлы')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Исследование отклонений
def otlonenia(f, n, node_type, title):
    x = np.linspace(-1, 1, 100)
    y = f(x)

    if node_type == 'equidistant':
        nodes = ravnostoyashchikh(n)
    elif node_type == 'chebyshev':
        nodes = chebyshev(n)
    else:
        raise ValueError("Неверный тип узлов")

    y_interp = lagrange(nodes, f(nodes), x)

    deviation = np.abs(y - y_interp)
    plt.figure(figsize=(8, 6))
    plt.plot(x, deviation, label='Отклонение', color='red')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Отклонение')
    plt.legend()
    plt.grid(True)
    plt.show()

n = int(input("Введите количество узлов (n): "))
print("График исходной функции и интерполяционного полинома для sin(x):")
interpolation(f, n, 'equidistant', 'Интерполяция Лагранжа (равноотстоящие узлы при n = {})'.format(n))
interpolation(f, n, 'chebyshev', 'Интерполяция Лагранжа (узлы Чебышева при n = {})'.format(n))

print("Исследование отклонения для функции 1/(1 + 25x^2):")
otlonenia(f_rational, n, 'equidistant', 'Отклонение ИП от 1/(1 + 25x^2) (равноотстоящие узлы при n = {})'.format(n))
otlonenia(f_rational, n, 'chebyshev', 'Отклонение ИП от 1/(1 + 25x^2) (узлы Чебышева при n = {})'.format(n))

print("Исследование отклонения для функции x:")
otlonenia(f_linear, n, 'equidistant', 'Отклонение ИП от |x| (равноотстоящие узлы при n = {})'.format(n))
otlonenia(f_linear, n, 'chebyshev', 'Отклонение ИП от |x| (узлы Чебышева при n = {})'.format(n))
