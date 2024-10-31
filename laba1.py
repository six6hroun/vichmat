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
print("График исходной функции и интерполяционного полинома для cos(x):")
interpolation(f, n, 'equidistant', 'Интерполяция Лагранжа (равноотстоящие узлы при n = {})'.format(n))
interpolation(f, n, 'chebyshev', 'Интерполяция Лагранжа (узлы Чебышева при n = {})'.format(n))

print("Исследование отклонения для функции 1/(1 + 25x^2):")
otlonenia(f_rational, n, 'equidistant', 'Отклонение ИП от 1/(1 + 25x^2) (равноотстоящие узлы при n = {})'.format(n))
otlonenia(f_rational, n, 'chebyshev', 'Отклонение ИП от 1/(1 + 25x^2) (узлы Чебышева при n = {})'.format(n))

print("Исследование отклонения для функции x:")
otlonenia(f_linear, n, 'equidistant', 'Отклонение ИП от |x| (равноотстоящие узлы при n = {})'.format(n))
otlonenia(f_linear, n, 'chebyshev', 'Отклонение ИП от |x| (узлы Чебышева при n = {})'.format(n))

# Определение функции f(x)
def f(x):
  return np.sin(np.pi * x)

# Определение кубического сплайна
def cubic_spline(x, x_data, y_data):
  n = len(x_data) - 1  # Количество интервалов
  h = np.diff(x_data)  # Шаг сетки

  # Вычисление коэффициентов сплайна
  a = y_data
  b = np.zeros(n)
  c = np.zeros(n)
  d = np.zeros(n)

  # Вычисление коэффициентов b и c
  for i in range(n - 1):
    b[i] = (y_data[i + 1] - y_data[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
  b[n - 1] = (y_data[n] - y_data[n - 1]) / h[n - 1] - h[n - 1] * (2 * c[n - 1] + c[n - 2]) / 3

  # Решение системы линейных уравнений для c
  c[1:n - 1] = (3 * (b[2:n] - b[1:n - 1]) / h[1:n - 1] - 3 * (b[1:n - 1] - b[0:n - 2]) / h[0:n - 2]) / (2 * h[1:n - 1] + 2 * h[0:n - 2])

  # Вычисление коэффициентов d
  for i in range(n-1):
    d[i] = (c[i + 1] - c[i]) / (3 * h[i])

  # Вычисление значений сплайна
  y_spline = np.zeros_like(x)
  for i in range(n):
    idx = np.where((x >= x_data[i]) & (x < x_data[i + 1]))
    y_spline[idx] = a[i] + b[i] * (x[idx] - x_data[i]) + c[i] * (x[idx] - x_data[i]) ** 2 + d[i] * (x[idx] - x_data[i]) ** 3

  return y_spline

# Задание отрезка и количества узлов
interval = [-1, 1]
n_values = [4, 8, 16]

# Построение графиков
plt.figure(figsize=(10,6))

# График исходной функции
x_plot = np.linspace(interval[0], interval[1], 100)
y_plot = f(x_plot)
for i, n in enumerate(n_values):
  # Создание массива узлов интерполяции
  x_data = np.linspace(interval[0], interval[1], n + 1)
  y_data = f(x_data)

  # Вычисление значений сплайна
  y_spline = cubic_spline(x_plot, x_data, y_data)

  # Построение графика сплайна
  plt.plot(x_plot, y_plot, label="f(x)")
  plt.plot(x_plot, y_spline, label="Кубический сплайн (n = {})".format(n))
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("n = {}".format(n))
  plt.legend()
  plt.grid(True)

plt.tight_layout()
plt.show()
