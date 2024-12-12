import numpy as np
import matplotlib.pyplot as plt

# Определение функции f(x, y) и точного решения
def f(x, y):
    return -2*x*y

def exact_solution(x):
    return np.exp(-x**2)

# Ввод данных
x0 = float(input("Введите начальное значение x (x0): "))
x_end = float(input("Введите конечное значение x (x_end): "))
y0 = float(input("Введите начальное значение y (y0): "))
n = int(input("Введите количество шагов (n): "))

# Вычисление шага
h = (x_end - x0) / n

# Метод Эйлера
def euler_method(x0, y0, h, n):
    x = np.linspace(x0, x_end, n+1)
    y = np.zeros_like(x)
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

# Метод Рунге-Кутта 2-го порядка
def runge_kutta2(x0, y0, h, n):
    x = np.linspace(x0, x_end, n+1)
    y = np.zeros_like(x)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h/2 * k1)
        y[i+1] = y[i] + h * k2
    return x, y

# Вычисление приближенных решений
x_euler, y_euler = euler_method(x0, y0, h, n)
x_rk2, y_rk2 = runge_kutta2(x0, y0, h, n)

# Визуализация
plt.figure(figsize=(8, 6))
plt.plot(x_euler, y_euler, label='Метод Эйлера')
plt.plot(x_rk2, y_rk2, label='Метод Рунге-Кутта 2-го порядка')
plt.plot(x_euler, exact_solution(x_euler), label='Точное решение')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Решение задачи Коши')
plt.legend()
plt.grid(True)
plt.show()

# Исследование влияния шага
h_values = [0.1, 0.05, 0.01]
for h in h_values:
    x_euler, y_euler = euler_method(x0, y0, h, int((x_end - x0) / h))
    x_rk2, y_rk2 = runge_kutta2(x0, y0, h, int((x_end - x0) / h))
    plt.figure(figsize=(8, 6))
    plt.plot(x_euler, y_euler, label='Метод Эйлера (h = {})'.format(h))
    plt.plot(x_rk2, y_rk2, label='Метод Рунге-Кутта 2-го порядка (h = {})'.format(h))
    plt.plot(x_euler, exact_solution(x_euler), label='Точное решение')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Решение задачи Коши (h = {})'.format(h))
    plt.legend()
    plt.grid(True)
    plt.show()
