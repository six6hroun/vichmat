import numpy as np
import matplotlib.pyplot as plt

def F(x):
    # Пример функции, которую мы хотим решить
    return x ** 3 - x - 2

def bisection_method(a, b, tol):
    if F(a) * F(b) >= 0:
        print("Функция должна иметь разные знаки на концах отрезка [a, b]")
        return None

    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        if F(midpoint) == 0:
            return midpoint  # Найден точный корень
        elif F(a) * F(midpoint) < 0:
            b = midpoint  # Корень находится в левой половине
        else:
            a = midpoint  # Корень находится в правой половине

    return (a + b) / 2.0  # Возвращаем приближенное значение корня


def secant_method(x0, x1, tol):
    while abs(x1 - x0) > tol:
        if F(x1) - F(x0) == 0:
            print("Деление на ноль!")
            return None

        # Обновляем x0 и x1
        x_temp = x1
        x1 = x1 - F(x1) * (x1 - x0) / (F(x1) - F(x0))
        x0 = x_temp

    return x1  # Возвращаем приближенное значение корня


# Задаем границы для метода бисекции
a = 1
b = 2
tolerance = 1e-1

# Находим корни
root_bisection = bisection_method(a, b, tolerance)
x0 = 1
x1 = 2
root_secant = secant_method(x0, x1, tolerance)

# Создаем график функции
x_values = np.linspace(0, 3, 400)
y_values = F(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='F(x) = x^3 - x - 2', color='blue')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')

# Отмечаем найденные корни
if root_bisection is not None:
    plt.plot(root_bisection, F(root_bisection), 'ro', label=f'Корень (метод деления отрезка пополам): {root_bisection:.5f}')
if root_secant is not None:
    plt.plot(root_secant, F(root_secant), 'go', label=f'Корень (метод секущих): {root_secant:.5f}')

plt.title('График функции и найденные корни')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid()
plt.show()