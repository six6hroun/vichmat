import numpy as np
import matplotlib.pyplot as plt


def divided_differences(xs, ys):
    dd_table = []
    dd_table.append(list(ys))

    for i in range(1, len(xs)):
        row = []
        for j in range(len(xs) - i):
            row.append((dd_table[-1][j + 1] - dd_table[-1][j]) / (xs[j + i] - xs[j]))
        dd_table.append(row)

    return dd_table


def newtons_polynomial(dd_table, xs, x):
    result = dd_table[0][0]
    term = 1
    for i in range(1, len(xs)):
        term *= (x - xs[i - 1])
        result += dd_table[i][0] * term
    return result


def interpolate_newton(xs, ys, x_values):
    dd_table = divided_differences(xs, ys)
    interpolated_ys = [newtons_polynomial(dd_table, xs, x) for x in x_values]

    return interpolated_ys


xs = [0, 1, 2, 3]
ys = [1, 4, 9, 16]
x_values = np.linspace(min(xs), max(xs), 30)
interpolated_ys = interpolate_newton(xs, ys, x_values)

for x, y in zip(x_values, interpolated_ys):
    print(f'При x = {x:.2f}, интерполированное значение y = {y:.2f}')

# Построение графика
plt.figure(figsize=(8, 6))
plt.scatter(xs, ys, color="red", label="Исходные точки")
plt.plot(x_values, interpolated_ys, color="blue", label="Интерполяционный полином")
plt.title("Интерполяционный полином Ньютона")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()