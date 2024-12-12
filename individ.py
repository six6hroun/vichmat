import numpy as np
import matplotlib.pyplot as plt

'''def gauss_jordan_inverse(A):
  # Проверка определителя
  if np.linalg.det(A) == 0:
    return None

  # создаем расширенную мтарицу
  I = np.eye(A.shape[0])
  augmented_matrix = np.concatenate((A, I), axis=1)

  # Жордан-Гаусс
  for i in range(A.shape[0]):
    pivot = augmented_matrix[i, i]
    augmented_matrix[i, :] /= pivot
    for j in range(A.shape[0]):
      if i != j:
        factor = augmented_matrix[j, i]
        augmented_matrix[j, :] -= factor * augmented_matrix[i, :]

  # Обратная матрица из расширенной
  inverse_matrix = augmented_matrix[:, A.shape[0]:]

  return inverse_matrix

# Матрица
A = np.array([[1, 2], [3, 5]])
inverse_matrix = gauss_jordan_inverse(A)
print(inverse_matrix)'''

'''def newtons_polynomial(dd_table, xs, x):
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


xs = [1, 2, 3, 4]
ys = [1, 2, 5, 7]
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
plt.show()'''
