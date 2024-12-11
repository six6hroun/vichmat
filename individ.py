import numpy as np

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

'''def divided_differences(x, y):
  n = len(y)
  coef = np.zeros((n, n))
  coef[:, 0] = y  # Первый столбец - значения функции

  for j in range(1, n):
    for i in range(n - j):
      coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

  return coef[0]  # Возвращаем только первую строку с коэффициентами


def newton_interpolation(x, y, x_interp):
  n = len(x)
  coef = divided_differences(x, y)

  # Вычисляем значение полинома в точке x_interp
  result = coef[0]
  product_term = 1.0

  for i in range(1, n):
    product_term *= (x_interp - x[i - 1])
    result += coef[i] * product_term

  return result


# Данные
x_points = np.array([1, 2, 3, 4])
y_points = np.array([1, 2, 5, 7])

# Точка для интерполяции
x_interp = 1.5
result = newton_interpolation(x_points, y_points, x_interp)
print(f"Значение интерполяционного полинома в точке {x_interp} равно {result}")'''