import numpy as np
n = int(input("Введите размерность матрицы A: "))           #пример при котором работает программа 2 1 0, 1 2 1, 0 1 2
A = np.zeros((n, n))
print("Введите элементы матрицы A построчно:")
for i in range(n):
  row = input().split()
  A[i] = [float(x) for x in row]

print("Введите элементы вектора f:")
f = [float(x) for x in input().split()]
f = np.array(f)

# Метод Гаусса
def gaussian_elimination(A, f):
    n = len(A)
    # Создаем расширенную матрицу
    augmented_matrix = np.concatenate((A, f.reshape(-1, 1)), axis=1)
    # Прямой ход
    for i in range(n):
        # Делим i-ю строку на диагональный элемент
        if augmented_matrix[i, i] == 0:
            # Диагональный элемент равен нулю, система может быть несовместна
            return None
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
        # Вычитаем i-ю строку из остальных строк, умноженную на соответствующий коэффициент
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]
    # Обратная подстановка
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, n]
        for j in range(i + 1, n):
            x[i] -= augmented_matrix[i, j] * x[j]
    return x

# Метод квадратного корня
def metod_kvadrat_koren(A, f):
  n = len(A)
  # Проверка на положительную определенность матрицы
  if np.linalg.det(A) <= 0:
    raise ValueError("Матрица не является положительно определенной.")
  # Создаем нижнюю треугольную матрицу L
  L = np.zeros((n, n))
  for i in range(n):
    for j in range(i):
      if L[j, j] == 0:
        raise ValueError("Диагональный элемент матрицы L равен нулю.")
      L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    # Проверяем, не отрицательно ли значение под корнем
    if A[i, i] - np.sum(L[i, :i]**2) < 0:
      raise ValueError("Значение под корнем отрицательно.")
    L[i, i] = np.sqrt(A[i, i] - np.sum(L[i, :i]**2))
  # Решаем систему Ly = f
  y = np.linalg.solve(L, f)
  # Решаем систему L^T x = y
  x = np.linalg.solve(L.T, y)
  return x

# Решаем систему уравнений обоими методами
try:
  x_gauss = gaussian_elimination(A.copy(), f.copy())
  x_cholesky = metod_kvadrat_koren(A.copy(), f.copy())

  print("Решение методом Гаусса:")
  print(x_gauss)
  print("Решение методом Холецкого:")
  print(x_cholesky)
except ValueError as e:
  print(e)
  x_cholesky = None
