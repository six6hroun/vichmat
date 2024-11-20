import numpy as np
import matplotlib.pyplot as plt

# Метод Якоби
def jacobi(A, b, x0, tol=1e-10, max_iterations=100):
    n = len(b)
    x = x0.copy()
    norm_residuals = []
    for _ in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum_ = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_) / A[i][i]

        # Норма невязки
        residual = b - A @ x_new
        norm_residuals.append(np.linalg.norm(residual))

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, norm_residuals


# Метод Зейделя
def gauss_seidel(A, b, x0, tol=1e-10, max_iterations=100):
    n = len(b)
    x = x0.copy()
    norm_residuals = []
    for _ in range(max_iterations):
        for i in range(n):
            sum_ = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_) / A[i][i]

        # Норма невязки
        residual = b - A @ x
        norm_residuals.append(np.linalg.norm(residual))

        if np.linalg.norm(residual) < tol:
            break

    return x, norm_residuals

# Функция для получения входных данных от пользователя
def get_input():
    n = int(input("Введите размерность системы (количество уравнений): "))

    print("Введите элементы матрицы A построчно (через пробел):")
    A = np.array([list(map(float, input().split())) for _ in range(n)])

    print("Введите элементы вектора b (через пробел):")
    b = np.array(list(map(float, input().split())))

    print("Введите начальное приближение (через пробел):")
    x0 = np.array(list(map(float, input().split())))

    tol = float(input("Введите допустимую погрешность: "))
    max_iterations = int(input("Введите максимальное количество итераций: "))

    return A, b, x0, tol, max_iterations

# Получаем входные данные от пользователя
A, b, x0, tol, max_iterations = get_input()

# Визуализация результатов
initial_guesses = [x0]

for initial_guess in initial_guesses:
    # Метод Якоби
    solution_jacobi, residuals_jacobi = jacobi(A, b, initial_guess, tol, max_iterations)
    # Метод Зейделя
    solution_seidel, residuals_seidel = gauss_seidel(A, b, initial_guess, tol, max_iterations)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(residuals_jacobi, label='Метод Якоби')
    plt.yscale('log')
    plt.title('Норма невязки (Метод Якоби)')
    plt.xlabel('Итерация')
    plt.ylabel('Норма невязки')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(residuals_seidel, label='Метод Зейделя', color='orange')
    plt.yscale('log')
    plt.title('Норма невязки (Метод Зейделя)')
    plt.xlabel('Итерация')
    plt.ylabel('Норма невязки')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Вывод результатов в консоль
    print(f"Начальное приближение: {initial_guess}")
    print(f"Решение (Метод Якоби): {solution_jacobi}, Норма невязки: {residuals_jacobi[-1]}")
    print(f"Решение (Метод Зейделя): {solution_seidel}, Норма невязки: {residuals_seidel[-1]}")