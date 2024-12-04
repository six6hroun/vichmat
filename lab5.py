import numpy as np

# Метод прямой итерации
def power_iteration(A, x0, tol, max_iter=100):
    x = x0 / np.linalg.norm(x0)
    prev_x = x
    iter_count = 0
    while True:
        x = np.dot(A, x)
        x = x / np.linalg.norm(x)
        if np.linalg.norm(x - prev_x) < tol:
            break
        prev_x = x
        iter_count += 1
        if iter_count > max_iter:
            print("Метод прямой итерации не сошелся.")
            break

    eigenvalue = np.dot(np.transpose(x), np.dot(A, x))
    return eigenvalue, x, iter_count


# Метод вращений
def qr_algorithm(A, tol, max_iter=100):
    n = A.shape[0]
    Q = np.eye(n)
    A_qr = A
    eigenvalues = []
    eigenvectors = []
    iter_count = 0
    while True:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A_qr[i, j]) < tol:
                    continue

                c = A_qr[i, i] - A_qr[j, j]
                s = -2 * A_qr[i, j] / c
                t = np.sqrt(abs((s + 1) / 2))
                c /= (s + 1)
                s *= t

                G = np.eye(n)
                G[i, i] = c
                G[i, j] = s
                G[j, i] = -s
                G[j, j] = c

                A_qr = np.dot(np.dot(G, A_qr), np.transpose(G))
                Q = np.dot(G, Q)

        eigenvalues_qr, eigenvectors_qr = np.linalg.eig(A_qr)
        eigenvalues.extend(eigenvalues_qr)
        eigenvectors.extend(eigenvectors_qr.T)
        if np.allclose(np.max(abs(eigenvalues - eigenvalues_qr)), 0, atol=tol):
            break
        eigenvalues = eigenvalues_qr
        iter_count += 1
        if iter_count > max_iter:
            print("Метод вращений не сошелся.")
            break

    return np.array(eigenvalues), np.array(eigenvectors), iter_count


# Пример
A = np.array([[10, 2], [7, 1]])
tol = 1e-6

# Метод прямой итерации
eigenvalue, eigenvector, iter_count = power_iteration(A, np.array([1, 1]), tol)
print("Метод прямой итерации:")
print("Собственное число:", eigenvalue)
print("Собственный вектор:", eigenvector)
print("Количество итераций:", iter_count)

# Метод вращений
eigenvalues, eigenvectors, iter_count = qr_algorithm(A, tol)
print("\nМетод вращений:")
print("Собственные числа:", eigenvalues)
print("Собственные векторы:")
for i in range(len(eigenvalues)):
    print(eigenvectors[:, i])
print("Количество итераций:", iter_count)
