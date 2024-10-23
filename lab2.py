import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return np.sin(x)

def right_prymougol(a, b, n): # квадратурная формула правых прямоугольников
  h = (b - a) / n
  x = np.linspace(a + h, b, n)
  return h * np.sum(f(x))

def gaus_3_porydka(a, b): # квадратурная формула Гаусса 3-го порядка
  c = [(5.0/9.0), (8.0/9.0), (5.0/9.0)]
  x = [(-np.sqrt(3.0/5.0)), (0.0), (np.sqrt(3.0/5.0))]
  return (b - a) / 2 * np.sum([c[i] * f((a + b) / 2 + (b - a) / 2 * x[i]) for i in range(3)])

def rule_runge(I1, I2, p): # правило Рунге для оценки погрешности
  return abs(I2 - I1) / (2**p - 1)

def visual(a, b, n, I1, I2, pogreshnost, gaus_points=None):
  x = np.linspace(a, b, 100)
  y = f(x)

  plt.plot(x, y, label="f(x)")
  plt.fill_between(x, y, 0, alpha=0.3, label="Площадь")

  # Визуализация прямоугольников (только для метода правых прямоугольников)
  if I1 is not None:
    h = (b - a) / n
    x_prymougol = np.linspace(a + h, b, n)
    y_prymougol = f(x_prymougol)
    plt.bar(x_prymougol, y_prymougol, width=h, alpha=0.5, label="Правые прямоугольники")

  # Визуализация точек Гаусса
  if gaus_points is not None:
    plt.scatter(gaus_points, [f(point) for point in gaus_points], color='blue', label="Точки Гаусса")

  plt.xlabel("x")
  plt.ylabel("y")
  plt.title(f"Интеграл от {a} до {b}\nПогрешность: {pogreshnost:.4f}")
  plt.legend()
  plt.show()

if __name__ == "__main__":
  a = float(input("Введите нижний предел интегрирования (a): "))
  b = float(input("Введите верхний предел интегрирования (b): "))
  tochnost = float(input("Задайте точность вычисления (Пример:1e-4) (accuracy): "))
  n = int(input("Введите кол-во разбиений (n): "))
  p = 2  # Порядок метода Рунге

  I1 = right_prymougol(a, b, n)
  I2 = right_prymougol(a, b, 2 * n)
  pogreshnost = rule_runge(I1, I2, p)

  while pogreshnost > tochnost:
    n *= 2
    I1 = right_prymougol(a, b, n)
    I2 = right_prymougol(a, b, 2 * n)
    pogreshnost = rule_runge(I1, I2, p)

  print(f"Интеграл по формуле правых прямоугольников: {I2:.4f}")
  print(f"Шаг: {(b - a) / n:.4f}")
  print(f"Количество разбиений: {n}")
  visual(a, b, n, I1, I2, pogreshnost)

  I1_gaus = gaus_3_porydka(a, b)
  I2_gaus = gaus_3_porydka(a, b)
  pogreshnost_gausa = rule_runge(I1_gaus, I2_gaus, p)

  # Вычисление точек Гаусса
  c = [(5.0/9.0), (8.0/9.0), (5.0/9.0)]
  x = [(-np.sqrt(3.0/5.0)), (0.0), (np.sqrt(3.0/5.0))]
  gaus_points = [(a + b) / 2 + (b - a) / 2 * point for point in x]

  print(f"Интеграл по формуле Гаусса 3-го порядка: {I2_gaus:.4f}")
  print(f"Погрешность: {pogreshnost_gausa:.4f}")
  visual(a, b, None, None, I2_gaus, pogreshnost_gausa, gaus_points=gaus_points)