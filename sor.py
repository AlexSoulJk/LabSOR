import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

class Info:
    def __init__(self, size):
        self.history_of_norm = np.zeros(size)
    def show_history_of_norm(self):
        plt.plot(self.history_of_norm)
        plt.grid()
        plt.title("History of norm")
        plt.show()

def sinc_function(x):
    return np.sinc(x)


def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def compute_kernel_matrix(X, gamma):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(X[i], X[j], gamma)
    return K


def get_LD_matrix(A):
    D = np.diag(np.diag(A))  # Диагональная часть
    L = np.tril(A, k=-1)
    return D, L


def sor_algorithm(H, E, c, C,tolerance=1e-3, max_iter=10000, omega=1.5):
    info = Info(max_iter)
    A = H + E
    a = np.zeros_like(c)

    # Разложение A = L + D + L^T
    D = np.diag(np.diag(A))  # Диагональная часть
    L = np.tril(A, k=-1)  # Строго нижняя треугольная часть

    # Инвертированная диагональ
    diag = np.diag(D)
    if np.any(diag == 0):
        raise ValueError("Diagonal elements of D contain zeros!")
    diag_inv = 1 / diag

    for iteration in range(max_iter):
        a_prev = a.copy()

        # Обновление a с использованием цикла по j
        for j in range(len(a)):
            sum_L = np.dot(L[j, :j], a[:j])  # Вклад от строго нижней треугольной части
            sum_D_U = np.dot(A[j, j:], a_prev[j:])  # Вклад от диагональной и верхней частей
            a[j] = a_prev[j] - omega * diag_inv[j] * (sum_L + sum_D_U - c[j])
            a[j] = max(0, min(C, a[j]))  # Проекция на [0, C]

        # Проверка на сходимость
        norm = np.linalg.norm(a - a_prev)
        info.history_of_norm[iteration] = norm
        if norm < tolerance:
            break

    return a, info


def predict(x_new, x_train, a, b, gamma):
    n = len(x_train)
    y_pred = np.zeros(len(x_new))
    for i in range(len(x_new)):
        y_pred[i] = b
        for j in range(n):
            y_pred[i] += (a[j] - a[j + n]) * rbf_kernel(x_train[j], x_new[i], gamma)
    return y_pred


def sinc_test():
    l = 200
    x_min, x_max = -5, 5
    C = 1
    epsilon = 0.01
    gamma = 0.5
    x = np.linspace(x_min, x_max, l)
    y_true = sinc_function(x)

    noise = np.random.normal(0, epsilon, l) * 0
    y_noisy = y_true + noise

    K = compute_kernel_matrix(x, gamma)

    H = np.block([[K, -K], [-K, K]])
    E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])

    c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])

    a, info = sor_algorithm(H, E, c, C, omega=1.5)

    b = -np.sum(a[l:] - a[:l])

    x_test = np.linspace(x_min, x_max, 500)
    y_pred = predict(x_test, x, a, b, gamma)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True function", color="black", linestyle="--")
    plt.scatter(x, y_noisy, label="Noisy data", color="blue", alpha=0.5)
    plt.plot(x_test, y_pred, label="Predicted function", color="red")
    plt.fill_between(x_test, y_pred - epsilon, y_pred + epsilon, color="gray", alpha=0.2, label="ε-tube")
    plt.legend()
    plt.title("Support Vector Regression with SOR")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()
    info.show_history_of_norm()
    mae = mean_absolute_error(y_true, predict(x, x, a, b, gamma))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")


sinc_test()
