import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

np.random.seed(42)


class Info:
    def __init__(self, size):
        self.history_of_norm = np.zeros(size)
        self.amount_of_sv = 0

    def show_history_of_norm(self):
        plt.plot(np.log(self.history_of_norm))
        plt.grid()
        plt.title("History of norm")
        plt.show()


def sinc_function(x):
    return np.sin(np.pi * x/4.0) + 0.5 * np.sin(np.pi * x)


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


def sor_algorithm(H, E, c, C, tolerance=1e-3, max_iter=10000, omega=1.5):
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


def extract_support_vectors(a, x_train, C):
    """
    Выделяет опорные векторы из массива a.
    :param a: Массив двойственных переменных (размер 2*l).
    :param x_train: Обучающие данные (размер l).
    :param C: Параметр регуляризации.
    :return: Индексы опорных векторов и их координаты.
    """
    l = len(x_train)
    support_vector_indices = []

    for i in range(l):
        if 0 < a[i] < C or 0 < a[i + l] < C:  # Проверяем условия для a_i и a_i^*
            support_vector_indices.append(i)

    support_vectors = x_train[support_vector_indices]
    return support_vector_indices, support_vectors


def sinc_test():
    l = 200
    x_min, x_max = -5, 5
    C = 3
    epsilon = 0.07
    gamma = 0.7
    x = np.linspace(x_min, x_max, l)
    y_true = sinc_function(x)

    noise = np.random.normal(0, 0.1, l)
    y_noisy = y_true + noise

    K = compute_kernel_matrix(x, gamma)

    H = np.block([[K, -K], [-K, K]])
    E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])

    c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])

    a, info = sor_algorithm(H, E, c, C, omega=1)

    b = -np.sum(a[l:] - a[:l])

    x_test = np.linspace(x_min, x_max, 500)
    y_pred = predict(x_test, x, a, b, gamma)

    support_vector_indices, support_vectors = extract_support_vectors(a, x, C)

    print(f"Количество опорных векторов: {len(support_vectors)}")
    print(f"Индексы опорных векторов: {support_vector_indices}")
    print(f"Опорные векторы: {support_vectors}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True function", color="black", linestyle="--")
    plt.scatter(x, y_noisy, label="Noisy data", color="blue", alpha=0.5)
    plt.scatter(x[support_vector_indices], y_noisy[support_vector_indices], color="red", alpha=0.9, label="Support vectors")
    plt.plot(x_test, y_pred, label="Predicted function", color="red")
    plt.fill_between(x_test, y_pred - epsilon, y_pred + epsilon, color="gray", alpha=0.2, label="ε-tube")
    plt.legend()
    plt.title("Support Vector Regression with SOR")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig("With_Noise.png")
    plt.show()

    info.show_history_of_norm()
    mae = mean_absolute_error(y_true, predict(x, x, a, b, gamma))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

def test_without_noise():
    """
    Тестирование SVR на данных без шума.
    """
    l = 200
    x_min, x_max = -5, 5
    C = 3
    epsilon = 0.07
    gamma = 0.7
    x = np.linspace(x_min, x_max, l)
    y_true = sinc_function(x)

    # Без шума
    y_noisy = y_true  # Зашумление отсутствует

    K = compute_kernel_matrix(x, gamma)

    H = np.block([[K, -K], [-K, K]])
    E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])

    c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])

    a, info = sor_algorithm(H, E, c, C, omega=1)

    b = -np.sum(a[l:] - a[:l])

    x_test = np.linspace(x_min, x_max, 500)
    y_pred = predict(x_test, x, a, b, gamma)
    support_vector_indices, support_vectors = extract_support_vectors(a, x, C)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True function", color="black", linestyle="--")
    plt.scatter(x, y_noisy, label="Noiseless data", color="blue", alpha=0.5, s=10)
    plt.plot(x_test, y_pred, label="Predicted function", color="red")
    plt.fill_between(x_test, y_pred - epsilon, y_pred + epsilon, color="gray", alpha=0.2, label="ε-tube")
    plt.scatter(x[support_vector_indices], y_noisy[support_vector_indices], color="red", alpha=0.9,
                label="Support vectors")
    plt.legend()
    plt.title("Support Vector Regression without Noise")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig("WithOut_Noise.png")
    plt.show()

    mae = mean_absolute_error(y_true, predict(x, x, a, b, gamma))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

def study_mae_vs_support_vectors():
    """
    Исследование зависимости ошибки MAE от количества опорных векторов.
    """
    l = 200
    x_min, x_max = -7, 7
    epsilon = 0.1
    gamma = 1
    x = np.linspace(x_min, x_max, l)
    y_true = sinc_function(x)

    # Добавление шума
    noise = np.random.normal(0, epsilon / 2, l)
    y_noisy = y_true + noise

    # Параметры исследования
    C_values = np.logspace(-2, 2, 20)  # Различные значения C
    results = []

    for C in C_values:
        K = compute_kernel_matrix(x, gamma)

        H = np.block([[K, -K], [-K, K]])
        E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])

        c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])

        a, info = sor_algorithm(H, E, c, C, omega=1)

        b = -np.sum(a[l:] - a[:l])

        # Вычисление MAE
        y_pred = predict(x, x, a, b, gamma)
        mae = mean_absolute_error(y_true, y_pred)

        # Подсчёт количества опорных векторов
        support_vector_indices, _ = extract_support_vectors(a, x, C)
        num_support_vectors = len(support_vector_indices)

        results.append((C, num_support_vectors, mae))

    # Преобразование результатов в массивы
    C_values, num_support_vectors, mae_values = zip(*results)

    # Визуализация
    plt.figure(figsize=(12, 6))

    # График зависимости MAE от числа опорных векторов
    plt.subplot(1, 2, 1)
    plt.plot(num_support_vectors, mae_values, marker='o', color='blue')
    plt.xlabel("Number of Support Vectors")
    plt.ylabel("MAE")
    plt.title("MAE vs Number of Support Vectors")
    plt.grid()

    # График зависимости MAE от параметра C
    plt.subplot(1, 2, 2)
    plt.plot(C_values, mae_values, marker='o', color='green')
    plt.xscale('log')
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel("MAE")
    plt.title("MAE vs C")
    plt.grid()

    plt.tight_layout()
    plt.savefig("Amount_sv_vs_MAE.jpg")
    plt.show()

sinc_test()
# test_without_noise()

# study_mae_vs_support_vectors()