import numpy as np
import matplotlib
from tqdm import tqdm

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
        if (0 < a[i] and np.isclose(a[i+l], 0)) or (0 < a[i + l] and np.isclose(a[i], 0)):  # Проверяем условия для a_i и a_i^*
            support_vector_indices.append(i)

    support_vectors = x_train[support_vector_indices]
    return support_vector_indices, support_vectors

def extract_bounded_support_vectors(a, x_train, C):
    """
    Выделяет связные векторы (граничные опорные векторы).
    :param a: Массив двойственных переменных (размер 2*l).
    :param x_train: Обучающие данные (размер l).
    :param C: Параметр регуляризации.
    :return: Индексы связных векторов и их координаты.
    """
    l = len(x_train)
    bounded_support_vector_indices = []

    for i in range(l):
        if np.isclose(a[i], C) or np.isclose(a[i + l], C):  # Проверяем условия для a_i и a_i^*
            bounded_support_vector_indices.append(i)

    bounded_support_vectors = x_train[bounded_support_vector_indices]
    return bounded_support_vector_indices, bounded_support_vectors


def sinc_test():
    omega = 1.0
    noise_sigma = 0.35
    l = 200
    x_min, x_max = -5, 5
    C = 15
    epsilon = noise_sigma + 0.042
    gamma = 1
    x = np.linspace(x_min, x_max, l)
    y_true = sinc_function(x)

    noise = np.random.normal(0, noise_sigma, l)
    y_noisy = y_true + noise

    K = compute_kernel_matrix(x, gamma)

    H = np.block([[K, -K], [-K, K]])
    E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])

    c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])

    a, info = sor_algorithm(H, E, c, C, omega=omega)

    b = -np.sum(a[l:] - a[:l])

    x_test = np.linspace(x_min, x_max, 500)
    y_pred = predict(x_test, x, a, b, gamma)

    support_vector_indices, support_vectors = extract_support_vectors(a, x, C)
    bounded_support_vector_indices, bounded_support_vectors = extract_bounded_support_vectors(a, x, C)
    print(f"Количество опорных векторов: {len(support_vectors)}")
    print(f"Кол-во связных векторов: {len(bounded_support_vector_indices)}")
    print(f"Индексы опорных векторов: {support_vector_indices}")
    print(f"Пересекающиеся индексы: {np.intersect1d(support_vector_indices, bounded_support_vectors)}")
    print(f"Опорные векторы: {support_vectors}")
    print(f"Кол-во итераций {len(info.history_of_norm[info.history_of_norm > 0.0])}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True function", color="black", linestyle="--")
    plt.scatter(x, y_noisy, label="Noisy data", color="blue", alpha=0.5)
    plt.scatter(x[support_vector_indices], y_noisy[support_vector_indices], color="red", alpha=0.9, label="Support vectors")
    plt.scatter(x[bounded_support_vector_indices], y_noisy[bounded_support_vector_indices], color="green", alpha=0.9,
                label="Bound vectors")
    plt.plot(x_test, y_pred, label="Predicted function", color="red")
    plt.fill_between(x_test, y_pred - epsilon, y_pred + epsilon, color="gray", alpha=0.2, label="ε-tube")
    plt.legend()
    plt.title("Support Vector Regression with SOR")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig(f"With_Noise_{noise_sigma}.png")
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
    C = 10
    epsilon = 0.01
    gamma = 1
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
    bounded_support_vector_indices, bounded_support_vectors = extract_bounded_support_vectors(a, x, C)
    print(f"Количество опорных векторов: {len(support_vectors)}")
    print(f"Кол-во связных векторов: {len(bounded_support_vector_indices)}")
    print(f"Индексы опорных векторов: {support_vector_indices}")
    print(f"Пересекающиеся индексы: {np.intersect1d(support_vector_indices, bounded_support_vectors)}")
    print(f"Опорные векторы: {support_vectors}")
    print(f"Кол-во итераций {len(info.history_of_norm[info.history_of_norm > 0.0])}")
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True function", color="black", linestyle="--")
    plt.scatter(x, y_noisy, label="Noiseless data", color="blue", alpha=0.5, s=10)
    plt.plot(x_test, y_pred, label="Predicted function", color="red")
    plt.fill_between(x_test, y_pred - epsilon, y_pred + epsilon, color="gray", alpha=0.2, label="ε-tube")
    plt.scatter(x[support_vector_indices], y_noisy[support_vector_indices], color="red", alpha=0.9, label="Support vectors")
    plt.scatter(x[bounded_support_vector_indices], y_noisy[bounded_support_vector_indices], color="green", alpha=0.9,
                label="Bound vectors")
    plt.legend()
    plt.title("Support Vector Regression without Noise")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig(f"WithOut_Noise_{epsilon}.png")
    plt.show()

    mae = mean_absolute_error(y_true, predict(x, x, a, b, gamma))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")





def study_mae_vs_support_vectors():
    """
    Исследование зависимости ошибки MAE от количества опорных векторов.
    """
    l = 200
    x_min, x_max = -5, 5
    noise_sigma = 0.1
    x = np.linspace(x_min, x_max, l)
    gamma = 2
    y_true = sinc_function(x)
    C = 7
    # Добавление шума
    noise = np.random.normal(0, noise_sigma, l)
    y_noisy = y_true + noise

    # Параметры исследования
    epsilon_values = np.linspace(0.15, 0.35, 20)
    results = []

    # Список для хранения количества итераций
    iteration_counts = []

    support_vector_counts = []

    for epsilon in tqdm(epsilon_values, desc="Epsilon Variation Progress", unit="value"):
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

        num_iterations = len(info.history_of_norm[info.history_of_norm > 0.0])
        iteration_counts.append(num_iterations)

        results.append((epsilon, num_support_vectors, mae))

    # Преобразование результатов в массивы
    epsilon_values_, num_support_vectors, mae_values = zip(*results)

    # Визуализация
    plt.figure(figsize=(16, 16))  # Увеличиваем размер фигуры для 2x2 сетки

    # График 1: Зависимость MAE от числа опорных векторов
    plt.subplot(2, 2, 1)
    plt.plot(num_support_vectors, mae_values, marker='o', color='blue')
    plt.xlabel("Number of Support Vectors")
    plt.ylabel("MAE")
    plt.title("MAE vs Number of Support Vectors")
    plt.grid()

    # График 2: Зависимость MAE от параметра epsilon
    plt.subplot(2, 2, 2)
    plt.plot(epsilon_values_, mae_values, marker='o', color='green')
    plt.xlabel("Eps (Tube parameter)")
    plt.ylabel("MAE")
    plt.title("MAE vs eps")
    plt.grid()

    # График 3: Зависимость числа опорных векторов от параметра epsilon
    plt.subplot(2, 2, 3)
    plt.plot(epsilon_values_, num_support_vectors, marker='o', color='purple')
    plt.xlabel("Eps (Tube parameter)")
    plt.ylabel("Number of Support Vectors")
    plt.title("Number of Support Vectors vs eps")
    plt.grid()

    # График 4: Зависимость количества итераций от параметра epsilon
    plt.subplot(2, 2, 4)
    plt.plot(epsilon_values_, iteration_counts, marker='o', color='red')
    plt.xlabel("Eps (Tube parameter)")
    plt.ylabel("Number of Iterations")
    plt.title("Number of Iterations vs eps")
    plt.grid()

    plt.tight_layout()
    plt.savefig("Amount_sv_vs_MAE.jpg")
    plt.show()

# study_mae_vs_support_vectors()

sinc_test()
test_without_noise()

# study_mae_vs_support_vectors()