import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# Generar datos de simulación
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Agregar un sesgo a la matriz de características X
X_b = np.c_[np.ones((100, 1)), X]

# Función de pérdida RMSE
def rmse_loss(predictions, targets):
    return np.sqrt(np.mean((predictions - targets)**2))

# Función de pérdida para el modelo de regresión lineal
def loss_function(theta, X, y):
    predictions = X.dot(theta)
    return rmse_loss(predictions, y)

# Gradiente de la función de pérdida RMSE (Primera derivada)
def gradient(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    gradients = (1 / m) * X.T.dot(error)
    return gradients

# Hessian de la función de pérdida RMSE (Segunda derivada)
def calculate_hessian(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    hessian = (1 / m) * X.T.dot(X)
    return hessian

# Método de Newton-Raphson
def newton_raphson(X, y, n_iterations):
    theta = np.random.randn(X.shape[1], 1)

    for _ in range(n_iterations):
        gradients = gradient(theta, X, y)
        hessian = calculate_hessian(theta, X, y)
        theta = theta - np.linalg.inv(hessian).dot(gradients)

    return theta


# Método para calcular las métricas
def evaluate_model(X, y, theta):
    predictions = X.dot(theta)

    # Calcular MSE
    mse = mean_squared_error(y, predictions)

    # Calcular R^2
    r2 = r2_score(y, predictions)

    print("Error Cuadrático Medio (MSE):", mse)
    print("Coeficiente de Determinación (R^2):", r2)

# Método de Descenso de Gradiente (Versión Normal)
def gradient_descent_normal(X, y, learning_rate, n_iterations):
    theta = np.random.randn(X.shape[1], 1)

    for iteration in range(n_iterations):
        gradients = gradient(theta, X, y)
        theta = theta - learning_rate * gradients

    return theta


# Método de Descenso de Gradiente Estocástico (SGD)
def stochastic_gradient_descent(X, y, learning_rate, n_epochs):
    m, n = X.shape
    theta = np.random.randn(n, 1)

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients

    return theta

# Aplicar Descenso de Gradiente Estocástico
theta_sgd = stochastic_gradient_descent(X_b, y, learning_rate=0.01, n_epochs=1000)

# Imprimir los resultados para el Descenso de Gradiente Estocástico
print("Parámetros finales Descenso de Gradiente Estocástico:", theta_sgd)
evaluate_model(X_b, y, theta_sgd)

print("\n-------------------------\n")


# Aplicar Descenso de Gradiente (Versión Normal)
theta_normal = gradient_descent_normal(X_b, y, learning_rate=0.01, n_iterations=1000)

# Imprimir los resultados para la versión normal
print("Parámetros finales Descenso de Gradiente (Versión Normal):", theta_normal)
evaluate_model(X_b, y, theta_normal)

print("\n-------------------------\n")

# Aplicar Newton-Raphson (Versión Optimizada)
theta_optimal = newton_raphson(X_b, y, n_iterations=5)

# Imprimir los resultados para la versión optimizada
print("Parámetros finales Newton-Raphson (Versión Optimizada):", theta_optimal)
evaluate_model(X_b, y, theta_optimal)


# Visualizar los datos y las rectas de regresión para Descenso de Gradiente (Normal) y Ecuación Normal
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta_normal), color='blue', label='Descenso de Gradiente (Normal)')
plt.plot(X, X_b.dot(theta_optimal), color='red', label='Newton-Raphson (Optimizado)')
plt.plot(X, X_b.dot(theta_sgd), color='green', label='Stochastic Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()