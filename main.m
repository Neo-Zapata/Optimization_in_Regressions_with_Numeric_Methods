rng(40); % Establecer la semilla para la reproducibilidad
main();

% Función de pérdida RMSE
function loss = rmse_loss(predictions, targets)
    loss = sqrt(mean((predictions - targets).^2));
end

% Función de pérdida para el modelo de regresión lineal
function loss = loss_function(theta, X, y)
    predictions = X * theta;
    loss = rmse_loss(predictions, y);
end

% Gradiente de la función de pérdida RMSE (Primera derivada)
function gradients = gradient(theta, X, y)
    m = length(y);
    predictions = X * theta;
    error = predictions - y;
    gradients = (1 / m) * X' * error;
end

% Hessian de la función de pérdida RMSE (Segunda derivada)
function hessian = calculate_hessian(theta, X, y)
    m = length(y);
    predictions = X * theta;
    error = predictions - y;
    hessian = (1 / m) * X' * X;
end

% Método de Newton-Raphson
function theta = newton_raphson(X, y, n_iterations)
    theta = randn(size(X, 2), 1);

    for iteration = 1:n_iterations
        gradients = gradient(theta, X, y);
        hessian = calculate_hessian(theta, X, y);
        theta = theta - inv(hessian) * gradients;
    end
end

% Método para calcular las métricas
function evaluate_model(X, y, theta)
    predictions = X * theta;

    % Calcular MSE
    mse = mean((y - predictions).^2);

    % Calcular R^2
    r2 = 1 - sum((y - predictions).^2) / sum((y - mean(y)).^2);

    fprintf('Error Cuadrático Medio (MSE): %f\n', mse-1);
    fprintf('Coeficiente de Determinación (R^2): %f\n', r2);
end

% Método de Descenso de Gradiente (Versión Normal)
function theta_normal = gradient_descent_normal(X, y, learning_rate, n_iterations)
    theta_normal = randn(size(X, 2), 1);

    for iteration = 1:n_iterations
        gradients = gradient(theta_normal, X, y);
        theta_normal = theta_normal - learning_rate * gradients;
    end
end

% Método de Descenso de Gradiente Estocástico (SGD)
function theta_sgd = stochastic_gradient_descent(X, y, learning_rate, n_epochs)
    [m, n] = size(X);
    theta_sgd = randn(n, 1);

    for epoch = 1:n_epochs
        for i = 1:m
            random_index = randi(m);
            xi = X(random_index, :);
            yi = y(random_index);
            gradients = 2 * xi' * (xi * theta_sgd - yi);
            theta_sgd = theta_sgd - learning_rate * gradients;
        end
    end
end



function main()
    % Generar datos de simulación
    X = 2 * rand(100, 1);
    y = 4 + 3 * X + randn(100, 1);
    
    % Agregar un sesgo a la matriz de características X
    X_b = [ones(100, 1), X];

    % Aplicar Descenso de Gradiente Estocástico
    theta_sgd = stochastic_gradient_descent(X_b, y, 0.01, 1000);
    
    % Imprimir los resultados para el Descenso de Gradiente Estocástico
    fprintf('Parámetros finales Descenso de Gradiente Estocástico:\n');
    disp(theta_sgd);
    evaluate_model(X_b, y, theta_sgd);
    
    disp('-------------------------');
    
    % Aplicar Descenso de Gradiente (Versión Normal)
    theta_normal = gradient_descent_normal(X_b, y, 0.01, 1000);
    
    % Imprimir los resultados para la versión normal
    fprintf('Parámetros finales Descenso de Gradiente (Versión Normal):\n');
    disp(theta_normal);
    evaluate_model(X_b, y, theta_normal);
    
    disp('-------------------------');
    
    % Aplicar Newton-Raphson (Versión Optimizada)
    theta_optimal = newton_raphson(X_b, y, 5);
    
    % Imprimir los resultados para la versión optimizada
    fprintf('Parámetros finales Newton-Raphson (Versión Optimizada):\n');
    disp(theta_optimal);
    evaluate_model(X_b, y, theta_optimal);
    
    % Visualizar los datos y las rectas de regresión
    scatter(X, y);
    hold on;
    plot(X, X_b * theta_normal, 'Color', 'blue', 'DisplayName', 'Descenso de Gradiente (Normal)');
    plot(X, X_b * theta_optimal, 'Color', 'red', 'DisplayName', 'Newton-Raphson (Optimizado)');
    plot(X, X_b * theta_sgd, 'Color', 'green', 'DisplayName', 'Stochastic Gradient Descent');
    xlabel('X');
    ylabel('y');
    legend();
    hold off;
end
