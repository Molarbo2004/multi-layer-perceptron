import numpy as np

class MultiLayerPerceptron:

    # Метод начальной инициализации
    
    def __init__(self, input_size, hidden_size, num_classes, learning_rate=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Инициализация весов начальными значениями 

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.num_classes) * 0.01
        self.b2 = np.zeros((1, self.num_classes))


    # Функция Relu

    def _relu(self, z):
        return np.maximum(0, z)
    
    # Функция для производной Relu для подсчета градиентов 

    def _relu_derivative(self, z):
        return (z > 0).astype(float)
    

    # Функция активации softmax

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # Функция потерь: Категориальная кросс-энтропия

    def _cross_entropy(self, y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    # Функция прямого прохода
    
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.softmax(self.z2)
    
    # Функция Back Propagation

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        delta2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        delta1 = np.dot(delta2, self.W2.T) * self._relu_derivative(self.a1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # Функция предсказания

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    # Функция обучения

    def fit(self, X, y, epochs=1000):
        m = len(y)
        y_one_hot = np.zeros((m, self.num_classes))
        for i in range(m):
            y_one_hot[i, y[i]] = 1

        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y_one_hot, y_pred)
            
            if epoch % 100 == 0:
                loss = self._cross_entropy(y_one_hot, y_pred)
                predictions = self.predict(X)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Accuracy: {accuracy:.4f}")
                print(f"Веса W1 (среднее: {np.mean(self.W1):.6f}, std: {np.std(self.W1):.6f}):")
                print(f"Веса W2 (среднее: {np.mean(self.W2):.6f}, std: {np.std(self.W2):.6f})")
                print("---")