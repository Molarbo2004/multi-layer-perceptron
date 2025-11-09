import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from src.multi_layer_perceptron import MultiLayerPerceptron

# Данные
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# Обучение
my_model = MultiLayerPerceptron(input_size=2, hidden_size=10, num_classes=2, learning_rate=0.5)
my_model.fit(X, y, epochs=1000)

# Оценка
preds = my_model.predict(X)
accuracy = np.mean(preds == y)
print(f"Точность: {accuracy:.4f}")

# Визуализация
h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = my_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title(f'MLP (Точность:Q {accuracy:.2%})')
plt.show()