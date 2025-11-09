import numpy as np
from src.multi_layer_perceptron import MultiLayerPerceptron

def test_prediction_is_integer():
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, 10)
    model = MultiLayerPerceptron(input_size=2, hidden_size=5, num_classes=2)
    model.fit(X, y, epochs=10)
    preds = model.predict(X)
    assert preds.dtype == np.int64 or preds.dtype == np.int32
    assert set(np.unique(preds)) <= {0, 1}

def test_gradient_shapes():
    X = np.random.randn(5, 3)
    y = np.random.randint(0, 3, 5)
    model = MultiLayerPerceptron(input_size=3, hidden_size=4, num_classes=3)
    model.fit(X, y, epochs=1)
    assert model.W1.shape == (3, 4)
    assert model.W2.shape == (4, 3)