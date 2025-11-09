# Multi-Layer Perceptron from Scratch

Реализация MLP (многослойного персептрона) на чистом NumPy — без использования sklearn, PyTorch или TensorFlow.

## Особенности
- Один скрытый слой с активацией ReLU
- Softmax + Cross-Entropy для многоклассовой классификации
- Backpropagation реализован вручную
- Онлайн-обучение (полный батч)
- Сравнение с `sklearn.neural_network.MLPClassifier`

## Установка
pip install -r requirements.txt 
pip install -e .

## Для запуска demo_moons.py 
py examples/demo_moons.py 

## Для просмотра математики и выполнения кода в .ipynb 
1_math_and_derivation.ipynb - пояснения и математические формулы (градиенты, веса и т.п) 
2_vs_sklearn - сравнение точности с Sklearn нашей реализации MLP 
