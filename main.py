import os
from data_utils import read_idx
from model import NeuralNetwork
from train import train_and_validate,plot_history,save_model
from test import test
import pickle

# 数据加载
X_train = read_idx('train-images-idx3-ubyte')
y_train = read_idx('train-labels-idx1-ubyte')
X_val = read_idx('t10k-images-idx3-ubyte')
y_val = read_idx('t10k-labels-idx1-ubyte')

# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0

# 初始化模型
model = NeuralNetwork(input_size=784, hidden_size=256, output_size=10)

# 训练模型
history = train_and_validate(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, reg_lambda=0.01)

#导出图像
plot_history(history)

# 保存最佳模型
save_model(model, 'best_model.pkl')

# 测试模型
test_accuracy = test(model, X_val, y_val)
print(f"Test Accuracy: {test_accuracy}")
