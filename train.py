import numpy as np
from model import cross_entropy_loss, softmax
from test import accuracy
import pickle
import matplotlib.pyplot as plt

def update_weights(model, gradients, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,reg_lambda=0.01):
    # 初始化动量和速率项
    if not hasattr(model, 'm1_weights'):
        model.m1_weights = [np.zeros_like(w) for w in [model.weights1, model.weights2]]
        model.m2_weights = [np.zeros_like(w) for w in [model.weights1, model.weights2]]
        model.m1_biases = [np.zeros_like(b) for b in [model.bias1, model.bias2]]
        model.m2_biases = [np.zeros_like(b) for b in [model.bias1, model.bias2]]

    # 解包梯度
    grad_weights1, grad_bias1, grad_weights2, grad_bias2 = gradients

    # 更新权重和偏置
    updates = [(model.weights1, grad_weights1, model.m1_weights[0], model.m2_weights[0]),
               (model.weights2, grad_weights2, model.m1_weights[1], model.m2_weights[1]),
               (model.bias1, grad_bias1, model.m1_biases[0], model.m2_biases[0]),
               (model.bias2, grad_bias2, model.m1_biases[1], model.m2_biases[1])]

    for (w, grad, m1, m2) in updates:
        m1[:] = beta1 * m1 + (1 - beta1) * grad
        m2[:] = beta2 * m2 + (1 - beta2) * (grad ** 2)
        m1_corrected = m1 / (1 - beta1 ** t)
        m2_corrected = m2 / (1 - beta2 ** t)
        w -= lr * m1_corrected/ (np.sqrt(m2_corrected) + epsilon)

    t += 1

#训练和验证模型，记录每个epoch的损失和准确率，并保存最优模型
def train_and_validate(model, X_train, y_train, X_val, y_val, epochs, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, reg_lambda=0.01):
    
    best_val_loss = float('inf')
    best_model_params = {}
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    t = 1 

    for epoch in range(epochs):
        # 计算梯度并更新权重
        gradients = model.compute_gradients(X_train, y_train)
        update_weights(model, gradients, t, lr, beta1, beta2, epsilon)

        # 计算训练损失和准确率
        train_output = model.forward(X_train)
        train_loss = cross_entropy_loss(train_output, y_train)
        train_pred = np.argmax(softmax(train_output), axis=1)
        train_acc = np.mean(train_pred == y_train)

        # 计算验证损失和准确率
        val_output = model.forward(X_val)
        val_loss = cross_entropy_loss(val_output, y_val)
        val_pred = np.argmax(softmax(val_output), axis=1)
        val_acc = np.mean(val_pred == y_val)

        # 保存历史数据
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_params = {
                'weights1': model.weights1.copy(),
                'bias1': model.bias1.copy(),
                'weights2': model.weights2.copy(),
                'bias2': model.bias2.copy(),
            }

        # 学习率衰减
        lr *= 0.95

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")

    # 恢复最优模型权重
    model.weights1, model.bias1 = best_model_params['weights1'], best_model_params['bias1']
    model.weights2, model.bias2 = best_model_params['weights2'], best_model_params['bias2']

    return history

#保存模型权重到文件
def save_model(model, filename):
    
    with open(filename, 'wb') as f:
        pickle.dump({
            'weights1': model.weights1,
            'bias1': model.bias1,
            'weights2': model.weights2,
            'bias2': model.bias2,
        }, f)


#绘制训练和验证损失及准确率曲线
def plot_history(history):
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()