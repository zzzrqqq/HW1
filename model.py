import numpy as np

#ReLU激活函数
def relu(x):
    return np.maximum(0, x)

#Softmax函数
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

#交叉熵损失函数
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

#三层神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.bias2 = np.zeros((1, output_size))
        
    #网络前向传播
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return self.z2
    
    #计算梯度用于反向传播
    def compute_gradients(self, X, y):
        m = y.shape[0]
        output = self.forward(X)
        p = softmax(output)
        p[range(m), y] -= 1
        p /= m
        grad_weights2 = np.dot(self.a1.T, p)
        grad_bias2 = np.sum(p, axis=0, keepdims=True)
        delta2 = np.dot(p, self.weights2.T)
        delta2[self.z1 <= 0] = 0 
        grad_weights1 = np.dot(X.T, delta2)
        grad_bias1 = np.sum(delta2, axis=0, keepdims=True)
        return grad_weights1, grad_bias1, grad_weights2, grad_bias2
