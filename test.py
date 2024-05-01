import numpy as np
from model import NeuralNetwork, softmax, cross_entropy_loss

#计算准确率
def accuracy(y_pred, y_true):
    
    return np.mean(y_pred == y_true)

#测试模型
def test(model, X_test, y_test):
   
    output = model.forward(X_test)
    predictions = np.argmax(softmax(output), axis=1)
    return accuracy(predictions, y_test)
