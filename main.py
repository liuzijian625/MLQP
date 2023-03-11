import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


def sigmoid(x):
    """归一化函数——sigmoid函数"""
    return 1 / (1 + math.exp(-x))


def sigmoid_gradient(x):
    """sigmoid函数的导数"""
    return x * (1 - x)


def linear(data, u, w, b):
    """由前一层计算后一层某个结点的值"""
    result=b+(u*data*data+w*data).sum()
    return sigmoid(result)


class OneLayerMLP:
    """只含一层隐藏层的MLP"""
    def __init__(self, train_data, test_data, learning_rate, momentum_constant):
        """初始化MLP"""
        # 保存性能最好的模型的参数、轮数和准确率
        self.j_train = None
        self.acc_train = None
        self.j_test = None
        self.acc_test = None
        # 初始化模型参数
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant
        self.input_layer=2
        self.hidden_layer=32
        self.output_layer=1
        self.train_num=5000
        self.u = np.random.random((self.hidden_layer, self.input_layer))/10
        self.w = np.random.random((self.hidden_layer, self.input_layer))/10
        self.b = np.zeros(self.hidden_layer)
        self.u_M = np.random.random(self.hidden_layer)/10
        self.w_M = np.random.random(self.hidden_layer)/10
        self.b_M = np.zeros(self.output_layer)
        # 存储每一轮的loss和准确率
        self.losses=[]
        self.accuracies=[]
        self.train_order=[]
        for i in range(self.train_num):
            self.train_order.append(i+1)

    def forward(self, data):
        """前向传播"""
        # 记录每一层节点的值
        results = [data]
        result = []
        for i in range(self.hidden_layer):
            result.append(linear(data[0:2], self.u[i], self.w[i], self.b[i]))
        results.append(np.array(result))
        result = linear(result,self.u_M, self.w_M, self.b_M)
        results.append(result)
        return results

    def backward(self, e, results):
        """反向传播"""
        # 偏置的梯度
        b_gradient = e * sigmoid_gradient(results[-1])
        u_m=self.u_M+0
        w_m = self.w_M + 0
        # 更新隐藏层到输出层的权重
        self.u_M = self.learning_rate * e * sigmoid_gradient(results[-1]) * results[
            -2] * results[-2] + self.momentum_constant * self.u_M
        self.w_M = self.learning_rate * e * sigmoid_gradient(results[-1]) * results[
            -2] + self.momentum_constant * self.w_M
        self.b_M = self.learning_rate * b_gradient + self.momentum_constant * self.b_M
        # 更新输入层到隐藏层的权重
        for i in range(self.hidden_layer):
            self.u[i] = self.learning_rate * sigmoid_gradient(results[-2][i]) * results[-3][0:2]* results[-3][0:2] * b_gradient * (2*u_m[i]*results[-2][i]+w_m[i]) + self.momentum_constant * self.u[i]
            self.w[i] = self.learning_rate * sigmoid_gradient(results[-2][i]) * results[-3][0:2] * b_gradient * (2*u_m[i]*results[-2][i]+w_m[i]) + self.momentum_constant *self.w[i]
            self.b[i] = self.learning_rate * sigmoid_gradient(results[-2][i]) * b_gradient *  (2*u_m[i]*results[-2][i]+w_m[i])+ self.momentum_constant * self.b[i]

    def train(self):
        """训练模型"""
        self.acc_train = 0
        # 对模型训练train_num轮
        for j in tqdm(range(self.train_num)):
            error_sum=0
            '''print("第" + str(j + 1) + "轮训练")'''
            i = 1
            # 每次使用一个样本点进行训练，300个样本点都训一遍为一轮
            for data in self.train_data:
                '''print("第" + str(i) + "次训练")'''
                results = self.forward(data)
                e = data[2] - results[-1]
                error = 0.5 * e * e
                error_sum=error_sum+error
                self.backward(e, results)
                '''print("error:" + str(error))'''
                i = i + 1
            # 用此轮模型去预测train_data
            acc_now = self.test(self.train_data)
            self.accuracies.append(acc_now)
            self.losses.append(error_sum)
            '''print(acc_now)
            print(error_sum)'''
            # 保存效果最好的模型
            if acc_now > self.acc_train:
                self.j_train = j + 0
                self.acc_train = acc_now + 0
                self.u_max_train=self.u+0
                self.w_max_train = self.w + 0
                self.b_max_train = self.b + 0
                self.u_M_max_train = self.u_M + 0
                self.w_M_max_train = self.w_M + 0
                self.b_M_max_train = self.b_M + 0
            '''acc_now = self.test(self.test_data)
            if acc_now > self.acc_test:
                self.j_test = j + 0
                self.acc_test = acc_now + 0
                self.w_max_test = self.w + 0
                self.b_max_test = self.b + 0
                self.w_M_max_test = self.w_M + 0
                self.b_M_max_test = self.b_M + 0'''
        print("训练集最好结果在第" + str(self.j_train + 1) + "轮产生")
        print("准确率：" + str(self.acc_train) + "%")
        '''print("测试集最好结果在第" + str(self.j_test + 1) + "轮产生")
        print("准确率：" + str(self.acc_test) + "%")'''

    def test(self, datas):
        """用datas去测试模型的准确率"""
        right_num = 0
        total_num = 0
        for data in datas:
            total_num = total_num + 1
            results = self.forward(data)
            if abs(results[-1] - data[2]) < 0.5:
                right_num = right_num + 1
        '''print("正确率：" + str(100 * right_num / total_num) + "%")'''
        return 100 * right_num / total_num

    def decision_boundary(self):
        """绘制决策边界"""
        x_min = min(self.train_data[:, 0].min() - .5, self.test_data[:, 0].min() - .5)
        x_max = min(self.train_data[:, 0].max() + .5, self.test_data[:, 0].max() + .5)
        y_min = min(self.train_data[:, 1].min() - .5, self.test_data[:, 1].min() - .5)
        y_max = min(self.train_data[:, 1].max() + .5, self.test_data[:, 1].max() + .5)
        h = 0.01
        # 对数据分布的区域进行网格划分
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        zz = []
        self.train_prep()
        # 计算每个网格点对应的预测值
        for i in tqdm(range(len(xx))):
            z = []
            for j in range(len(xx[0])):
                if self.forward([xx[i][j], yy[i][j]])[-1] > 0.5:
                    z.append(1)
                else:
                    z.append(0)
            zz.append(z)
        # 绘制决策边界
        plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral)
        # 绘制所有数据点
        for data in train_data:
            if data[2] == 1:
                plt.plot(data[0], data[1], color='blue', marker='o')
            else:
                plt.plot(data[0], data[1], color='red', marker='o')
        for data in test_data:
            if data[2] == 1:
                plt.plot(data[0], data[1], color='blue', marker='x')
            else:
                plt.plot(data[0], data[1], color='red', marker='x')
        plt.title("Decision Boundary")
        plt.show()

    '''def test_prep(self):
        self.b_M = self.b_M_max_test
        self.b = self.b_max_test
        self.w = self.w_max_test
        self.w_M = self.w_M_max_test'''

    def train_prep(self):
        """加载表现最好的模型权重参数"""
        self.u=self.u_max_train
        self.u_M=self.u_M_max_train
        self.b_M = self.b_M_max_train
        self.b = self.b_max_train
        self.w = self.w_max_train
        self.w_M = self.w_M_max_train

    def loss_change(self):
        """绘制loss的变化曲线"""
        plt.clf()
        plt.plot(self.train_order,self.losses)
        plt.show()

    def accuracy_change(self):
        """绘制准确率的变化曲线"""
        plt.clf()
        plt.plot(self.train_order, self.accuracies)
        plt.show()


if __name__ == '__main__':
    # 加载数据
    train_data = np.loadtxt('two_spiral_train_data.txt')
    test_data = np.loadtxt('two_spiral_test_data.txt')
    learning_rate = 0.05
    momentum_constant = 1
    MLP = OneLayerMLP(train_data, test_data, learning_rate, momentum_constant)
    MLP.train()
    MLP.decision_boundary()
    '''MLP.test_prep()'''
    acc=MLP.test(test_data)
    print("测试集准确率：" + str(acc) + "%")
    MLP.loss_change()
    MLP.accuracy_change()

