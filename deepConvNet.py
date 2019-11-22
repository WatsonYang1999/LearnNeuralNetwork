#coding=utf-8
from net.layers import *
from net.convolution import Convolution
from net.pooling import Pooling
from train.trainer import Trainer
from util.preprocess import *
import matplotlib.pyplot as plt
class DeepConvNet:
    """识别率为99%以上的高精度的ConvNet

    网络结构如下所示
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):
        # 初始化权重===========
        # 各层的神经元平均与前一层的几个神经元有连接（TODO:自动计算）
        pre_node_nums = np.array(
            [1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # 使用ReLU的情况下推荐的初始值

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(
                [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                       pre_channel_num,
                                                                                       conv_param['filter_size'],
                                                                                       conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 生成层===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads

def one_hot_to(T: np.ndarray):
    answer = np.zeros(T.shape[0], dtype=int)
    for idx, row in enumerate(T):
        answer[idx] = np.argmax(row)
    return answer

def to_one_hot(X: np.ndarray):
    # x like [1,2,3]
    # return -> [ [0,1,0,0,....],[],[]]
    T = np.zeros((X.size, 10), dtype=int)
    # print(T)
    for idx, xi in enumerate(X):
        T[idx][xi] = 1
    return T

if __name__ =='__main__':

    X_pred = np.load("../data/public_data/X_kannada_MNIST_test.npz")['arr_0']
    X_train = np.load("../data/public_data/X_kannada_MNIST_train.npz")['arr_0']
    y_train = np.load("../data/public_data/y_kannada_MNIST_train.npz")['arr_0']

    X_pred = X_pred.reshape([X_pred.shape[0], 1,28,28])
    X_train = X_train.reshape([X_train.shape[0], 1,28,28])
    # for i in range(0,80):
    #     plt.subplot(8, 10, i + 1)
    #
    #     plt.imshow(X_train[[i],[0]].reshape(28, 28))
    #
    # plt.show()
    print(X_train.shape)
    y_train = to_one_hot(y_train)
    test_size = 10000
    #test_mask = np.random.choice(X_train.shape[0], test_size)

    X_test = X_train[0:test_size]
    y_test = y_train[0:test_size]
    X_train = X_train[test_size:]
    y_train = y_train[test_size:]

    network = DeepConvNet()
    trainer = Trainer(network, X_train, y_train, X_test, y_test,
                      epochs=100, mini_batch_size=100,
                      optimizer='adagrad', optimizer_param={'lr': 0.01},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    y_pred = network.predict(X_pred)
    print(y_pred.shape)
    y_pred = one_hot_to(y_pred)
    submission = pd.DataFrame(y_pred, columns=["label"])

    submission.index += 1
    submission.to_csv("submission.csv", index_label='id')

