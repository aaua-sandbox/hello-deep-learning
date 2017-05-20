# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    # 初期化
    # - input_size: 入力層のニューロンの数
    # - hidden_size: 隠れ層のニューロンの数
    # - output_size: 出力層のニューロンの数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 認識(推論)を行う
    # - x: 画像データ
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # 0層 -> 1層
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        # 1層 -> 出力層
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 損失関数の値を求める
    # (predict()の結果と正解ラベルを元に交差エントロピー誤差を求める処理)
    # - x: 画像データ(入力データ)
    # - t: 正解ラベル(教師データ)
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 認識精度を求める
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 最も確率の高い要素のインデックスを取得
        t = np.argmax(t, axis=1)

        # ニューラルネットワークが予測した答えと正解ラベルとを比較して、正解した割合を認識精度とする
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 重みパラメータに対する勾配を求める
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # numerical_gradientの高速版
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


# # 手書き数字認識を行う場合は、入力画像サイズが28x28の計784個あり、出力は10個のクラスになる。隠れ層の個数は適当な値。
# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print("***** TwoLayerNet *****")
# print(net.params['W1'].shape) # (784, 100)
# print(net.params['b1'].shape) # (100, )
# print(net.params['W2'].shape) # (100, 10)
# print(net.params['b2'].shape) # (10, )
