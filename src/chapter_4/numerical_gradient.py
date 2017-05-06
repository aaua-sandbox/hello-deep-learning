# 勾配: すべての変数の偏微分をベクトルとしてまとめたもの
# 勾配降下法: 勾配法で最小値を探す
import numpy as np
import copy

# 勾配の対象とする数式
def function_2(x):
#    print("x -> " + str(x))
    return x[0]**2 + x[1]**2

# 勾配の計算
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形状の配列を生成

    # 変数ごとに数値微分の計算
    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
#        print("fxh1 -> " + str(fxh1))

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
#        print("fxh2 -> " + str(fxh2))

        grad[idx] = (fxh1 - fxh2) / (2*h)
#        print("fxh1 - fxh2 -> " + str(fxh1 - fxh2))
#        print("2*h -> " + str(2*h))
        x[idx] = tmp_val # 値を元に戻す

#        print("------------")
    return grad

# 勾配のテスト計算
print("----- 勾配のテスト計算 -----")
print(numerical_gradient(function_2, np.array([3.0, 4.0]))) # array([6., 8.])


# 勾配降下法
# - f : 最適化したい関数
# - init_x : 初期値
# - lr : Learning rate 学習率
# - step_num : 勾配法による繰り返しの数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = copy.deepcopy(init_x)
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    print("init_x -> " + str(init_x) + ", lr -> " + str(lr) + ", step_num -> " + str(step_num))
    print("  result : " + str(x))
    return x

print("----- 勾配降下法のテスト計算 -----")
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)

print("----- 学習率が大きすぎる例 -----")
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)

print("----- 学習率が小さすぎる例 -----")
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
