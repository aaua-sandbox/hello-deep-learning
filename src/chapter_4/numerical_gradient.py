# 勾配: すべての変数の偏微分をベクトルとしてまとめたもの
import numpy as np

# 勾配の対象とする数式
def function_2(x):
    print("x -> " + str(x))
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
        print("fxh1 -> " + str(fxh1))

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        print("fxh2 -> " + str(fxh2))

        grad[idx] = (fxh1 - fxh2) / (2*h)
        print("fxh1 - fxh2 -> " + str(fxh1 - fxh2))
        print("2*h -> " + str(2*h))
        x[idx] = tmp_val # 値を元に戻す

        print("------------")
    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0]))) # array([6., 8.])
