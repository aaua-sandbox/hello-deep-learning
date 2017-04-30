import numpy as np

# y = { 0 (b + w1x1 + w2x2 <= 0)
#       1 (b + w1x1 + w2x2 >  0)
def AND(x1, x2):
    # 入力
    x = np.array([x1, x2])
    # 重み
    w = np.array([0.5, 0.5])
    # バイアス
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print("AND -----------------")
print("0, 0 -> " + str(AND(0, 0)))
print("0, 1 -> " + str(AND(0, 1)))
print("1, 0 -> " + str(AND(1, 0)))
print("1, 1 -> " + str(AND(1, 1)))

# 重みとバイアスがANDと異なる
def NAND(x1, x2):
    # 入力
    x = np.array([x1, x2])
    # 重み
    w = np.array([-0.5, -0.5])
    # バイアス
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print("NAND ----------------")
print("0, 0 -> " + str(NAND(0, 0)))
print("0, 1 -> " + str(NAND(0, 1)))
print("1, 0 -> " + str(NAND(1, 0)))
print("1, 1 -> " + str(NAND(1, 1)))

# 重みとバイアスがANDと異なる
def OR(x1, x2):
    # 入力
    x = np.array([x1, x2])
    # 重み
    w = np.array([0.5, 0.5])
    # バイアス
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print("OR ------------------")
print("0, 0 -> " + str(OR(0, 0)))
print("0, 1 -> " + str(OR(0, 1)))
print("1, 0 -> " + str(OR(1, 0)))
print("1, 1 -> " + str(OR(1, 1)))

# XORは非線形なので層を重ねて実現する(多層パーセプトロン)
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print("XOR -----------------")
print("0, 0 -> " + str(XOR(0, 0)))
print("0, 1 -> " + str(XOR(0, 1)))
print("1, 0 -> " + str(XOR(1, 0)))
print("1, 1 -> " + str(XOR(1, 1)))
