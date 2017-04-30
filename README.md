[ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)

# SetUp
## CentOS6.5<br>
<br>
## pyenvのインストール<br>
参考: http://qiita.com/akito1986/items/be5dcd1a502aaf22010b<br>

```sh
yum install gcc bzip2 bzip2-devel openssl openssl-devel readline readline-devel
cd /usr/local/
git clone git://github.com/yyuu/pyenv.git ./pyenv
mkdir -p ./pyenv/versions ./pyenv/shims

cd /usr/local/pyenv/plugins/
git clone git://github.com/yyuu/pyenv-virtualenv.git

echo 'export PYENV_ROOT="/usr/local/pyenv"' | sudo tee -a /etc/profile.d/pyenv.sh
echo 'export PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"' | sudo tee -a /etc/profile.d/pyenv.sh

source /etc/profile.d/pyenv.sh
pyenv --version
```

## ``Python 3.4.1 :: Anaconda 2.1.0``のインストール

```sh
pyenv install -v anaconda3-2.1.0
pyenv global anaconda3-2.1.0
```

matplotlibの読み込みでエラーが出た場合
```python
ImportError: libSM.so.6: cannot open shared object file: No such file or directory
```
```sh
yum install libSM libXrender libXext fontconfig-devel
```
