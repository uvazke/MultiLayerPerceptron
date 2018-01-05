#coding:utf-8
import numpy as np
import sys
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
#-----------------------------------------------------------------------------------------
#各変数の引数に関しての注意
#f,f_diff:1~
#weight:1~
#hiddennum:1~
#delta:1~
#o:0~

#-----------------------------------------------------------------------------------------
#使える活性化関数とその微分形
#Rectified Linear Unit
def ReLU(x):
    return x * (x > 0)
"""
    for i in range (len(x)):
        return x * (x > 0)

        if x[i] >= 0.0:
            x[i] = x[i]
        else: 
            x[i] = 0.0
    return x
"""

def ReLU_diff(x):
    return 1. * (x > 0)
"""
    for i in range (len(x)):
        if x[i] >= 0.0:
            x[i] = 1.0
        else: 
            x[i] = 0.0
    return x
"""
#Sigmoid function
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Sigmoid_diff(x):
    return x * (1 - x)

# hypobaric tangent
def tanh(x):
    return np.tanh(x)

def tanh_diff(x):
    return 1.0 - x ** 2

#softmax
def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp)

#-----------------------------------------------------------------------------------------
class MultiLayerPerceptron:
    def __init__(self, InputNum, HiddenNum, OutputNum, LayerNum, f):
        
        #値の確認
        # print  InputNum, HiddenNum[0], OutputNum, LayerNum, f[0]
        
        #配列の初期化([0]を使わない分一つ多めに配列生成)
        self.f = [0 for i in range(LayerNum)]
        self.f_diff = [0 for i in range(LayerNum)]
        self.hiddennum = [0 for i in range(LayerNum - 1)]
        self.weight = [0 for i in range(LayerNum)]
        self.accuracy = 0.


        # f, HiddenNumに格納する値は、list型
        if not(isinstance(f, list)) or not(isinstance(HiddenNum, list)):
            print "type of HiddenNum and f are both list" 
        # fは(Layerの数　- 1) 個の活性化関数を引数に持つ必要がある。　
        elif LayerNum - 1 != len(f):
            print "please input %d function names to ingredient f and the format is ['a', 'b', ..,'z']" % (LayerNum-1)
        # HiddenNumは(Layerの数　- 2)要素の数列でなければいけない。
        elif LayerNum - 2 != len(HiddenNum):
            print "please input the number of each hiddenlayer's units for %d layers, and the format is [3, 5, 10, ... 3 ]" % (LayerNum-2)
            
        else:
            ####活性化関数を指定:f[1]~f[LayerNum-1]----------------
            for i in range (LayerNum - 1):
                if f[i] == "relu":
                    self.f[i+1] = ReLU
                    self.f_diff[i+1] = ReLU_diff
                elif f[i] == "sigmoid":
                    self.f[i+1] = Sigmoid
                    self.f_diff[i+1] = Sigmoid_diff
                elif f[i] == "tanh":
                    self.f[i+1] = tanh
                    self.f_diff[i+1] = tanh_diff
                elif f[i] == "softmax":
                    self.f[i] = softmax
                else:
                    print "f require 'relu', 'sigmoid' or 'tanh'"

            #入力層と中間層にはbias unit があるので+1
            self.inputnum = InputNum + 1
            #入力層を第０層と呼称した場合、中間層は第１層から第(LayerNum - 2)層まで、そして出力層は第(LayerNum- 1)層なので、第 i 層のunit数は self.hiddennum[i]とした ＊bias unitの分+1している
            for i in range (LayerNum - 2):
                self.hiddennum[i+1] =HiddenNum[i] + 1
            self.outputnum = OutputNum
            self.layernum = LayerNum
            #   print "self.layernum=%d" % self.layernum

        ###重みベクトル------------------------------------------------------------
            #重みベクトルを　from -0.5 to 0.5 の　float型で ランダムに決定する
            #for i in range (LayerNum - 1):
            #第一層のunit数(self.hiddennum[1]) * 入力層のunit数(self.inputnum)  の　matrix を生成。
            #つまり、["現在の層の全ユニットから次の層の第一ユニットへの入力"に対してそれぞれかかる重みのベクトル]が行ベクトルとして上から積み重なっている
            self.weight[1] = np.random.uniform(-0.5 ,0.5 , (self.hiddennum[1], self.inputnum))
          
            #中間層のweight(2~self.hiddennum-2層目まで)
            #  if layernum > 3:
            for k in range(self.layernum-1)[2:self.layernum-1]:
                self.weight[k] = np.random.uniform(-0.5, 0.5 , (self.hiddennum[k], self.hiddennum[k-1]))
            #出力層のunit数(self.outputnum) * 中間層の最後の層のunit数(self.hiddennum[self.layernum-2]) の　matrix　を生成。
            self.weight[self.layernum-1] = np.random.uniform(-1.0, 1.0 , (self.outputnum, self.hiddennum[1]))

    #学習
    def fit(self, X, t, learning_rate = 0.1 , UpdateNum = 10000):
        #入力 X の第一列に１を入れる as bias unit
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        t = np.array(t)
        #Noiseを入れる確率
        self.NoisePercentage = 0.00
        #UpdateNum回の重み更新を行う
        #print "running ..."
        for i in range(UpdateNum):
            iternum = (i+1) % 10000
            """
            if iternum == 0:
                print "Update No.%d" % (i+1)
            """
            #訓練データをランダムに選択し(j番目の訓練データ)、xに格納し、それを入力とする。
            j = np.random.randint(X.shape[0])
            x = X[j]
            #Noiseを入れる. randの値によってNoiseが入るか決まる。
            for k in range(len(x)):
                rand = np.random.rand()
                if rand < self.NoisePercentage:
                    x[k] = np.random.rand()
                
            
            #とりあえず行列であることだけ規定しておく（あとで変更するから）
            o = [[0 for k in range(10)] for l in range(self.layernum)]
            delta = [[[0 for k in range(10)] for l in range(10)] for m in range(self.layernum)]

 
            #入力を中間層を通して伝播させていき、出力を得る。第k層の出力をo[k](self.hiddennum[k] 次元の　vector)とする。o[k] =f[k](Σ(weight[k] * o[k-1]))
            for k in range(self.layernum - 1):
                o[k] = np.zeros(self.hiddennum[k])
            o[self.layernum - 1] = np.zeros(self.outputnum)

            #o[0] が入力、o[LayerNum-1] が出力
            o[0] = x 
            for k in range(self.layernum - 1):
                o[k+1] = self.f[k+1](np.dot(self.weight[k+1], o[k]))
            #出力
            out = o[self.layernum - 1]
            

            ###誤差関数に二乗誤差を使った場合
            """
            #出力層におけるdelta
            delta[self.layernum-1] = self.f_diff[self.layernum-1](out) * (out - t[j])
            """
            ###誤差関数に交差エントロピー誤差を使った場合
            
            #出力層におけるdelta
            delta[self.layernum-1] =  (out - t[j])
            
            #中間層におけるdelta(self.layernum-2~ 1)
            for k in range(self.layernum -1)[:0:-1]:
                delta[k] = self.f_diff[k](o[k]) *  np.dot(self.weight[k+1].T, delta[k+1]) 


            #重み更新
            #行列演算をするために2次元ベクトルに変換
            for k in range(self.layernum - 1):
                o[k] = np.atleast_2d(o[k])
                delta[k+1] = np.atleast_2d(delta[k+1])
                self.weight[k+1] -= learning_rate * np.dot(delta[k+1].T, o[k])
            """
            #毎回テストする場合
            prediction = []
            answer = []
            prediction.append(np.argmax(out))
            answer.append(np.argmax(t[j]))
            if(i > 0):
                self.accuracy += accuracy_score(answer,prediction)
                nowaccuracy = self.accuracy / i
                print str(nowaccuracy)
            """ 
    #学習した重みで実行
    def UseMLP(self, x):
        x = np.array(x)
        
        #add 1 as bias
        x = np.insert(x, 0, 1)

        #とりあえず行列であることだけ規定しておく（あとで変更するから）
        o = [[0 for k in range(10)] for l in range(self.layernum)]

        #出力を計算
        o[0] = x 
        for k in range(self.layernum - 1):
            o[k+1] = self.f[k+1](np.dot(self.weight[k+1], o[k]))

        #出力
        out = o[self.layernum - 1]
        return out
                

if __name__ == "__main__":
    """
    #XORの学習テスト---------------------------------------------------------------------------------
    mlp = MultiLayerPerceptron(2, [2], 1, 3, ["tanh", "sigmoid"])
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    mlp.fit(X, y)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print i, mlp.UseMLP(i)
    #-----------------------------------------------------------------------------------------------
    """
    #手書き数字認識テスト----------------------------------------------------------------------------
    #学習率と更新回数

    learning_rate = 0.001
    updatenumber = 100000

    # MNISTの数字データ
    # 70000サンプル, 28x28ピクセル
    # カレントディレクトリ（.）にmnistデータがない場合は
    # Webから自動的にダウンロードされる（時間がかかる）
    pictures = fetch_mldata('MNIST original', data_home=".")

    
    # 訓練データを作成
    X = pictures.data
    y = pictures.target

    # ピクセルの値を0.0~1.0に正規化
    X = X.astype(np.float64)
    X /= X.max()
    #-0.5 ~ 0.5に
    #X -=0.5
    # 多層パーセプトロンに通す
    #    mlp = MultiLayerPerceptron(28*28, [50], 10, 3, ["tanh","sigmoid"])
    mlp = MultiLayerPerceptron(28*28, [500], 10, 3, ["tanh","sigmoid"])
    #mlp = MultiLayerPerceptron(28*28, [50,50,50,50], 10, 6, ["tanh","tanh","tanh", "tanh","sigmoid"])
    #mlp = MultiLayerPerceptron(28*28, [80, 60, 70, 100, 70, 60, 80 ], 10, 9, ["relu","relu","relu","relu","relu","relu","relu", "sigmoid"])
    # 訓練データ（90%）とテストデータ（10%）に分解
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # 出力としては10元のベクトルを用意して、k個目のunitにはkが対応し、答えのunitに1が入る
    labels_train = LabelBinarizer().fit_transform(y_train)
    #labels_test = LabelBinarizer().fit_transform(y_test)

    # 訓練データを用いてニューラルネットの重みを学習
    mlp.fit(X_train, labels_train, learning_rate, updatenumber)


    # テストデータを用いて予測精度を計算
    predictions = []
    for i in range(X_test.shape[0]):
        o = mlp.UseMLP(X_test[i])
        # 最大の出力を持つクラスに分類
        predictions.append(np.argmax(o))

    # 誤認識したデータのみ描画
    # 誤認識データ数と誤っているテストデータのidxを収集
    """
    cnt = 0
    error_idx = []
    for idx in range(len(y_test)):
        if y_test[idx] != predictions[idx]:
            print "error: %d : %d => %d" % (idx, y_test[idx], predictions[idx])
            error_idx.append(idx)
            cnt += 1
    """   
    #結果の表示
    print confusion_matrix(y_test, predictions)
    print classification_report(y_test, predictions)
    print "activation functions are.. "
    for i in range(len(mlp.f))[1:]: 
        print "in LayerNo.%d: " % (i) + str(mlp.f[i])
    print "The number of update is %d" % updatenumber
    print "learning rate is %f" % learning_rate 
    print "Input Noise are  %f percent" % (mlp.NoisePercentage * 100)
    for i in range(mlp.layernum - 2 ):
        print "The number of unit in No.%d layer is %d" %((i + 1), mlp.hiddennum[i+1]) 
    print "accuracy = " + str(accuracy_score(y_test, predictions))


    """
    # 描画
    import pylab
    for i, idx in enumerate(error_idx):
        pylab.subplot(cnt/5 + 1, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(X_test[idx].reshape((28, 28)), cmap=pylab.cm.gray_r)
        pylab.title('%d : %i => %i' % (idx, y_test[idx], predictions[idx]))
    pylab.show()
    """
    #----------------------------------------------------------------------------------

