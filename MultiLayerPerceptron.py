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

"""
#結果
avg / total       0.35      0.52      0.40      7000
The number of update is 100000
learning rate is 0.005000
The number of unit in No.1 layer is 901

avg / total       0.19      0.26      0.20      7000
The number of update is 100000
learning rate is 0.001000
The number of unit in No.1 layer is 901

avg / total       0.44      0.55      0.45      7000
The number of update is 100000
learning rate is 0.010000
The number of unit in No.1 layer is 901

             precision    recall  f1-score   support

        0.0       0.73      0.97      0.83       717
        1.0       0.84      0.97      0.90       813
        2.0       0.35      0.93      0.51       678
        3.0       0.43      0.81      0.56       721
        4.0       0.48      0.18      0.27       660
        5.0       0.21      0.05      0.08       627
        6.0       0.00      0.00      0.00       707
        7.0       0.49      0.94      0.64       733
        8.0       0.21      0.00      0.01       701
        9.0       0.17      0.02      0.03       643

avg / total       0.40      0.51      0.40      7000

The number of update is 100000
learning rate is 0.020000
The number of unit in No.1 layer is 901

             precision    recall  f1-score   support

        0.0       0.71      0.96      0.81       648
        1.0       0.88      0.98      0.93       808
        2.0       0.63      0.88      0.74       750
        3.0       0.70      0.87      0.78       762
        4.0       0.73      0.91      0.81       710
        5.0       0.41      0.39      0.40       648
        6.0       0.71      0.08      0.14       670
        7.0       0.81      0.91      0.86       696
        8.0       0.14      0.00      0.00       649
        9.0       0.65      0.86      0.74       659

avg / total       0.65      0.70      0.64      7000

The number of update is 100000
learning rate is 0.015000
The number of unit in No.1 layer is 791

        0.0       0.76      0.95      0.85       714
        1.0       0.83      0.97      0.89       787
        2.0       0.70      0.08      0.15       685
        3.0       0.63      0.88      0.74       685
        4.0       0.72      0.83      0.77       666
        5.0       0.60      0.79      0.68       639
        6.0       0.72      0.92      0.81       678
        7.0       0.83      0.89      0.86       701
        8.0       0.15      0.00      0.01       744
        9.0       0.64      0.82      0.72       701


avg / total       0.66      0.71      0.65      7000

The number of update is 100000
learning rate is 0.010000
The number of unit in No.1 layer is 791

          precision    recall  f1-score   support

        0.0       0.73      0.97      0.83       672
        1.0       0.87      0.98      0.92       805
        2.0       0.74      0.88      0.80       694
        3.0       0.59      0.93      0.72       731
        4.0       0.62      0.94      0.75       630
        5.0       0.58      0.27      0.37       647
        6.0       0.79      0.96      0.86       683
        7.0       0.59      0.95      0.72       697
        8.0       0.60      0.00      0.01       701
        9.0       0.50      0.00      0.00       740

avg / total       0.66      0.69      0.60      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fb1cfa1db90>
in LayerNo.2: <function Sigmoid at 0x7fb1cfa1daa0>
The number of update is 200000
learning rate is 0.010000
The number of unit in No.1 layer is 791


             precision    recall  f1-score   support

        0.0       0.90      0.95      0.92       680
        1.0       0.92      0.97      0.94       784
        2.0       0.74      0.88      0.81       717
        3.0       0.65      0.89      0.75       714
        4.0       0.81      0.90      0.85       657
        5.0       0.61      0.65      0.63       635
        6.0       0.82      0.95      0.88       645
        7.0       0.85      0.89      0.87       756
        8.0       0.00      0.00      0.00       730
        9.0       0.76      0.79      0.77       682

avg / total       0.71      0.78      0.74      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f97cfc8cb90>
in LayerNo.2: <function Sigmoid at 0x7f97cfc8caa0>
The number of update is 100000
learning rate is 0.010000
The number of unit in No.1 layer is 101

[[616   0  11   5   1   4   6   1  15   3]
 [  1 738  10   1   0   6   3   4  13   2]
 [  8  10 620  21  16   5  10  15  19   5]
 [  8   4  21 579   5  31   4  17  20  15]
 [  2   2   5   5 628   4  11   4  10  38]
 [ 20   7  11  32  19 498   8   5  30  18]
 [ 10   2  15   4  11  15 599   0   8   0]
 [  6   9  13   5   9   3   1 675  10  23]
 [  7  15  23  39   7  22   8   4 571  15]
 [  5   3   3  10  29  14   1  21  12 543]]
             precision    recall  f1-score   support

        0.0       0.90      0.93      0.92       662
        1.0       0.93      0.95      0.94       778
        2.0       0.85      0.85      0.85       729
        3.0       0.83      0.82      0.82       704
        4.0       0.87      0.89      0.88       709
        5.0       0.83      0.77      0.80       648
        6.0       0.92      0.90      0.91       664
        7.0       0.90      0.90      0.90       754
        8.0       0.81      0.80      0.80       711
        9.0       0.82      0.85      0.83       641

avg / total       0.87      0.87      0.87      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f7e458f6b90>
in LayerNo.2: <function Sigmoid at 0x7f7e458f6aa0>
The number of update is 100000
learning rate is 0.010000
The number of unit in No.1 layer is 101

             precision    recall  f1-score   support

        0.0       0.90      0.96      0.93       646
        1.0       0.96      0.97      0.97       817
        2.0       0.88      0.86      0.87       731
        3.0       0.86      0.86      0.86       721
        4.0       0.73      0.93      0.82       692
        5.0       0.69      0.79      0.73       640
        6.0       0.92      0.92      0.92       719
        7.0       0.72      0.94      0.81       698
        8.0       0.68      0.86      0.76       640
        9.0       0.00      0.00      0.00       696

avg / total       0.74      0.81      0.77      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fe150216b90>
in LayerNo.2: <function Sigmoid at 0x7fe150216aa0>
The number of update is 100000
learning rate is 0.020000
The number of unit in No.1 layer is 101

            precision    recall  f1-score   support

        0.0       0.46      0.81      0.58       676
        1.0       0.55      0.85      0.66       788
        2.0       0.32      0.20      0.24       688
        3.0       0.25      0.07      0.11       722
        4.0       0.38      0.83      0.52       679
        5.0       0.33      0.16      0.21       611
        6.0       0.46      0.25      0.32       729
        7.0       0.41      0.84      0.55       739
        8.0       0.06      0.01      0.02       701
        9.0       0.15      0.03      0.05       667

avg / total       0.34      0.41      0.33      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7ff527d2bb90>
in LayerNo.2: <function Sigmoid at 0x7ff527d2baa0>
The number of update is 100000
learning rate is 0.001000
The number of unit in No.1 layer is 1001

             precision    recall  f1-score   support

        0.0       0.95      0.95      0.95       719
        1.0       0.95      0.97      0.96       744
        2.0       0.88      0.88      0.88       689
        3.0       0.84      0.87      0.85       708
        4.0       0.87      0.87      0.87       684
        5.0       0.88      0.83      0.85       646
        6.0       0.93      0.92      0.93       698
        7.0       0.90      0.89      0.89       733
        8.0       0.84      0.80      0.82       701
        9.0       0.78      0.83      0.81       678

avg / total       0.88      0.88      0.88      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f50a633fc08>
in LayerNo.2: <function Sigmoid at 0x7f50a633fb18>
The number of update is 100000
learning rate is 0.020000
The number of unit in No.1 layer is 51

           precision    recall  f1-score   support

        0.0       0.05      0.00      0.00       667
        1.0       0.93      0.95      0.94       773
        2.0       0.67      0.86      0.75       699
        3.0       0.72      0.82      0.77       731
        4.0       0.81      0.87      0.84       734
        5.0       0.48      0.66      0.56       637
        6.0       0.79      0.93      0.86       706
        7.0       0.33      0.00      0.00       715
        8.0       0.57      0.81      0.67       645
        9.0       0.50      0.77      0.61       693

avg / total       0.59      0.67      0.61      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f52ea1b5c08>
in LayerNo.2: <function Sigmoid at 0x7f52ea1b5b18>
The number of update is 100000
learning rate is 0.010000
The number of unit in No.1 layer is 51

             precision    recall  f1-score   support

        0.0       0.92      0.96      0.94       717
        1.0       0.97      0.95      0.96       742
        2.0       0.90      0.86      0.88       729
        3.0       0.87      0.89      0.88       696
        4.0       0.89      0.92      0.90       705
        5.0       0.89      0.82      0.85       591
        6.0       0.94      0.91      0.92       701
        7.0       0.90      0.94      0.92       778
        8.0       0.83      0.88      0.85       674
        9.0       0.89      0.87      0.88       667

avg / total       0.90      0.90      0.90      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f0fef2d8c08>
in LayerNo.2: <function Sigmoid at 0x7f0fef2d8b18>
The number of update is 100000
learning rate is 0.030000
The number of unit in No.1 layer is 51

             precision    recall  f1-score   support

        0.0       0.94      0.93      0.94       689
        1.0       0.96      0.97      0.97       810
        2.0       0.89      0.89      0.89       736
        3.0       0.86      0.87      0.87       685
        4.0       0.87      0.93      0.90       646
        5.0       0.88      0.85      0.87       623
        6.0       0.91      0.94      0.92       662
        7.0       0.93      0.91      0.92       745
        8.0       0.88      0.89      0.88       692
        9.0       0.91      0.85      0.87       712

avg / total       0.90      0.90      0.90      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f96fc7ffc08>
in LayerNo.2: <function Sigmoid at 0x7f96fc7ffb18>
The number of update is 100000
learning rate is 0.040000
The number of unit in No.1 layer is 51

             precision    recall  f1-score   support

        0.0       0.95      0.96      0.96       702
        1.0       0.97      0.96      0.96       808
        2.0       0.90      0.91      0.90       698
        3.0       0.90      0.87      0.89       684
        4.0       0.90      0.94      0.92       661
        5.0       0.85      0.90      0.88       614
        6.0       0.94      0.94      0.94       710
        7.0       0.90      0.94      0.92       731
        8.0       0.87      0.89      0.88       708
        9.0       0.95      0.82      0.88       684

avg / total       0.91      0.91      0.91      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f5438e37c08>
in LayerNo.2: <function Sigmoid at 0x7f5438e37b18>
The number of update is 100000
learning rate is 0.040000
The number of unit in No.1 layer is 81

             precision    recall  f1-score   support

        0.0       0.94      0.97      0.96       708
        1.0       0.00      0.00      0.00       773
        2.0       0.85      0.89      0.87       697
        3.0       0.50      0.92      0.65       696
        4.0       0.00      0.00      0.00       666
        5.0       0.75      0.90      0.82       595
        6.0       0.81      0.95      0.87       692
        7.0       0.89      0.93      0.91       786
        8.0       0.63      0.87      0.73       676
        9.0       0.65      0.89      0.75       711

avg / total       0.60      0.73      0.65      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f9a08c4bc08>
in LayerNo.2: <function Sigmoid at 0x7f9a08c4bb18>
The number of update is 100000
learning rate is 0.040000
The number of unit in No.1 layer is 101

             precision    recall  f1-score   support

        0.0       0.91      0.97      0.94       681
        1.0       0.95      0.98      0.96       759
        2.0       0.95      0.87      0.91       716
        3.0       0.89      0.89      0.89       717
        4.0       0.88      0.91      0.90       674
        5.0       0.92      0.80      0.86       652
        6.0       0.93      0.95      0.94       683
        7.0       0.85      0.94      0.89       714
        8.0       0.86      0.89      0.88       719
        9.0       0.89      0.82      0.85       685

avg / total       0.90      0.90      0.90      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f3405b98b90>
in LayerNo.2: <function Sigmoid at 0x7f3405b98aa0>
The number of update is 100000
learning rate is 0.040000
Input Noise are  0.010000 percent
The number of unit in No.1 layer is 81

             precision    recall  f1-score   support

        0.0       0.54      0.92      0.68       696
        1.0       0.65      0.97      0.78       771
        2.0       0.00      0.00      0.00       738
        3.0       0.67      0.09      0.16       682
        4.0       0.30      0.92      0.45       693
        5.0       0.29      0.08      0.13       610
        6.0       0.54      0.95      0.69       697
        7.0       0.66      0.93      0.77       719
        8.0       0.00      0.00      0.00       668
        9.0       0.09      0.00      0.01       726

avg / total       0.38      0.50      0.37      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f3be7facb90>
in LayerNo.2: <function Sigmoid at 0x7f3be7facaa0>
The number of update is 100000
learning rate is 0.040000
Input Noise are  0.250000 percent
The number of unit in No.1 layer is 81

             precision    recall  f1-score   support

        0.0       0.00      0.00      0.00       667
        1.0       0.94      0.95      0.95       764
        2.0       0.00      0.00      0.00       685
        3.0       0.88      0.28      0.43       718
        4.0       0.90      0.78      0.84       666
        5.0       0.30      0.57      0.39       646
        6.0       0.73      0.88      0.80       662
        7.0       0.82      0.86      0.84       801
        8.0       0.29      0.80      0.43       696
        9.0       0.69      0.65      0.67       695

avg / total       0.57      0.59      0.54      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f8a6747ab90>
in LayerNo.2: <function tanh at 0x7f8a6747ab90>
in LayerNo.3: <function tanh at 0x7f8a6747ab90>
in LayerNo.4: <function tanh at 0x7f8a6747ab90>
in LayerNo.5: <function tanh at 0x7f8a6747ab90>
in LayerNo.6: <function tanh at 0x7f8a6747ab90>
in LayerNo.7: <function tanh at 0x7f8a6747ab90>
in LayerNo.8: <function Sigmoid at 0x7f8a6747aaa0>
The number of update is 100000
learning rate is 0.050000
Input Noise are  0.010000 percent
The number of unit in No.1 layer is 81
The number of unit in No.2 layer is 61
The number of unit in No.3 layer is 41
The number of unit in No.4 layer is 21
The number of unit in No.5 layer is 41
The number of unit in No.6 layer is 61
The number of unit in No.7 layer is 81

            precision    recall  f1-score   support

        0.0       0.85      0.92      0.88       678
        1.0       0.95      0.87      0.90       799
        2.0       0.64      0.76      0.69       676
        3.0       0.00      0.00      0.00       668
        4.0       0.59      0.39      0.47       715
        5.0       0.66      0.67      0.66       663
        6.0       0.55      0.96      0.70       676
        7.0       0.86      0.68      0.76       718
        8.0       0.35      0.21      0.26       678
        9.0       0.39      0.75      0.51       729

avg / total       0.59      0.63      0.59      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f1111798b90>
in LayerNo.2: <function tanh at 0x7f1111798b90>
in LayerNo.3: <function tanh at 0x7f1111798b90>
in LayerNo.4: <function tanh at 0x7f1111798b90>
in LayerNo.5: <function tanh at 0x7f1111798b90>
in LayerNo.6: <function tanh at 0x7f1111798b90>
in LayerNo.7: <function tanh at 0x7f1111798b90>
in LayerNo.8: <function Sigmoid at 0x7f1111798aa0>
The number of update is 100000
learning rate is 0.050000
Input Noise are  0.010000 percent
The number of unit in No.1 layer is 81
The number of unit in No.2 layer is 61
The number of unit in No.3 layer is 41
The number of unit in No.4 layer is 21
The number of unit in No.5 layer is 41
The number of unit in No.6 layer is 61
The number of unit in No.7 layer is 81

             precision    recall  f1-score   support

        0.0       0.12      0.01      0.01       725
        1.0       0.11      1.00      0.20       782
        2.0       0.00      0.00      0.00       679
        3.0       0.00      0.00      0.00       687
        4.0       0.00      0.00      0.00       682
        5.0       0.00      0.00      0.00       621
        6.0       0.00      0.00      0.00       706
        7.0       0.00      0.00      0.00       752
        8.0       0.00      0.00      0.00       657
        9.0       0.00      0.00      0.00       709

avg / total       0.03      0.11      0.02      7000

activation functions are.. 
in LayerNo.1: <function ReLU at 0x7fee6d2aaa28>
in LayerNo.2: <function ReLU at 0x7fee6d2aaa28>
in LayerNo.3: <function ReLU at 0x7fee6d2aaa28>
in LayerNo.4: <function ReLU at 0x7fee6d2aaa28>
in LayerNo.5: <function ReLU at 0x7fee6d2aaa28>
in LayerNo.6: <function ReLU at 0x7fee6d2aaa28>
in LayerNo.7: <function ReLU at 0x7fee6d2aaa28>
in LayerNo.8: <function Sigmoid at 0x7fee6d2aab18>
The number of update is 100000
learning rate is 0.050000
Input Noise are  0.010000 percent
The number of unit in No.1 layer is 81
The number of unit in No.2 layer is 61
The number of unit in No.3 layer is 71
The number of unit in No.4 layer is 101
The number of unit in No.5 layer is 71
The number of unit in No.6 layer is 61
The number of unit in No.7 layer is 81

             precision    recall  f1-score   support

        0.0       0.95      0.96      0.95       679
        1.0       0.96      0.98      0.97       749
        2.0       0.91      0.90      0.91       708
        3.0       0.91      0.91      0.91       727
        4.0       0.93      0.89      0.91       658
        5.0       0.94      0.84      0.89       625
        6.0       0.92      0.97      0.94       748
        7.0       0.95      0.91      0.93       731
        8.0       0.87      0.92      0.89       700
        9.0       0.86      0.91      0.89       675

avg / total       0.92      0.92      0.92      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f453dd5ac08>
in LayerNo.2: <function Sigmoid at 0x7f453dd5ab18>
The number of update is 100000
learning rate is 0.050000
Input Noise are  0.010000 percent
The number of unit in No.1 layer is 81

             precision    recall  f1-score   support

        0.0       0.91      0.96      0.93       678
        1.0       0.92      0.97      0.94       780
        2.0       0.92      0.89      0.90       712
        3.0       0.85      0.91      0.88       725
        4.0       0.70      0.96      0.81       708
        5.0       0.81      0.87      0.84       654
        6.0       0.91      0.94      0.93       695
        7.0       0.77      0.94      0.85       736
        8.0       0.75      0.87      0.80       623
        9.0       0.00      0.00      0.00       689

avg / total       0.76      0.83      0.79      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f810ec00c08>
in LayerNo.2: <function Sigmoid at 0x7f810ec00b18>
The number of update is 100000
learning rate is 0.050000
Input Noise are  0.010000 percent
The number of unit in No.1 layer is 81

             precision    recall  f1-score   support

        0.0       0.97      0.96      0.97       741
        1.0       0.97      0.96      0.97       795
        2.0       0.95      0.91      0.93       717
        3.0       0.88      0.92      0.90       723
        4.0       0.92      0.94      0.93       698
        5.0       0.91      0.87      0.89       642
        6.0       0.94      0.95      0.95       662
        7.0       0.94      0.94      0.94       753
        8.0       0.87      0.90      0.88       647
        9.0       0.90      0.89      0.90       622

avg / total       0.93      0.93      0.93      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f07de716c08>
in LayerNo.2: <function Sigmoid at 0x7f07de716b18>
The number of update is 100000
learning rate is 0.100000
Input Noise are  0.010000 percent
The number of unit in No.1 layer is 71

            precision    recall  f1-score   support

        0.0       0.93      0.97      0.95       703
        1.0       0.95      0.98      0.96       766
        2.0       0.91      0.89      0.90       712
        3.0       0.94      0.87      0.90       738
        4.0       0.89      0.94      0.92       661
        5.0       0.89      0.84      0.86       587
        6.0       0.92      0.94      0.93       696
        7.0       0.89      0.94      0.92       690
        8.0       0.91      0.86      0.89       689
        9.0       0.89      0.88      0.88       758

avg / total       0.91      0.91      0.91      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fc105f34c08>
in LayerNo.2: <function Sigmoid at 0x7fc105f34b18>
The number of update is 100000
learning rate is 0.100000
Input Noise are  0.050000 percent
The number of unit in No.1 layer is 71

             precision    recall  f1-score   support

        0.0       0.91      0.97      0.94       684
        1.0       0.94      0.97      0.95       842
        2.0       0.88      0.90      0.89       682
        3.0       0.90      0.89      0.90       749
        4.0       0.91      0.94      0.92       663
        5.0       0.88      0.86      0.87       659
        6.0       0.95      0.93      0.94       672
        7.0       0.94      0.92      0.93       699
        8.0       0.89      0.85      0.87       673
        9.0       0.91      0.86      0.89       677

avg / total       0.91      0.91      0.91      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fefe769dc08>
in LayerNo.2: <function Sigmoid at 0x7fefe769db18>
The number of update is 100000
learning rate is 0.100000
Input Noise are  0.050000 percent
The number of unit in No.1 layer is 51

             precision    recall  f1-score   support

        0.0       0.93      0.96      0.94       722
        1.0       0.95      0.97      0.96       786
        2.0       0.89      0.91      0.90       681
        3.0       0.94      0.85      0.89       709
        4.0       0.90      0.92      0.91       695
        5.0       0.87      0.90      0.88       641
        6.0       0.95      0.94      0.94       714
        7.0       0.96      0.90      0.93       729
        8.0       0.88      0.89      0.88       664
        9.0       0.86      0.89      0.88       659

avg / total       0.92      0.91      0.91      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f5f41661c08>
in LayerNo.2: <function Sigmoid at 0x7f5f41661b18>
The number of update is 100000
learning rate is 0.160000
Input Noise are  0.050000 percent
The number of unit in No.1 layer is 51

             precision    recall  f1-score   support

        0.0       0.93      0.97      0.95       715
        1.0       0.92      0.99      0.95       825
        2.0       0.91      0.92      0.91       711
        3.0       0.92      0.86      0.89       732
        4.0       0.91      0.90      0.91       666
        5.0       0.88      0.89      0.88       607
        6.0       0.94      0.94      0.94       687
        7.0       0.93      0.91      0.92       702
        8.0       0.93      0.82      0.87       696
        9.0       0.83      0.91      0.87       659

avg / total       0.91      0.91      0.91      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fd07ff36c08>
in LayerNo.2: <function Sigmoid at 0x7fd07ff36b18>
The number of update is 1000000
learning rate is 0.080000
Input Noise are  0.050000 percent
The number of unit in No.1 layer is 51

the followings are generating weight between -0.5 ~ 0.5 
--------------------------------------------------------------------
             precision    recall  f1-score   support

        0.0       0.91      0.98      0.94       699
        1.0       0.96      0.99      0.98       718
        2.0       0.95      0.89      0.92       719
        3.0       0.95      0.90      0.92       732
        4.0       0.90      0.96      0.93       672
        5.0       0.96      0.86      0.91       659
        6.0       0.94      0.97      0.96       679
        7.0       0.93      0.93      0.93       730
        8.0       0.91      0.92      0.92       694
        9.0       0.90      0.91      0.91       698

avg / total       0.93      0.93      0.93      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f7325be0c08>
in LayerNo.2: <function Sigmoid at 0x7f7325be0b18>
The number of update is 100000
learning rate is 0.100000
Input Noise are  0.050000 percent
The number of unit in No.1 layer is 51

            precision    recall  f1-score   support

        0.0       0.97      0.96      0.97       686
        1.0       0.97      0.97      0.97       766
        2.0       0.92      0.95      0.94       712
        3.0       0.93      0.89      0.91       726
        4.0       0.92      0.94      0.93       725
        5.0       0.94      0.88      0.91       634
        6.0       0.93      0.96      0.94       655
        7.0       0.93      0.94      0.94       689
        8.0       0.91      0.93      0.92       684
        9.0       0.93      0.89      0.91       723

avg / total       0.93      0.93      0.93      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fe2da8c6c08>
in LayerNo.2: <function Sigmoid at 0x7fe2da8c6b18>
The number of update is 100000
learning rate is 0.070000

             precision    recall  f1-score   support

        0.0       0.96      0.97      0.96       668
        1.0       0.92      0.98      0.95       772
        2.0       0.90      0.94      0.92       702
        3.0       0.93      0.88      0.91       713
        4.0       0.95      0.93      0.94       683
        5.0       0.88      0.95      0.91       649
        6.0       0.95      0.96      0.96       697
        7.0       0.92      0.93      0.93       764
        8.0       0.96      0.83      0.89       660
        9.0       0.92      0.89      0.90       692

avg / total       0.93      0.93      0.93      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f574af9bc08>
in LayerNo.2: <function Sigmoid at 0x7f574af9bb18>
The number of update is 1000000
learning rate is 0.070000
Input Noise are  0.050000 percent
The number of unit in No.1 layer is 81

            precision    recall  f1-score   support

        0.0       0.93      0.96      0.94       722
        1.0       0.91      0.98      0.94       780
        2.0       0.90      0.92      0.91       709
        3.0       0.93      0.88      0.90       722
        4.0       0.94      0.90      0.92       665
        5.0       0.87      0.86      0.86       636
        6.0       0.89      0.95      0.92       676
        7.0       0.91      0.94      0.92       733
        8.0       0.92      0.85      0.88       665
        9.0       0.92      0.88      0.90       692

avg / total       0.91      0.91      0.91      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fac3ede5c08>
in LayerNo.2: <function Sigmoid at 0x7fac3ede5b18>
The number of update is 1000000
learning rate is 0.010000
Input Noise are  0.050000 percent
The number of unit in No.1 layer is 121

こっからしたはnoiseの％をなおした。
-------------------------------------------------------

             precision    recall  f1-score   support

        0.0       0.75      0.96      0.84       689
        1.0       0.70      0.99      0.82       789
        2.0       0.16      0.01      0.02       694
        3.0       1.00      0.00      0.00       699
        4.0       0.65      0.90      0.75       698
        5.0       0.51      0.86      0.64       643
        6.0       0.00      0.00      0.00       684
        7.0       1.00      0.00      0.01       734
        8.0       0.33      0.86      0.48       651
        9.0       0.54      0.90      0.68       719

avg / total       0.57      0.55      0.43      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f14ab084c08>
in LayerNo.2: <function Sigmoid at 0x7f14ab084b18>
The number of update is 100000
learning rate is 0.010000
Input Noise are  5.000000 percent
The number of unit in No.1 layer is 121

             precision    recall  f1-score   support

        0.0       0.83      0.96      0.89       707
        1.0       0.79      0.99      0.88       810
        2.0       0.67      0.00      0.01       673
        3.0       0.72      0.86      0.79       754
        4.0       0.80      0.89      0.84       685
        5.0       0.65      0.81      0.72       628
        6.0       0.65      0.95      0.77       692
        7.0       0.85      0.93      0.89       744
        8.0       0.00      0.00      0.00       660
        9.0       0.61      0.87      0.72       647

avg / total       0.67      0.74      0.66      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f5833c7ec08>
in LayerNo.2: <function Sigmoid at 0x7f5833c7eb18>
The number of update is 100000
learning rate is 0.008000
Input Noise are  5.000000 percent
The number of unit in No.1 layer is 121

             precision    recall  f1-score   support

        0.0       0.81      0.96      0.88       678
        1.0       0.86      0.98      0.92       839
        2.0       0.00      0.00      0.00       752
        3.0       0.00      0.00      0.00       669
        4.0       0.83      0.95      0.89       681
        5.0       0.60      0.89      0.71       614
        6.0       0.72      0.96      0.82       654
        7.0       0.83      0.93      0.88       744
        8.0       0.56      0.87      0.68       695
        9.0       0.79      0.89      0.84       674

avg / total       0.60      0.74      0.66      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7eff90b6dc08>
in LayerNo.2: <function Sigmoid at 0x7eff90b6db18>
The number of update is 200000
learning rate is 0.010000
Input Noise are  5.000000 percent
The number of unit in No.1 layer is 121

             precision    recall  f1-score   support

        0.0       0.86      0.97      0.91       698
        1.0       0.91      0.98      0.94       778
        2.0       0.79      0.88      0.83       703
        3.0       0.00      0.00      0.00       753
        4.0       0.86      0.92      0.89       739
        5.0       0.63      0.81      0.71       616
        6.0       0.88      0.95      0.91       636
        7.0       0.84      0.91      0.88       729
        8.0       0.65      0.86      0.74       658
        9.0       0.85      0.84      0.84       690

avg / total       0.73      0.81      0.76      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f8c917c4c08>
in LayerNo.2: <function Sigmoid at 0x7f8c917c4b18>
The number of update is 100000
learning rate is 0.010000
Input Noise are  5.000000 percent
The number of unit in No.1 layer is 71

            precision    recall  f1-score   support

        0.0       0.83      0.98      0.90       695
        1.0       0.87      0.98      0.92       782
        2.0       0.59      0.91      0.71       643
        3.0       0.74      0.91      0.82       728
        4.0       0.78      0.93      0.85       642
        5.0       0.57      0.85      0.68       653
        6.0       0.00      0.00      0.00       715
        7.0       0.91      0.91      0.91       759
        8.0       0.00      0.00      0.00       681
        9.0       0.69      0.90      0.78       702

avg / total       0.60      0.74      0.66      7000



activation functions are.. 
in LayerNo.1: <function tanh at 0x7f5de29f7c08>
in LayerNo.2: <function Sigmoid at 0x7f5de29f7b18>
The number of update is 300000
learning rate is 0.010000
Input Noise are  5.000000 percent
The number of unit in No.1 layer is 71


上のように0があるときは過学習っぽい
noise増やすと良くなった。
            precision    recall  f1-score   support

        0.0       0.94      0.97      0.96       691
        1.0       0.93      0.98      0.95       839
        2.0       0.90      0.86      0.88       695
        3.0       0.88      0.86      0.87       721
        4.0       0.89      0.92      0.91       725
        5.0       0.89      0.80      0.84       627
        6.0       0.93      0.96      0.94       697
        7.0       0.93      0.89      0.91       705
        8.0       0.85      0.86      0.85       665
        9.0       0.84      0.87      0.86       635

avg / total       0.90      0.90      0.90      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fe0d0957c08>
in LayerNo.2: <function Sigmoid at 0x7fe0d0957b18>
The number of update is 300000
learning rate is 0.010000
Input Noise are  10.000000 percent
The number of unit in No.1 layer is 71

             precision    recall  f1-score   support

        0.0       0.81      0.95      0.88       682
        1.0       0.79      0.99      0.88       737
        2.0       0.77      0.74      0.75       727
        3.0       0.67      0.83      0.74       729
        4.0       0.50      0.00      0.00       712
        5.0       0.00      0.00      0.00       615
        6.0       0.77      0.93      0.84       645
        7.0       0.88      0.84      0.86       768
        8.0       0.56      0.78      0.65       669
        9.0       0.49      0.83      0.61       716

avg / total       0.63      0.70      0.63      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fe17a1aec08>
in LayerNo.2: <function Sigmoid at 0x7fe17a1aeb18>
The number of update is 300000
learning rate is 0.010000
Input Noise are  25.000000 percent
The number of unit in No.1 layer is 71

             precision    recall  f1-score   support

        0.0       0.94      0.97      0.95       686
        1.0       0.84      0.98      0.90       725
        2.0       0.85      0.85      0.85       753
        3.0       0.76      0.87      0.81       718
        4.0       0.86      0.87      0.86       709
        5.0       0.64      0.74      0.69       624
        6.0       0.84      0.95      0.89       686
        7.0       0.89      0.91      0.90       725
        8.0       0.00      0.00      0.00       679
        9.0       0.66      0.88      0.76       695

avg / total       0.73      0.81      0.77      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f1ef5615c08>
in LayerNo.2: <function Sigmoid at 0x7f1ef5615b18>
The number of update is 300000
learning rate is 0.010000
Input Noise are  15.000000 percent
The number of unit in No.1 layer is 71

             precision    recall  f1-score   support

        0.0       0.83      0.96      0.89       693
        1.0       0.87      0.97      0.92       793
        2.0       0.84      0.84      0.84       689
        3.0       0.81      0.84      0.82       742
        4.0       0.80      0.77      0.78       676
        5.0       0.92      0.65      0.76       637
        6.0       0.88      0.92      0.90       676
        7.0       0.85      0.87      0.86       763
        8.0       0.81      0.78      0.79       648
        9.0       0.75      0.72      0.73       683

avg / total       0.84      0.83      0.83      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fc8c54acc08>
in LayerNo.2: <function Sigmoid at 0x7fc8c54acb18>
The number of update is 500000
learning rate is 0.005000
Input Noise are  15.000000 percent
The number of unit in No.1 layer is 71

             precision    recall  f1-score   support

        0.0       0.76      0.96      0.85       720
        1.0       0.75      0.99      0.85       733
        2.0       0.81      0.30      0.43       714
        3.0       0.69      0.89      0.78       697
        4.0       0.61      0.91      0.73       682
        5.0       0.91      0.46      0.61       653
        6.0       0.83      0.87      0.85       647
        7.0       0.80      0.89      0.84       783
        8.0       0.70      0.71      0.70       681
        9.0       0.87      0.50      0.64       690

avg / total       0.77      0.75      0.73      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f41f424cc08>
in LayerNo.2: <function Sigmoid at 0x7f41f424cb18>
The number of update is 1000000
learning rate is 0.005000
Input Noise are  15.000000 percent
The number of unit in No.1 layer is 71

             precision    recall  f1-score   support

        0.0       0.00      0.00      0.00       678
        1.0       0.78      0.99      0.87       772
        2.0       0.71      0.79      0.74       693
        3.0       0.77      0.84      0.80       735
        4.0       0.57      0.87      0.69       731
        5.0       0.64      0.73      0.68       634
        6.0       0.00      0.00      0.00       678
        7.0       0.83      0.90      0.86       716
        8.0       0.40      0.68      0.51       675
        9.0       0.78      0.80      0.79       688

avg / total       0.55      0.67      0.60      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fa41d9b4c08>
in LayerNo.2: <function Sigmoid at 0x7fa41d9b4b18>
The number of update is 1000000
learning rate is 0.005000
Input Noise are  10.000000 percent
The number of unit in No.1 layer is 71

            precision    recall  f1-score   support

        0.0       0.91      0.97      0.94       686
        1.0       0.92      0.98      0.95       768
        2.0       0.87      0.90      0.88       734
        3.0       0.92      0.85      0.88       710
        4.0       0.86      0.91      0.88       664
        5.0       0.87      0.90      0.89       656
        6.0       0.96      0.90      0.93       656
        7.0       0.95      0.90      0.93       700
        8.0       0.89      0.81      0.85       715
        9.0       0.87      0.88      0.87       711

avg / total       0.90      0.90      0.90      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fe961f79c08>
in LayerNo.2: <function Sigmoid at 0x7fe961f79b18>
The number of update is 300000
learning rate is 0.070000
Input Noise are  15.000000 percent
The number of unit in No.1 layer is 51

            precision    recall  f1-score   support

        0.0       0.93      0.95      0.94       686
        1.0       0.90      0.98      0.94       777
        2.0       0.92      0.84      0.88       699
        3.0       0.91      0.83      0.87       678
        4.0       0.90      0.88      0.89       678
        5.0       0.82      0.86      0.84       625
        6.0       0.89      0.95      0.92       694
        7.0       0.90      0.95      0.92       746
        8.0       0.92      0.78      0.84       695
        9.0       0.84      0.88      0.86       722

avg / total       0.89      0.89      0.89      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f4968d85c08>
in LayerNo.2: <function Sigmoid at 0x7f4968d85b18>
The number of update is 300000
learning rate is 0.050000
Input Noise are  15.000000 percent
The number of unit in No.1 layer is 51
             precision    recall  f1-score   support

        0.0       0.93      0.97      0.95       716
        1.0       0.90      0.99      0.94       761
        2.0       0.92      0.84      0.88       736
        3.0       0.85      0.92      0.88       698
        4.0       0.87      0.90      0.89       654
        5.0       0.90      0.82      0.86       626
        6.0       0.94      0.92      0.93       698
        7.0       0.91      0.94      0.93       745
        8.0       0.90      0.83      0.86       661
        9.0       0.89      0.85      0.87       705

avg / total       0.90      0.90      0.90      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fb7f34a1c08>
in LayerNo.2: <function Sigmoid at 0x7fb7f34a1b18>
The number of update is 300000
learning rate is 0.090000
Input Noise are  15.000000 percent
The number of unit in No.1 layer is 51

           precision    recall  f1-score   support

        0.0       0.00      0.00      0.00       678
        1.0       0.57      0.99      0.72       769
        2.0       1.00      0.00      0.00       730
        3.0       0.74      0.87      0.80       700
        4.0       0.00      0.00      0.00       672
        5.0       0.20      0.89      0.33       634
        6.0       0.00      0.00      0.00       695
        7.0       0.34      0.96      0.50       710
        8.0       0.16      0.00      0.01       668
        9.0       0.00      0.00      0.00       744

avg / total       0.31      0.37      0.24      7000

activation functions are.. 
in LayerNo.1: <function ReLU at 0x7f145c130aa0>
in LayerNo.2: <function ReLU at 0x7f145c130aa0>
in LayerNo.3: <function Sigmoid at 0x7f145c130b90>
The number of update is 1000000
learning rate is 0.003000
Input Noise are  0.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51

             precision    recall  f1-score   support

        0.0       0.97      0.97      0.97       701
        1.0       0.98      0.97      0.98       789
        2.0       0.95      0.93      0.94       695
        3.0       0.90      0.94      0.92       690
        4.0       0.95      0.96      0.95       711
        5.0       0.92      0.91      0.92       643
        6.0       0.95      0.96      0.96       703
        7.0       0.96      0.94      0.95       703
        8.0       0.89      0.91      0.90       661
        9.0       0.92      0.91      0.92       704

avg / total       0.94      0.94      0.94      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f78e6b11c80>
in LayerNo.2: <function tanh at 0x7f78e6b11c80>
in LayerNo.3: <function Sigmoid at 0x7f78e6b11b90>
The number of update is 1000000
learning rate is 0.003000
Input Noise are  0.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51

             precision    recall  f1-score   support

        0.0       0.97      0.98      0.98       670
        1.0       0.98      0.98      0.98       821
        2.0       0.95      0.94      0.95       687
        3.0       0.95      0.93      0.94       709
        4.0       0.95      0.95      0.95       649
        5.0       0.95      0.95      0.95       644
        6.0       0.96      0.97      0.97       695
        7.0       0.95      0.96      0.96       712
        8.0       0.94      0.94      0.94       702
        9.0       0.94      0.93      0.93       711

avg / total       0.95      0.95      0.95      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f19870d2c80>
in LayerNo.2: <function tanh at 0x7f19870d2c80>
in LayerNo.3: <function tanh at 0x7f19870d2c80>
in LayerNo.4: <function Sigmoid at 0x7f19870d2b90>
The number of update is 10000000
learning rate is 0.001000
Input Noise are  0.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51
The number of unit in No.3 layer is 51

             precision    recall  f1-score   support

        0.0       0.44      0.92      0.60       674
        1.0       0.27      0.99      0.43       794
        2.0       0.53      0.26      0.34       743
        3.0       0.34      0.08      0.13       730
        4.0       0.45      0.31      0.36       688
        5.0       0.67      0.00      0.01       653
        6.0       0.59      0.36      0.45       676
        7.0       0.49      0.38      0.43       732
        8.0       0.28      0.04      0.06       647
        9.0       0.32      0.31      0.31       663

avg / total       0.44      0.37      0.31      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f3eafe8fc08>
in LayerNo.2: <function tanh at 0x7f3eafe8fc08>
in LayerNo.3: <function tanh at 0x7f3eafe8fc08>
in LayerNo.4: <function Sigmoid at 0x7f3eafe8fb18>
The number of update is 10000000
learning rate is 0.001000
Input Noise are  25.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51
The number of unit in No.3 layer is 51

             precision    recall  f1-score   support

        0.0       0.64      0.72      0.68       694
        1.0       0.51      0.97      0.66       748
        2.0       0.41      0.19      0.26       672
        3.0       0.51      0.50      0.50       707
        4.0       0.46      0.71      0.55       709
        5.0       0.43      0.00      0.01       615
        6.0       0.44      0.87      0.58       702
        7.0       0.61      0.73      0.66       739
        8.0       0.19      0.03      0.05       716
        9.0       0.21      0.09      0.13       698

avg / total       0.44      0.49      0.42      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f8a43123c08>
in LayerNo.2: <function tanh at 0x7f8a43123c08>
in LayerNo.3: <function tanh at 0x7f8a43123c08>
in LayerNo.4: <function Sigmoid at 0x7f8a43123b18>
The number of update is 100000
learning rate is 0.001000
Input Noise are  25.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51
The number of unit in No.3 layer is 51

             precision    recall  f1-score   support

        0.0       0.74      0.88      0.80       679
        1.0       0.91      0.96      0.94       816
        2.0       0.65      0.82      0.73       668
        3.0       0.70      0.76      0.73       721
        4.0       0.73      0.85      0.79       673
        5.0       0.46      0.49      0.48       569
        6.0       0.00      0.00      0.00       706
        7.0       0.87      0.85      0.86       749
        8.0       0.53      0.62      0.57       676
        9.0       0.69      0.74      0.71       743

avg / total       0.64      0.70      0.67      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f1fb0115c08>
in LayerNo.2: <function tanh at 0x7f1fb0115c08>
in LayerNo.3: <function tanh at 0x7f1fb0115c08>
in LayerNo.4: <function Sigmoid at 0x7f1fb0115b18>
The number of update is 100000
learning rate is 0.001000
Input Noise are  0.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51
The number of unit in No.3 layer is 51


            precision    recall  f1-score   support

        0.0       0.93      0.97      0.95       665
        1.0       0.98      0.96      0.97       798
        2.0       0.93      0.89      0.91       747
        3.0       0.92      0.88      0.90       712
        4.0       0.92      0.94      0.93       652
        5.0       0.91      0.89      0.90       648
        6.0       0.94      0.95      0.95       680
        7.0       0.92      0.94      0.93       702
        8.0       0.88      0.89      0.88       686
        9.0       0.88      0.92      0.90       710

avg / total       0.92      0.92      0.92      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f6771a5dc08>
in LayerNo.2: <function tanh at 0x7f6771a5dc08>
in LayerNo.3: <function tanh at 0x7f6771a5dc08>
in LayerNo.4: <function Sigmoid at 0x7f6771a5db18>
The number of update is 100000
learning rate is 0.010000
Input Noise are  0.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51
The number of unit in No.3 layer is 51
             precision    recall  f1-score   support

        0.0       0.97      0.98      0.97       692
        1.0       0.97      0.97      0.97       794
        2.0       0.92      0.94      0.93       716
        3.0       0.92      0.91      0.92       723
        4.0       0.95      0.94      0.94       681
        5.0       0.94      0.86      0.90       630
        6.0       0.94      0.96      0.95       667
        7.0       0.93      0.96      0.94       745
        8.0       0.89      0.91      0.90       666
        9.0       0.92      0.92      0.92       686

avg / total       0.94      0.94      0.94      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7ff296117c08>
in LayerNo.2: <function tanh at 0x7ff296117c08>
in LayerNo.3: <function tanh at 0x7ff296117c08>
in LayerNo.4: <function Sigmoid at 0x7ff296117b18>
The number of update is 100000
learning rate is 0.020000
Input Noise are  0.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51
The number of unit in No.3 layer is 51

             precision    recall  f1-score   support

        0.0       0.96      0.97      0.96       715
        1.0       0.97      0.97      0.97       794
        2.0       0.93      0.93      0.93       680
        3.0       0.97      0.90      0.93       730
        4.0       0.94      0.95      0.94       668
        5.0       0.90      0.94      0.92       637
        6.0       0.96      0.97      0.96       718
        7.0       0.94      0.95      0.95       732
        8.0       0.92      0.94      0.93       670
        9.0       0.93      0.91      0.92       656

avg / total       0.94      0.94      0.94      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f1648dc6c08>
in LayerNo.2: <function tanh at 0x7f1648dc6c08>
in LayerNo.3: <function tanh at 0x7f1648dc6c08>
in LayerNo.4: <function Sigmoid at 0x7f1648dc6b18>
The number of update is 100000
learning rate is 0.030000
Input Noise are  0.000000 percent
The number of unit in No.1 layer is 51
The number of unit in No.2 layer is 51
The number of unit in No.3 layer is 51

             precision    recall  f1-score   support

        0.0       0.15      0.75      0.25       683
        1.0       0.11      0.00      0.01       828
        2.0       0.14      0.11      0.12       660
        3.0       0.03      0.00      0.00       706
        4.0       0.22      0.35      0.27       684
        5.0       0.11      0.05      0.07       603
        6.0       0.01      0.00      0.01       702
        7.0       0.03      0.00      0.00       715
        8.0       0.05      0.02      0.03       705
        9.0       0.10      0.15      0.12       714

avg / total       0.09      0.14      0.09      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fce5d67eb90>
in LayerNo.2: <function Sigmoid at 0x7fce5d67eaa0>
The number of update is 50
learning rate is 0.010000
Input Noise are  5.000000 percent
The number of unit in No.1 layer is 501

上二乗誤差下交差エントロピー誤差
[[334  29   0  57   0   4  77  29   1 214]
 [  0 621   0  12   0   0   2   1   0 153]
 [ 11 102  92  54   0   0 249  47   0 123]
 [  5 355   3 241   1   1  23  16   1  89]
 [  1  36   2   2   4   0 137  37   0 420]
 [  8 245   0  85   0   1  30  16   1 224]
 [  0  21   0   4   0   0 441   6   0 177]
 [  5  55   1  11   0   1  10 425   0 257]
 [  2 462   1  21   0   1  45  37   1 134]
 [  4  78   2   7   0   0  93  66   0 436]]
             precision    recall  f1-score   support

        0.0       0.90      0.45      0.60       745
        1.0       0.31      0.79      0.44       789
        2.0       0.91      0.14      0.24       678
        3.0       0.49      0.33      0.39       735
        4.0       0.80      0.01      0.01       639
        5.0       0.12      0.00      0.00       610
        6.0       0.40      0.68      0.50       649
        7.0       0.62      0.56      0.59       765
        8.0       0.25      0.00      0.00       704
        9.0       0.20      0.64      0.30       686

avg / total       0.50      0.37      0.32      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7fef89adeb90>
in LayerNo.2: <function Sigmoid at 0x7fef89adeaa0>
The number of update is 50
learning rate is 0.010000
Input Noise are  5.000000 percent
The number of unit in No.1 layer is 501

             precision    recall  f1-score   support

        0.0       0.93      0.99      0.96       737
        1.0       0.96      0.98      0.97       808
        2.0       0.94      0.90      0.92       709
        3.0       0.88      0.94      0.91       716
        4.0       0.91      0.95      0.93       636
        5.0       0.94      0.88      0.91       633
        6.0       0.99      0.95      0.97       683
        7.0       0.88      0.97      0.92       679
        8.0       0.96      0.89      0.92       659
        9.0       0.95      0.86      0.90       740

avg / total       0.93      0.93      0.93      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7ff63382eb90>
in LayerNo.2: <function Sigmoid at 0x7ff63382eaa0>
The number of update is 100000
learning rate is 0.050000
Input Noise are  1.000000 percent
The number of unit in No.1 layer is 81
accuracy = 0.932714285714

             precision    recall  f1-score   support

        0.0       0.36      0.33      0.34       688
        1.0       0.49      0.39      0.43       724
        2.0       0.37      0.10      0.15       732
        3.0       0.33      0.44      0.38       697
        4.0       0.50      0.21      0.30       679
        5.0       0.30      0.27      0.28       660
        6.0       0.78      0.21      0.33       693
        7.0       0.56      0.39      0.46       733
        8.0       0.21      0.67      0.32       696
        9.0       0.25      0.33      0.29       698

avg / total       0.42      0.33      0.33      7000

activation functions are.. 
in LayerNo.1: <function tanh at 0x7f00d6736b90>
in LayerNo.2: <function Sigmoid at 0x7f00d6736aa0>
The number of update is 50
learning rate is 0.010000
Input Noise are  5.000000 percent
The number of unit in No.1 layer is 501
accuracy = 0.333142857143



unit 数が少ない時は　learning rate の最適値は大きくなる
unit数は小さめのほうが学習が捗る

"""
