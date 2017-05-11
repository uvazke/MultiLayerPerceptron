#coding:utf-8
import MultiLayerPerceptron
import numpy as np
import sys
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    #学習率
    learning_rate = 0.01
    updatenumber = 10000
    
    pictures = fetch_mldata('MNIST original', data_home=".")
    # 訓練データを作成
    X = pictures.data
    y = pictures.target
    
    # ピクセルの値を0.0~1.0に正規化
    X = X.astype(np.float64)
    X /= X.max()
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # 出力としては10元のベクトルを用意して、k個目のunitにはkが対応し、答えのunitに1が入る
    labels_train = LabelBinarizer().fit_transform(y_train)

    mlp = MultiLayerPerceptron.MultiLayerPerceptron(28*28, [1000], 10, 3, ["tanh","sigmoid"])
    for i in range (10):
        # 訓練データを用いてニューラルネットの重みを学習
        mlp.fit(X_train, labels_train, learning_rate, updatenumber)


        # テストデータを用いて予測精度を計算
        predictions = []
        for i in range(X_test.shape[0]):
            ###入力にノイズを混ぜる
            for k in range(len(X_test[i])):
                rand = np.random.rand()
                if rand < 0.25:
                    X_test[i][k] = np.random.rand()
            o = mlp.UseMLP(X_test[i])
            # 最大の出力を持つクラスに分類
            predictions.append(np.argmax(o))

        #結果の表示
        # print classification_report(y_test, predictions)
        print str(accuracy_score(y_test, predictions))
    
    #結果詳細の表示
    print confusion_matrix(y_test, predictions)
    print classification_report(y_test, predictions)
    print "activation functions are.. "
    for i in range(len(mlp.f))[1:]: 
        print "in LayerNo.%d: " % (i) + str(mlp.f[i])
        print "The number of update is %d" % (updatenumber * i) 
        print "learning rate is %f" % learning_rate 
        print "Input Noise are  %f percent" % (mlp.NoisePercentage * 100)
        for i in range(mlp.layernum - 2 ):
            print "The number of unit in No.%d layer is %d" %((i + 1), mlp.hiddennum[i+1]) 
    print "accuracy = " + str(accuracy_score(y_test, predictions))
        
