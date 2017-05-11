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

    for updatenumber in range (1000):
        mlp = MultiLayerPerceptron.MultiLayerPerceptron(28*28, [500], 10, 3, ["tanh","sigmoid"])
        # 訓練データを用いてニューラルネットの重みを学習
        mlp.fit(X_train, labels_train, learning_rate, updatenumber)


        # テストデータを用いて予測精度を計算
        predictions = []
        for i in range(X_test.shape[0]):
            o = mlp.UseMLP(X_test[i])
            # 最大の出力を持つクラスに分類
            predictions.append(np.argmax(o))

        #結果の表示
        # print classification_report(y_test, predictions)
        print str(accuracy_score(y_test, predictions))
    
