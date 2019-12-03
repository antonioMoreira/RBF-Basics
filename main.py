import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from rbf import RBF

def preProcessing_Data(path):
    fp = open(path, 'r')

    X = [] # features
    Y = [] # labels

    for i in fp:
        x = []

        for j in range(16):
            aux = list(map(float, i.split()[16*j:16*(j+1)]))
            aux = np.array(aux, dtype='int')
            
            shift = np.arange(len(aux)-1, -1, -1)

            x.append(np.sum(aux << shift))             

        X.append(x)
        
        #y = np.array(list(map(float, i.split()[-10:])))
        #Y.append(np.argwhere(y == 1)[0][0])
        Y.append(list(map(float, i.split()[-10:])))

    fp.close()

    return (np.array(X, dtype=float),np.array(Y, dtype=int))

def main():
    (X, Y) = preProcessing_Data("./data/semeion.data")

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    #Y_norm = scaler.fit_transform(Y)

    #rbfSize = 64

    #kmeans = KMeans(n_clusters=rbfSize).fit(X_norm)
    #print(kmeans.cluster_centers_)

    #print(np.shape(kmeans.cluster_centers_))

    #rbf = RBF(X_norm, Y, kmeans.cluster_centers_, hL_size=rbfSize, oL_size=10)
    #rbf.learningPhase_2()

    kf = KFold(n_splits=5)
    fold = 1

    for train_index, test_index in kf.split(X_norm):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        rbfSize = 16

        print(fold, "º Fold")

        while(rbfSize <= 72):
            kmeans = KMeans(n_clusters=rbfSize).fit(X_train)
            
            rbf = RBF(X_train, Y_train, kmeans.cluster_centers_, hL_size= rbfSize, oL_size = 10)
            rbf.learningPhase_2()

            Y_pred = []

            for i in X_test:
                Y_pred.append(rbf.predict(i))
            
            score =  accuracy_score(Y_test, np.array(Y_pred))
            
            print("\tRBF Size: ", rbfSize, "| Score: ", score)

            rbfSize += 16

        fold += 1
        print("=======================")


if __name__ == "__main__":
    main()