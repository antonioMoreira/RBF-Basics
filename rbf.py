import numpy as np 
import numpy.linalg as LA
from scipy.spatial.distance import euclidean  
from scipy.spatial import distance_matrix

class RBF:
    def gaussian(self, x, centroid, spread):
        return np.exp( (-1/(2*pow(spread,2))) * pow(LA.norm(x-centroid),2) )

    def sigmoid(self, x, derivate=False):
        return 1/(1+np.exp(-x)) if not(derivate) else (x*(1-x))

    def init_Weights(self, row, col):
        return (np.random.standard_normal(size=(row, col))*1e-4)

    def getSpread(self, centroids):
        spread = []
        
        for i in distance_matrix(centroids, centroids):
            spread.append(np.max(i))

        return np.array(spread)

    def activationFunc(self, X):
        return 1 if X>0 else 0

    def feedRBF(self, x):
        f_net_hL = np.zeros(self.hL_size)

        for i in range(self.hL_size):
            f_net_hL[i] = (self.gaussian(x, self.centroids[i], self.spread[i]))
        
        return f_net_hL

    def feedForward(self, weight, X):
        X = np.append(X, 1)
        net = np.dot(weight, np.transpose(X))
        
        f_net_oL = np.zeros(len(net))

        for i in range(len(net)):
            f_net_oL[i] = self.activationFunc(net[i])

        return f_net_oL

    def feedForward_2(self, weight, X):
        X = np.append(X, 1)
        net = np.dot(weight, np.transpose(X))
        
        return self.sigmoid(net)

    def MSE(self, y, f_net, derivate=False):
        return sum(pow(y-f_net, 2)) if not(derivate) else (y-f_net)

    def learningPhase(self):
        error = 2*self.threshold
        epoch = 0

        while(error > self.threshold and epoch < self.n_Epochs):
            error = 0
            row = 0

            for i in self.X:
                f_net_hL = self.feedRBF(i)
                f_net_oL = self.feedForward(self.oL_Weights, f_net_hL)

                error += self.MSE(self.Y[row], f_net_oL)

                self.oL_Weights += self.eta * np.transpose([self.Y[row]-f_net_oL]) @ [np.append(f_net_hL,1)]

                row += 1

            error /= np.shape(self.X)[0]
            epoch += 1

            print('Epoch: ', epoch, '|', 'Error: ', error)

    def learningPhase_2(self):
        error = 2*self.threshold
        epoch = 0

        while(error > self.threshold and epoch < self.n_Epochs):
            error = 0
            row = 0

            for i in self.X:
                f_net_hL = self.feedRBF(i)
                f_net_oL = self.feedForward_2(self.oL_Weights, f_net_hL)

                error += self.MSE(self.Y[row], f_net_oL)
                d_error = self.MSE(self.Y[row], f_net_oL, True)

                localGrd = np.multiply(d_error, self.sigmoid(f_net_oL, True))
                self.oL_Weights += self.eta * np.multiply(np.transpose([localGrd]), np.append(f_net_hL, 1))

                row += 1

            error /= np.shape(self.X)[0]
            epoch += 1

            #print('Epoch: ', epoch, '|', 'Error: ', error)

    def predict(self, sample):
        f_net_hL = self.feedRBF(sample)
        f_net_oL = self.feedForward_2(self.oL_Weights, f_net_hL)

        maximum = np.argmax(f_net_oL)

        for i in range(len(f_net_oL)):
            if (i != maximum):
                f_net_oL[i] = 0
            else:
                f_net_oL[i] = 1

        return np.array(f_net_oL, dtype=int)

    def __init__(self, X, Y, centroids = None, hL_size = 2, oL_size = 1, eta = 1e-2, n_Epochs = 5e2, threshold = 1e-1):
        self.X = X
        self.Y = Y

        self.hL_size = hL_size
        self.oL_size = oL_size

        self.eta = eta
        self.threshold = threshold
        self.n_Epochs = n_Epochs

        self.oL_Weights = self.init_Weights(self.oL_size, self.hL_size+1)

        self.centroids = centroids
        self.spread = self.getSpread(self.centroids)