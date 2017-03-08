import numpy as np
import sys, os, struct, math
from array import array as pyarray
from pylab import *
import matplotlib.pyplot as plt
from random import randint
class MLP:
    def __init__(self, NHiddenLayers, NNodes, ActFunc, NClasses, DimInput, DActFunc):
        #setting up the data structures: list of delta, in and bias vectors and weight matrices
        self.NumLayers = NHiddenLayers + 2 #number of layers excluding the softmax layer
        self.NClasses = NClasses
        self.DimInput = DimInput
        self.NNodes = [DimInput] + NNodes + [NClasses]
        self.DeltaList = [np.zeros((self.NNodes[i], 1)) for i in range(self.NumLayers)]
        self.InputList = [np.zeros((self.NNodes[i], 1)) for i in range(self.NumLayers)]
        self.BiasList = [0.0 * np.random.random((self.NNodes[i], 1)) for i in range(self.NumLayers)]
        self.WeightList = [0.002 * (np.random.random((self.NNodes[i + 1], self.NNodes[i])) - 0.5) for i in range(self.NumLayers - 1)]
        self.WeightGradList = [np.zeros((self.NNodes[i + 1], self.NNodes[i])) for i in range(self.NumLayers - 1)]
        self.ActFunc = np.vectorize(ActFunc)
        self.DActFunc = np.vectorize(DActFunc);

    def ForwardFlow(self, X, Y):
        self.InputList[0] = X;
        for i in range(self.NumLayers - 1):
            self.InputList[i + 1] = np.dot(self.WeightList[i], self.ActFunc(self.InputList[i])) + self.BiasList[i + 1];
        Out = np.exp(self.ActFunc(self.InputList[self.NumLayers - 1]))
        Out = Out / (np.sum(Out))
        return Out, -np.dot(np.transpose(Y), np.log(Out))[0, 0]

    def BackwardFlow(self, Y):
        self.DeltaList[self.NumLayers - 1] = np.exp(self.ActFunc(self.InputList[self.NumLayers - 1]))/np.sum(np.exp(self.ActFunc(self.InputList[self.NumLayers - 1]))) - Y
        self.DeltaList[self.NumLayers - 1] = self.DeltaList[self.NumLayers - 1] * self.DActFunc(self.InputList[self.NumLayers - 1])
        for i in range(self.NumLayers - 2, -1, -1):
            self.DeltaList[i] = self.DActFunc(self.InputList[i]) * np.dot(np.transpose(self.WeightList[i]), self.DeltaList[i + 1])

    def Gradient(self):
        for i in range(self.NumLayers - 1):
            self.WeightGradList[i] = np.dot(self.DeltaList[i + 1], np.transpose(self.ActFunc(self.InputList[i])))


    def CheckGradient(self, XBatch, YBatch):
        epsilon = 1e-7
        BackPropagationGradients = []
        NumericalGradients = []
        for (X, Y) in zip(XBatch, YBatch):
            (_, JP) = self.ForwardFlow(X, Y)
            self.BackwardFlow(Y)
            self.Gradient()
            B = 0
            N = 0
            for i in range(self.NumLayers - 1):
                Dim = np.shape(self.WeightList[i])
                for j in range(Dim[0]):
                    for k in range(Dim[1]):
                        self.WeightList[i][j, k] = self.WeightList[i][j, k] + epsilon
                        (_, JN) = self.ForwardFlow(X, Y)
                        B += self.WeightGradList[i][j, k] ** 2
                        N += ((JN - JP) / epsilon) ** 2
                        self.WeightList[i][j, k] = self.WeightList[i][j, k] - epsilon
            for i in range(1, self.NumLayers):
                Dim = np.shape(self.BiasList[i])
                for j in range(Dim[0]):
                    self.BiasList[i][j, 0] = self.BiasList[i][j, 0] + epsilon
                    (_, JN) = self.ForwardFlow(X, Y)
                    B += self.DeltaList[i][j, 0] ** 2
                    N += ((JN - JP) / epsilon) ** 2
                    self.BiasList[i][j, 0] = self.BiasList[i][j, 0] - epsilon
            print(B, N)
            BackPropagationGradients.append(B)
            NumericalGradients.append(N)

        i = (np.arange(len(XBatch)) + 1).tolist()
        plt.plot(i, BackPropagationGradients, color = 'blue', marker = '*', label = 'BackPropagation', fillstyle = 'none')
        plt.plot(i, NumericalGradients, color = 'red', marker = 'o', label = 'Numerical', fillstyle = 'none')
        plt.xlabel('Sample Number')
        plt.ylabel('Sum of squared gradients')
        plt.legend(loc = 'best')
        plt.savefig('Gradients.png')
        plt.close()


    def BatchGradients(self, X, Y, MiniBatch):
        NumSamples = len(X)
        temp = 0
        WeightBatch = [0] * self.NumLayers
        BiasBatch = [0] * self.NumLayers
        for j in range(MiniBatch):
            k = randint(0, NumSamples - 1)
            (_, Objective) = self.ForwardFlow(X[k], Y[k])
            temp += Objective
            self.BackwardFlow(Y[k])
            self.Gradient()
            for i in range(self.NumLayers - 1):
                WeightBatch[i] += self.WeightGradList[i]
            for i in range(1, self.NumLayers):
                BiasBatch[i] += self.DeltaList[i]
        for i in range(self.NumLayers - 1):
            self.WeightGradList[i] = WeightBatch[i] / MiniBatch
        for i in range(1, self.NumLayers):
                self.DeltaList[i] = BiasBatch[i] / MiniBatch
        return temp / MiniBatch

    def SGDMomentum(self, X, Y, Iterations, Gamma, MiniBatch, LearningRate):
        NumSamples = len(X)
        for t in range(Iterations):
            cost = self.BatchGradients(X, Y, MiniBatch)
            #print(t, cost)
            WeightMomentumList = [0] * self.NumLayers
            BiasMomentumList = [0] * self.NumLayers
            for i in range(self.NumLayers - 1):
                WeightMomentumList[i] = Gamma * WeightMomentumList[i] + LearningRate * self.WeightGradList[i]
                self.WeightList[i] = (self.WeightList[i] - WeightMomentumList[i])
            for i in range(1, self.NumLayers):
                BiasMomentumList[i] = Gamma * BiasMomentumList[i] + LearningRate * self.DeltaList[i]
                self.BiasList[i] = self.BiasList[i] - BiasMomentumList[i]

    
    def RMSProp(self, X, Y, Iterations, Gamma, LearningRate, MiniBatch, epsilon = 1e-8):
        NumSamples = len(X)
        for t in range(Iterations):
            cost = self.BatchGradients(X, Y, MiniBatch)
            print(t, cost)
            WeightRMSList = [0] * self.NumLayers
            BiasRMSList = [0] * self.NumLayers
            for i in range(self.NumLayers - 1):
                WeightRMSList[i] = Gamma * WeightRMSList[i] + (1 - Gamma) * self.WeightGradList[i] * self.WeightGradList[i]
                self.WeightList[i] = self.WeightList[i] - LearningRate * self.WeightGradList[i]/(np.sqrt(WeightRMSList[i]) + epsilon)
            for i in range(1, self.NumLayers):
                BiasRMSList[i] = Gamma * BiasRMSList[i] + (1 - Gamma) * self.DeltaList[i] * self.DeltaList[i]
                #self.BiasList[i] = self.BiasList[i] - LearningRate * self.DeltaList[i]/(np.sqrt(BiasRMSList[i]) + epsilon)

    def Test(self, X, Y, Numsamples):
        TotalSamples = 0
        Misclassifications = 0
        n = len(X)
        for i in range(Numsamples):
            k = randint(0, n - 1)
            (res, _) = self.ForwardFlow(X[k], Y[k])
            if(np.argmax(res) != np.argmax(Y[k])): Misclassifications = Misclassifications + 1
            TotalSamples = TotalSamples + 1
        return 100 * ((TotalSamples - Misclassifications) / TotalSamples) 

    def FinalTest(self, X, Y):
        TotalSamples = 0
        Misclassifications = 0
        for (x, y) in zip(X, Y):
            (res, _) = self.ForwardFlow(x, y)
            if(np.argmax(res) != np.argmax(y)): Misclassifications = Misclassifications + 1
            TotalSamples = TotalSamples + 1
        return 100 * ((TotalSamples - Misclassifications) / TotalSamples) 

    def Train(self, XTrain, YTrain, XValidation, YValidation, OptimizerType, OptimizerParams, TotalIterations):
        PrevValidation = 0
        Iterations = 0
        Validation = 0
        OptimizerParams['X'] = XTrain
        OptimizerParams['Y'] = YTrain
        T = []
        V = []
        while(Iterations < TotalIterations):
            if(OptimizerType == 'Momentum'): self.SGDMomentum(**OptimizerParams)
            else: self.RMSProp(**OptimizerParams)
            Train = 100 - self.Test(XTrain, YTrain, 1000)
            Validation = 100 - self.Test(XValidation, YValidation, 1000)
            T.append(Train)
            V.append(Validation)
            print(Train, Validation)
            Iterations = Iterations + OptimizerParams['Iterations']
            print(Iterations)
        i = (OptimizerParams['Iterations']*(np.arange(len(T)) + 1)).tolist()
        plt.plot(i, T, color = 'blue', label = 'Training Error', fillstyle = 'none')
        plt.plot(i, V, color = 'red', label = 'Validation Error', fillstyle = 'none')
        plt.xlabel('Epoch')
        plt.ylabel('Misclassification Percentage')
        plt.grid(True)
        plt.legend(loc = 'best')
        plt.savefig('TrainVsValidation.png')
        plt.close()

def Relu(x):
    if x > 0: return x
    else: return -x

def DRelu(x):
    if(x > 0): return 1
    else: return -1

def Tanh(x):
    return math.tanh(x)

def DTanh(x):
    return 1 - Tanh(x)**2

def Sigmoid(x):
    return 1/(1 + math.exp(-x))

def DSigmoid(x):
    return Sigmoid(x)*(1 - Sigmoid(x))

def load_mnist(dataset="training", digits = np.arange(10), path="."):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    lbl_file = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", lbl_file.read(8))
    lbl = pyarray("b", lbl_file.read())
    lbl_file.close()
    img_file = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", img_file.read(16))
    img = pyarray("B", img_file.read())
    img_file.close()
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        labels[i] = lbl[ind[i]]
    perm = np.random.permutation(images.shape[0])
    images = images[perm] / 256.0
    labels = labels[perm]
    X = list(images)
    X = [i.reshape((784, 1)) for i in X]
    labels = list(labels)
    Y = []
    for i in labels:
        t = np.zeros((10, 1))
        t[i[0], 0] = 1
        Y.append(t)
    return X, Y


(X, Y) = load_mnist("training")
(XTest, YTest) = load_mnist("testing")
XTrain = X[:45000]
YTrain = Y[:45000]
XVal = X[45000:]
YVal = Y[45000:]
skynet = MLP(2, [100, 25], Relu, 10, 784, DRelu)
#skynet.CheckGradient(XTrain[:20], YTrain[:20])
skynet.Train(XTrain, YTrain, XVal, YVal, 'RMSProp', {'Iterations': 1000, 'Gamma': 0.9, 'LearningRate': 0.003, 'MiniBatch': 10}, 2000)
print(skynet.FinalTest(XTest, YTest))

