import numpy as np
from core.directional import VonMisesFisher
from core.hotelling import Hotelling
from core.bayes import NaiveBayes
import pickle


items, dims = 20, 5
np.random.seed(1)
dt = np.random.random(items * dims).reshape(dims, -1)
sqrt, bias = np.random.randint(1, 4, dims), np.random.randint(1, 5, dims)  # bias = [2, 2, 3, 2, 2]
dt = np.array([dt[i] * np.sqrt(sqrt[i]) + bias[i] for i in range(len(dt))]).T


print('-'*50, 'test for hotelling.py', '-'*50)
model = Hotelling(model_import=dt)
print('result for model-regular instance:', model.predict(data_import=model.model)[0])
print('result for anomalous instance:', model.predict(data_import=np.array([[5, 2, 3, 2, 7]]))[0])

print('-'*50, 'test for directional.py', '-'*50)
model = VonMisesFisher(model_import=dt)
print('result for model-regular direction:', model.predict(data_import=np.array([bias]))[0])
print('result for anomalous instance:', model.predict(data_import=np.array([[5, 2, 3, 2, 7]]))[0])

print('-'*50, 'test for bayes.py', '-'*50)
with open('../dataset/ts/Xy', 'rb') as f:
    X, y = pickle.load(f)
clf = NaiveBayes()
clf.fit(x=X, y=y)
print('result of threshold after fitting:', clf.threshold)
clf.update(x=X[200:400], y=y[200: 400])
print('result of threshold after updating:', clf.threshold)
print('implementation of prediction after updating:', clf.predict(data_import=X[50:55]))


if __name__ == '__main__':
    pass
