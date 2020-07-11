from PIL import Image
import mnist
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()

# print(x_train.shape)

x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

x_train = (x_train/256)
x_test = (x_test/256)

# print(x_train[0])

clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64))

clf.fit(x_train, y_train)
print('done')

pred = clf.predict(x_test)

acc2 = accuracy_score(y_test, pred)

print(acc2)
