# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.iterative_method import BasicIterativeMethod
from art.attacks.carlini import CarliniL2Method

from art.classifiers import KerasClassifier
from art.utils import load_dataset

ddir = "/home/bwbell/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/"

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """ 
    b = a.swapaxes(axis, -1)
    n = a.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    b = b[..., idx]
    return b.swapaxes(axis, -1)

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))

# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
k.set_learning_phase(1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier = KerasClassifier((min_, max_), model=model)
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=128)

# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))

# Craft adversarial samples 

################ FGSM;
epsilon = .1  # Maximum perturbation
adv_crafter = FastGradientMethod(classifier)
x_test_adv = adv_crafter.generate(x=x_test, eps=epsilon)
# Evaluate the classifier on the adversarial examples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100)
)
# grab a particular example to play wit
a = (preds != np.argmax(y_test, axis=1))
nat_img = x_test[a]
adv_img = x_test_adv[a]
adv_nse = adv_img - nat_img
adv_prd = preds[a]
# compute variance and plot (some) example(s)
adv_var = np.sqrt(np.var(adv_nse)/np.var(nat_img))
adv_plt = np.column_stack([nat_img[0].reshape(28,28),
                        adv_nse[0].reshape(28,28),
                        adv_img[0].reshape(28,28)])

                        
fig= plt.figure()
plt.imshow(adv_plt)
plt.title("y_test:{},y_adv:{} -- Var: {}".format(np.argmax(y_test[0]), adv_prd[0], adv_var ))
fig.set_size_inches(24,12)
fo = ddir+"O{}A{}_varx{}_examp.png".format(np.argmax(y_test[0]), adv_prd[0], adv_var )
fig.savefig(fo, dpi=100)  

################ IFGSM;
epsilon = .05  # Maximum perturbation
adv_crafter = BasicIterativeMethod(classifier)
x_test_adv = adv_crafter.generate(x=x_test, norm=2, eps=epsilon, eps_step = epsilon/3)
# Evaluate the classifier on the adversarial examples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100)
)
# grab a particular example to play wit
a = (preds != np.argmax(y_test, axis=1))
nat_img = x_test[a]
adv_img = x_test_adv[a]
adv_nse = adv_img - nat_img
adv_prd = preds[a]
# compute variance and plot (some) example(s)
adv_var = np.sqrt(np.var(adv_nse)/np.var(nat_img))
adv_plt = np.column_stack([nat_img[0].reshape(28,28),
                        adv_nse[0].reshape(28,28),
                        adv_img[0].reshape(28,28)])

                        
fig= plt.figure()
plt.imshow(adv_plt)
plt.title("IFGSM:y_test:{},y_adv:{} -- Var: {}".format(np.argmax(y_test[0]), adv_prd[0], adv_var ))
fig.set_size_inches(24,12)
fo = ddir+"IFGSM-O{}A{}_varx{}_examp.png".format(np.argmax(y_test[0]), adv_prd[0], adv_var )
fig.savefig(fo, dpi=100)  

################ CW;
adv_crafter = CarliniL2Method(classifier)
#targets
y_test_tar = np.random.randint(0,10,y_test.__len__())
y_test_tarm = np.zeros(y_test.shape)
y_test_tarm[np.arange(0,y_test.shape[0]),y_test_tar] = 1
x_test_adv = adv_crafter.generate(x=x_test, y=y_test_tarm)
# Evaluate the classifier on the adversarial examples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100)
)
# grab a particular example to play with
a = (preds != np.argmax(y_test, axis=1))
nat_img = x_test[a]
adv_img = x_test_adv[a]
adv_nse = adv_img - nat_img
adv_prd = preds[a]

for i in np.random.choice(adv_img.__len__(),100):
# compute variance and plot (some) example(s)
    adv_var = np.sqrt(np.var(adv_nse[i])/np.var(nat_img[i]))
    adv_plt = np.column_stack([nat_img[i].reshape(28,28),
                               adv_nse[i].reshape(28,28),
                               adv_img[i].reshape(28,28)])

                        
    fig= plt.figure()
    plt.imshow(adv_plt)
    plt.title("CW:{:03}-y_test:{},y_adv:{} -- Var: {}".format(i,np.argmax(y_test[i]), adv_prd[i], adv_var ))
    fig.set_size_inches(24,12)
    fo = ddir+"CW{:03}-O{}A{}_varx{}_examp.png".format(i,np.argmax(y_test[i]), adv_prd[i], adv_var )
    fig.savefig(fo, dpi=100)  


