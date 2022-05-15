
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
import numpy as np


IMG_SIZE = 200
nb_classes=23
MODEL_NAME = 'handsign.model'

#Verileri etiketleme fonksiyonu
def one_hot_targets_(labels_dense,nb_classes):
    targets = np.array(labels_dense).reshape(-1)
    print(targets)
    one_hot_targets = np.eye(nb_classes,)[targets]
    return one_hot_targets

#Veri setinin yüklenmesi
train_data = np.load('train_data.npy',encoding="latin1",allow_pickle=True)
test_data = np.load('test_data.npy',encoding="latin1",allow_pickle=True)

print('traindatlen:'+str(len(train_data)))
print('testdatalen:'+str(len(test_data)))

#X değişkenine train data verileri yeniden boyutlandırılır, Y değişkenine ise train data etiket isimleri eklenir.
X = np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train_data]
Y1=one_hot_targets_(Y,nb_classes)

print('val y'+str(Y1))
print('len X:'+str(len(X)))
print('len Y:'+str(len(Y)))
test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test_data]
test_y1=one_hot_targets_(test_y,nb_classes)
test_y=test_y1
Y=Y1
print('test_x:'+str(len(test_x)))
print('test_y:'+str(len(test_y)))
print('val y'+str(test_y1))

#CNN mimarisi katmanları oluşturuluyor.
classifier = Sequential()
classifier.add(Conv2D(16,kernel_size=(3,3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu'))	
classifier.add(MaxPooling2D(pool_size =(2,2)))	
classifier.add(Conv2D(32,kernel_size=(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Conv2D(64,kernel_size=(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu'))

#classifier.add(Dropout(0.5))
classifier.add(Dense(nb_classes,activation = 'softmax'))

#Optimizasyon ve loss değerleri belirlenir ve model compile edilir.	
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


#Model eğitilir.
classifier.fit(X,  Y,epochs=20, validation_data=(test_x,  test_y),  steps_per_epoch = 800 )



classifier.save(MODEL_NAME)
score = classifier.evaluate(test_x, test_y)
print('Test accuarcy: %0.4f%%' % (score[1] * 100))

