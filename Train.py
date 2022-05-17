#Part 1
#Kütüphane ve paketler import edilir.
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers


#CNN modeli oluşturuluyor.
classifier = Sequential()

# Step 1 - Convolutio Layer -Evrişim Katmanı
classifier.add(Conv2D(32, (3,  3), input_shape = (64, 64, 3), activation = 'relu'))

#step 2 - Pooling - Havuzlama Katmanı
classifier.add(MaxPooling2D(pool_size =(2,2)))

# İkinci convolution layer eklenir. 
classifier.add(Conv2D(32, (3,  3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Üçüncü Concolution Layer eklenir.
classifier.add(Conv2D(64, (3,  3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))


#Step 3 - Flattening - Düzleştirme
classifier.add(Flatten())

#Step 4 - Full Connection -Tam bağımlı katman
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(0.5))
#22 sınıfa ayrılır.
classifier.add(Dense(22, activation = 'softmax'))

#CNN compile edilir.
classifier.compile(
              optimizer = keras.optimizers.SGD(learning_rate=0.01),
              # model.compile(loss='categorical_crossentropy', optimizer=opt)(lr = 0.01),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Veriler yüklenir. Önişleme aşamaları
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
#Modelin eğitilmesi
model = classifier.fit_generator(
        training_set,
        steps_per_epoch=800,
        epochs=35,
        validation_data = test_set,
        validation_steps = 5500
      )

#Modelin kaydedilmesi
import h5py
classifier.save('Trained_model.h5')

print(model.history.keys())



import matplotlib.pyplot as plt
# Başarım Oranı
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Yitim Fonksiyonu 
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
