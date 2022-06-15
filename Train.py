
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

#Convolution2d katmanı 2 boyutlu matrisler olarak görülen girdi görüntülerimizle ilgilenecek evrişim katmanlarıdır.
# 32x32x3 girişe sahip 32 nöronlu , 3x3 filtre uygulanmış Convolution Katmanı
#Aktivasyon fonksiyonu olarak her katmanda ReLu kullanılarak sıfırın altında kalan kısımları sıfırlayıp işlem yükünü hafifletiriz.
classifier.add(Conv2D(32, (3, 3), padding = 'same',  activation = 'relu',input_shape = (32, 32, 3)))

#İkinci konvolüsyon katmanı ekleme
classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))

#Pooling işlemi ile görüntü yükseklik ve genişlik bakımından boyutu azaltılır. 
#2x2 boyutunda kümeler alınarak  bu kümeler içerisindeki en büyük değerler alınır.Stride(adım boyutu)=2 adım ilerleyip tekrar pooling işlemi yapar.
classifier.add(MaxPooling2D(pool_size =(2, 2),strides=2))

 #Üçüncü konvolüsyon katmanı eklenir.
classifier.add(Conv2D(64, (3,  3), padding='same', activation = 'relu'))
#Pooling 
classifier.add(MaxPooling2D(pool_size =(2, 2),strides=2))
#Dördüncü konvolüsyon katmanı eklenir.
classifier.add(Conv2D(128, (3,  3), padding= 'same', activation = 'relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size =(2,2),strides=2))

#Flatten, convolution  ve dense katmanlar arasında bir bağlantı görevi görür.
#YSA matris boyutunu kabul etmez. Bu yüzden Flatten(düzleştirme) işlemi yapılır.
#'Flatten' Genellikle Convolutional bölümünün sonuna konan Flatten metodu çok boyutlu olan verimizi tek boyutlu hale getirerek standart yapay sinir ağı için hazır hale getirir.
#Flattening
classifier.add(Flatten())

# Fully Connected Layer
# 'Dense' Bir standart yapay sinir ağı katmanı oluşturur, ilk parametrede verilen sayı kadar nöron barındırır.
# modele bir katman ekliyoruz(gizli katman) katmanımızda her olası sonuç için bir tane olmak üzere 256 nöron olacak.
classifier.add(Dense(256, activation = 'relu'))

#Dropout ile aşırı öğrenmeyi(overfitting) engellemek için nöronların %50 si unutulur.
classifier.add(Dropout(0.5));

#22 sınıfa ayrılır. Çıkış katmanı olarak genellikle 'softmax' önerilir. 
# 0-1 arasında değer gelir 1 e en yakın değeri softmax döndürür.
classifier.add(Dense(22, activation = 'softmax'))

#CNN compile edilir.
# Optimizer eğitim boyunca öğrenme hızını ayarlar.
#Kayıp fonksiyonumuz için 'categorical_crossentropy' kullanacağız. Bu, sınıflandırma için en yaygın seçimdir. 
#İşlerin yorumlanmasını daha da kolaylaştırmak için, modeli eğitirken doğrulama setindeki doğruluk puanını görmek için 'accuracy' metriğini kullanacağız.
classifier.compile(
              optimizer = keras.optimizers.Adagrad(learning_rate=0.02),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

classifier.summary()

#Veriler yüklenir. Önişleme aşamaları
from keras.preprocessing.image import ImageDataGenerator
#Feature Extraction işlemleri ile veriler okunur.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,        #bükme
        zoom_range=0.2,         #yakınlaştırma
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')
#Modelin eğitilmesi
model = classifier.fit_generator(
        training_set,
        steps_per_epoch=1000,
        epochs=45,
        validation_data = test_set,
        validation_steps = 8800
      )
score = classifier.evaluate(test_set,verbose=0)
print("Test loss: ", score[0])
print("Test Accuracy: ", score[1])

#Modelin kaydedilmesi
import h5py
classifier.save('Trained_model.h5')

print(model.history.keys())



import matplotlib.pyplot as plt
# Başarım Oranı
plt.plot(model.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# Yitim Fonksiyonu 
plt.plot(model.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()








