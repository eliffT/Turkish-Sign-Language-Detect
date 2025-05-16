# *TÜRK İŞARET DİLİ ALFABESİ TANIMA*

Bu proje, Türk İşaret Dili alfabesini tanımak için bir evrişimli sinir ağı (CNN) modeli kullanılarak geliştirilmiştir. Proje, işaret dilini tanıyarak alfabedeki harfleri doğru bir şekilde sınıflandırmayı amaçlamaktadır. 

Makale: [Evrişimli Sinir Ağları kullanılarak Türk İşaret Dili Alfabesinin Tespit Edilmesi](https://dergipark.org.tr/tr/pub/rahva/issue/89471/1556400)

## *Veri Seti*

Bu proje için kullanılan veri kümesi, farklı el hareketleri ve pozisyonları içeren görüntülerden oluşmaktadır. Veri kümesi, her harf için yeterli sayıda örnek içerir ve modelin eğitimi için kullanılmıştır. Veri seti içerisinde dinamik görüntülerden oluşan harfler (ç, ğ, i, j, ö, ş, ü) bulunmamaktadır. 22 harf kullanılarak model eğitilmiştir.

Veri Seti: [TİD-Veri-Seti](https://www.kaggle.com/datasets/berkaykocaoglu/tr-sign-language/data?select=tr_signLanguage_dataset)

## *Projenin Çalıştırılması*
1. Gerekli kütüphaneleri yükleyin:

   ```bash
   pip install tensorflow opencv-python numpy matplotlib 

2. Veri kümesini indirin ve data/ dizinine yerleştirin.

3. Modeli eğitmek için aşağıdaki komutu çalıştırın:
   ```bash
      python Train.py
     ``` 
5. Eğitilmiş modeli kullanarak tahmin yapmak için:
   ```bash
      python Predict.py
     ```

## *Sonuçlar*
Model, %96.84 doğruluk oranına ulaşmıştır. Eğitim ve doğrulama süreci boyunca elde edilen sonuçlar aşağıdaki grafikte gösterilmektedir.

