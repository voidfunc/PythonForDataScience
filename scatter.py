import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

dataset = {"İsim":  ["Mert", "Nilay", "Dogancan", "Omer", "Merve", "Onur"],
           "Soyad": ["Cobanov", "Mertal", "Mavideniz", "Cengiz", "Noyan", "Sahil"],
           "Yas":   [24,22,24,23, "bilinmiyor", 23],
           "Sehir": ["Bursa", "Ankara", "Istanbul", np.nan, "Izmir", "Istanbul"],
           "Ulke":  ["Turkiye", "Turkiye", "Turkiye", "Turkiye", "Turkiye", "Turkiye"],
           "GANO":  [np.nan, np.nan, np.nan, np.nan, 3.90, np.nan]}

df = pd.DataFrame(dataset)

"""
#data frame hakkında bilgi, nan değerler, featureların data typeları
df.info()
"""

"""
#we ask a question to program "is it nan?"
print(df.isna())
print(df.isna().sum())
print(df.isna().sum().sum())
"""

"""
# remove missing values
# axis: 1 -> column / 0 -> row, Any = 0, how: Any = "any", thresh: Any = None,
# subset: Any = None, inplace: Any = False
# print(df.dropna(axis=1, how="any", thresh=3))
df_2 = df.drop(labels=["GANO", "Ulke"], axis=1)
print(df_2)
print(df)
"""

df_2 = df.drop(labels=["GANO", "Ulke"], axis=1)
df_2["Yas"].replace("bilinmiyor", np.nan, inplace=True)
#df_2["Yas"].fillna(value=df_2["Yas"].mean(), inplace=True)

"""
# bilinmioyr yazılı featureların yerine np.nan yazar
df_2["Yas"].replace("bilinmiyor", np.nan, inplace=True)

# Yas kolonundaki NaN değerleri value'ya atadığımız verilerle doldurma işlemi
# Yas kolonundaki değerlerin oratalmasını NaN değerlerine atar.
df_2["Yas"].fillna(value=df_2["Yas"].mean(), inplace=True)
"""

imp_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
df_2["Yas"] = imp_freq.fit_transform(df_2[["Yas"]])
"""
# SimpleImputer objesi oluşturduk. en çok tekrarlanan değeri NaN değerlrine atama yapar.
imp_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
# Verilen columndaki değrleri tarar.  Bu işlemi fit_transform method'uyla yapar.
df_2["Yas"] = imp_freq.fit_transform(df_2[["Yas"]])
"""

"""
# Enterpolasyon : varolan sayısal değerleri kullanarak, boş noktalardaki değerlerin tahmin edilmesi
s = pd.Series([0, 1, np.nan, 3])
print(s.interpolate())
"""

"""
# Elimizdeki matrisi  DataFrame'e çevirerek daha kolay işlem yapabilriiz.
X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
X = pd.DataFrame(X)


# KNNImputer objesi oluşturarak, NaN değerlerini erafındaki featurelara bakarak doldurma işlemi yapar.
#NaN valuelarınıı doldurmak için çok iyi bir yöntem.
imputer = KNNImputer(n_neighbors=2, weights="uniform")
X = imputer.fit_transform(X)
X = pd.DataFrame(X)
print(X)
"""
df_2["Sehir"] = df_2["Sehir"].replace(np.nan, "diger")

"""
# Standard Scaling = bir kolondaki dağılımı, ortalması 0 ve standart sapması 1 olacak şekilde yerine scale etme işlemin denir.
df_ss = df_2.copy()
df_ss["Yas_Scaled"] = StandardScaler().fit_transform(df_ss[["Yas"]])
print(df_ss)
"""

print(df_2)
print("------------------------------")

"""
#MinMaxScaler kolondaki değrleri 0 ila 1 arasındaki değerlere eşler ve sıralar.
df_mm  = df_2.copy()
df_mm["Yas_Scaled"] = MinMaxScaler().fit_transform(df_mm[["Yas"]])
print(df_mm)
"""

"""
#kolondaki herrbir farklı değeri bir sayı ile ifade etmene yarar.
# Örneğin; şehirleri plaka numaraları ile sınıflandırma
df_le = df_2.copy()
le = LabelEncoder()
le.fit(df_le["Sehir"])

df_le["Sehir"] = le.transform(df_le["Sehir"])
print(df_le)
"""

"""
#OneHotEncoding, kolondaki değerleri kolonlara çevirerek 0 veya 1 ile sınıflandırma yapar.
print(pd.get_dummies(df_2["Sehir"]))
"""


# 3x3 lük bir matris

"""
X = np.array ([[9., 5., 15],
               [0., 6., 14],
               [6., 3., 11]])

# kolondaki değerlerini n_bins de verdiğimiz değer kadar bölümlere ayırır.
# İlk kolon için 3 dedik, bu yüzden 9 -> 2, 0 -> 0, 6 ise 1 oldu. 
KBD = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode="ordinal").fit_transform(X)
print(X)
print(KBD)
"""

"""
X = np.array ([[9., 5., 15],
               [0., 6., 14],
               [6., 3., 11]])
# Belirledğimiz thresholda göre 1 veya 0 atanır.
binarizer = preprocessing.Binarizer(threshold=5.1)
B = binarizer.transform(X)

print(X)
print(B)
"""

# Feature Selection : fazlalık verileri atma, overfitting'i azaltır.
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("mushrooms.csv")

# Target'ımızı droplayarak X'e sadece featurelar kalıyor.
X = data.drop(["class"], axis=1)
# Targetlarımızı buraya atadık.
y = data["class"]

# Burada tüm kolonlardaki herbir farklı faeture için kolon oluşturuldu ve 0 veya 1 ile gösterim yapıldı
#One Hot Encoder yapıldı.
X_encoded = pd.get_dummies(X, prefix_sep="_")

# Target featurelarımız 0 veya 1 olarak gösterilmeye başlandı
y_encoded = LabelEncoder().fit_transform(y)

# encode ettiğimiz X_encoded DataFrame'ini standardScaler ile ortalaması 0 standart sapmsaı 1 olan bir DF'e çevirir.
X_scaled = StandardScaler().fit_transform(X_encoded)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Featurelarımızı ilk one hot encode ettik daha sonra standard scale işlemine soktuk FAKAT,
# Target class'ımızı sadece Label Encode işlemine tabi tuttuk
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Modeli eğitim kısmına geçmek, veriyi train ve test işlemlerine sokma işlmei
# train_test_split fonksiyonu veriyi train ve test olarak ikiye böler.
# test_size test için verinin %30'unu ayırdığını belirtiyor.
# random_state veriyi random alaraktest ve train eder.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=101)

# Modeli çalıştırıyoruz..........
from sklearn.ensemble import RandomForestClassifier
# Performansı ölçmek için
from sklearn.metrics import classification_report, confusion_matrix
import time

start = time.process_time() # modeli ne kadar sürede train ettiğimizi ölçtük
model = RandomForestClassifier(n_estimators=700).fit(X_train, y_train)
print(time.process_time() - start)

# Bakalım modelimiz nasıl tahmin etmiş, performansını ölçelim
preds = model.predict(X_test) # X_test teki değerlere bakarak tahminde bulunuyor.
print(confusion_matrix(y_test, preds)) # burada az önce tahminde bulunduumuz değerlerle gerçek değerleri karşılaştırıyoeuz.

import matplotlib.pyplot as plt
# Feature importance,  değerlerin importance değerlerini verir. Burda hangi kolonların önemini görmek istedğimizi belirttik.
# FI anladıpım kadarıyla bir mantarın zehirli olduğunu anlamak için en çok hangi özellikler rol oynuyor onları görmemizi sağlıyor
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
feature_imp = pd.Series(model.feature_importances_, index=X_encoded.columns)
print(feature_imp.nlargest(10).plot(kind="barh"))
#plt.show()


# Şimdi en önemli 4 colum'u alarak modelimiz eğieceğiz bu sayede daha hızlı bir sonuç elde edeceiğiz
# Fakat zamandan kazanç sağlarken kaliteden ne kadar ödün vericez göreceğiz..............................
best_feat = feature_imp.nlargest(4).index.to_list() # en önemli 4 kolonu aldık ve bir listeye dönüştürdük
X_reduced = X_encoded[best_feat] # X_reduced adlı bir data frame' e çevirdik.
print("---------------------")

# X_reduced standard scale işlemine soktuk
Xr_scaled = StandardScaler().fit_transform(X_reduced) # modele vermeden önce yapmalıyız.

# Burada ilk 4 kolonu eğitime ve teste gönderiyoruz.
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_scaled, y, test_size=0.3, random_state=101)

# İşlem süresini ölçtüğümüzde ciddi anlamda bir zamandan kazanç sağladık.
start = time.process_time() # modeli ne kadar sürede train ettiğimizi ölçtük
rmodel = RandomForestClassifier(n_estimators=700).fit(Xr_train, yr_train)
print(time.process_time() - start)

# Şimdi kaliteden ne kadar ödün verdik onu ölçeceğiz!!!!!!!!!!!!!!!!!!!!!!!!!!!!

rpred = rmodel.predict(Xr_test)
print(confusion_matrix(yr_test, rpred))
print(classification_report(yr_test, rpred))
# precision değeri %100'den %97'ye geriledi. Fakat süre bakımından %40'lık bir gelişmeye bakılırsa,
# bu %3'lük kısım görmezden gelinebilir.( Bazı druumlarda)


import seaborn as sns
# Korelasyon Matrisi
X = data.drop(["class"], axis=1)  # Target'ımızı droplayarak X'e sadece featurelar kalıyor.
y = data["class"] # Targetlarımızı buraya atadık.

# Burada tüm kolonlardaki herbir farklı faeture için kolon oluşturuldu ve 0 veya 1 ile gösterim yapıldı
#One Hot Encoder yapıldı.
X_encoded = pd.get_dummies(X, prefix_sep="_")

# Target featurelarımız 0 veya 1 olarak gösterilmeye başlandı
y_encoded = LabelEncoder().fit_transform(y)

X_encoded["class"] = y_encoded

#sns.heatmap(X_encoded.iloc[:, -7:].corr(), annot=True) # Annot, heatmap'te correlation değerlerini göstermemize yarar.
# Yüksek korelasyon değerlerine sahip olan kolonları seçmek bir feature selection tekniğidir.
# Bu kolonlar en çok bilgiyi taşıyan kolonlar denir

print(X_encoded.corr().abs()["class"].nlargest(10))
#plt.show()

X_reduced_col_names = X_encoded.corr().abs()["class"].nlargest(10).index

sns.heatmap(X_encoded[X_reduced_col_names].corr().abs(), annot=True)
#plt.show()
plt.figure(figsize=(10, 10), dpi=400)



