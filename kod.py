import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


print("csv dosyalarından veri okuma ve düzenleme çalışmaları.")
print("test_csv branchi üzerinde değişiklikler yapılıyor.")

df = pd.read_csv("diabetes.csv", encoding='unicode_escape')

"""
#countplot bir axesplot nsnesi oldupu için 1 tane plot oljuşturur. Ve tek bir parametre vermek yeterli oluyor.
#Çünkü; y-axis'i deafult olarak count ediyor. x-axis'teki değeri sayma işlemine tabi tutuyor.
sns.set_palette("RdBu")
sns.countplot(x="Age", data=df)
"""

"""
#catplot(categorical plot)
#aspect means x ekseni y eksenin 3 katı uzunluunda olsun demek.
#kind bar,box,count
sns.catplot(x="Age", aspect=3, data=df, kind="count")
"""

"""
# buradaki  "hue" c parameterisi gibi sınıflarınızın dağılımını görmenizi sağlayan bir parametredir.
sns.scatterplot(x="Age", y="Insulin", data=df, hue="Outcome")
"""

"""
sns.relplot(x="Age", y="BloodPressure", data=df, hue="Outcome", kind="scatter")
"""

"""
ci:confidence interval

"""
""" #2 boyutlu plotlar oluşuturabiliriz. 
sns.relplot(x="Insulin", y="Glucose", data=df, kind="scatter", col="Outcome", row="BMI")
"""

"""
#HeatMap, Açık renkli noktalara bakmakta fayda var.
# glucose değerleri outcome'ı etkiliyormuş gibi düşünebilirsiniz.
# Ml'de hangi öznitekileir kullanmak istediğiniz isz seçeebilirsiniz.
#
sns.set_palette("RdBu")
correlation = df.corr()
# annot=True parametriesi sayesinde öznitelikler arasındaki correlationlarını(ilişkileri) görebiliyoruz.
sns.heatmap(correlation, annot=True )
"""

"""
#Categorical PLot
#barplotlarını veya countplot ların önemi: bizim verimizin dapışımı hakkında yorum yapabilmemize yarar.
# BU dağılımın arası çok açık olmaması gerek gzel bir ML modeli oluşutrabilmek için.
#dengesizliği göremk için çok gğzel br yöntem

sns.catplot(x="Outcome", y="Insulin", data=df, kind="bar")
"""


"""
# InterQuantile Range = %25 ile %75 percentile arasını gösterir.
#maximum value Q3'ün sağına doğru +1.5 IQR katı kadar
#minimum value Q1'in soluna doğru -1,5 IQR katı kadar

sns.catplot(x="Outcome", y="Insulin", kind="box", data=df)
"""

"""
# Bu plotta outcome ların farkı çok açık oldupu için bu modelimiz istedğimiz accuracy'i veremicek.
# Sınıflarınız örneklem saysısı eşitse accuracy'ye bakın.
sns.catplot(x="Outcome", data=df, kind="count")
"""

"""
pairplot
"""


plt.show()








"""data = df.head()

fig, ax = plt.subplots()
ax.bar(df.Age, df.Insulin)
ax.set_xticklabels(df.Age, rotation=45)

ax.set_xlabel("Age")
ax.set_ylabel("Insulin")
#fig.savefig("age.png")



plt.show()"""


"""print("-"*20)

data = df.head()

# display
print(data)

print("-"*20)

print(data[["name", "age"]])

print("-"*20)

print(len(data))

print("-"*20)
data_copy = data.copy()

data_copy["ıncome"] = [5000, 12000, 7000, 4250]

print(data_copy)

x = data_copy[["age"]]
y = data_copy[["ıncome"]]
plt.scatter(x, y, s=389, alpha=0.2, cmap="viridis")

plt.colorbar()
plt.xlabel("age")
plt.ylabel("ıncome")
plt.title("title aq")

plt.show()"""