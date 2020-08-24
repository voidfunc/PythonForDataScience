import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Supervised Learning
#  1-  Classification
#  2-  Regression

"""
rng = np.random.RandomState()

x = 10 * rng.rand(50)

# eğimi 2, kesişim katsayısı -5
y = 2 * x - 5 + rng.randn(50)

plt.figure(dpi=200)
plt.scatter(x, y)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', rotation=0, fontsize=18)


print(x)
print(y)


# fit_intercept bias değerini yani kesişim katsayısının modele dahil edildiğini gösterir.
model = LinearRegression(fit_intercept=True)
# newaxis boyut artırmaya yarar.
model.fit(x[:, np.newaxis], y)

#linspace 0 ve 10 arasındaki değerlerden 1000 tane değer üzerinden bir linear çizgi oluştur diyor.
xfit = np.linspace(0, 10, 1000)
yfit =model.predict(xfit[:, np.newaxis])
plt.plot(xfit, yfit)
plt.show()

print("modelin eğimi: ", model.coef_[0])
print("Model kesişimi: ", model.intercept_)
"""

"""m = 100
X = 6 * np.random.rand(m, 1) -3
print(X)
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.figure(dpi=200)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', rotation=0, fontsize=18)
plt.scatter(X, y)
plt.show()"""

iris = datasets.load_iris()
X = iris["data"][:, 3:] #petal width
y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0

# l2penalty
log_reg = LogisticRegression(solver="liblinear", random_state=42)
log_reg.fit(X, y)
