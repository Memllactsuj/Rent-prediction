import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import pickle

data = pd.read_csv("data.csv")

# перевірка пропущених значень (покаже скільни nan у кожному стовпці)
print(data.isnull().sum())

data = data.dropna() # видалення всіх рядків з пропущеними значеннями

print(data.duplicated().sum()) # перевірка на дублікати

data = data.drop_duplicates() # видалити дублікати

# розділення на ознаки (Х) та цільову змінну (у)
# допустимо я хочу передбачити ціну
X = data.drop(columns=['price'])
y = data['price']

# перетворення нечислових ознак у числові
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

'''
оцінка якості моделі
після того як я навчила модель мені треба перевірити наскільки добре 
вона працює. для цього я: 
1. отримую передбачення (y_pred), тобто беру тестові однаки X_test і за допомогою 
навченої можелі передбачую
2. MAE оцінюю наскільки гарні передбачення, тепер використовую mean_absolute_error, шоб 
обчислити середню різницю між фактичними і передбаченими цінами (бере абсолютне значення)
3. MSE знаходжу середню квадратичну помилку (порівнюю справжні і передбачені значення але різніця
у квадраті)
4. R^2 знаходжу коефіцієнт детермінації (показник якості регресії)
'''

y_pred = model.predict(X_test)
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("R^2: ", r2_score(y_test, y_pred))

importance = model.coef_
features = X.columns

feature_impotance = pd.DataFrame({
	'Feature': features,
	'Impact': importance
}).sort_values(by='Impact', ascending=False)

print(feature_impotance)

# якщо точки лягають по прямій то все добре
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, colorizer='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Real price')
plt.ylabel('Predicted price')
plt.title('Real vs predicted price')
plt.grid(True)
plt.show()

# гістограма залишків
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=30, color='green', edgecolor='black')
plt.xlabel('Error (y_test - y_pred)')
plt.ylabel('Count')
plt.title('Gistogram')
plt.grid(True)
plt.show()

# зберігаю у файли 
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
#X = pd.get_dummies(X)
with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)