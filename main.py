import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv("train.csv")

print("Первые строки датасета:")
print(df_train.head())

print("\nКоличество пропущенных значений:")
print(df_train.isnull().sum())

numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in numerical_cols:
    df_train[col] = df_train[col].fillna(df_train[col].median())

categories_cols = ["Cabin", "HomePlanet"]
for col in categories_cols:
    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

print("\nКоличество пропущенных значений после заполнения:")
print(df_train.isnull().sum())

# Нормализация
scaler = MinMaxScaler()
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])

# Преобразование категориальных данных
df_train = pd.get_dummies(df_train, columns=["HomePlanet", "Destination"], drop_first=True)

df_train.to_csv("processed_titanic_new.csv", index=False)
print("\nФайл processed_titanic_new.csv сохранен.")