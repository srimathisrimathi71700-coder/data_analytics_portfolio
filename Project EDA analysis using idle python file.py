import pandas as pd
import matplotlib.pyplot as plt
import sklearn
df = pd.read_csv(r"C:\Users\acer\Downloads\superstore_clean_200_rows.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df.dtypes
df['Delivery Days'] = (df['Ship Date'] - df['Order Date']).dt.days
df.head()
df['Order Month'] = df['Order Date'].dt.to_period('M')
df.head()
df.groupby("Category")["Sales"].sum().plot(kind="bar")
plt.title("Sales by Category")
plt.show()
df.groupby("Region")["Profit"].sum().plot(kind="bar")
plt.title("Profit by Region")
plt.show()
df.groupby("Order Month")["Sales"].sum().plot(kind="line")
plt.title("Monthly Sales Trend")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X = df[['Sales','Quantity','Discount']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, pred))
