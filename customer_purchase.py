import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Create sample customer data
np.random.seed(42)

data = pd.DataFrame({
    "Income": np.random.randint(10000,100000,200),
    "Age": np.random.randint(18,60,200),
    "Spending": np.random.randint(1,10,200)
})


# Purchase decision rule
data["Purchase"] = (
    (data["Income"] > 50000) &
    (data["Spending"] > 5)
).astype(int)


X = data[["Income","Age","Spending"]]
y = data["Purchase"]


# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# DNN
model = Sequential()

model.add(Dense(16,activation="relu",input_shape=(3,)))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Fast training
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    verbose=1
)


# Test
loss,acc = model.evaluate(X_test,y_test)

print("Accuracy:",acc)


# Try new customer
income = int(input("Enter Income: "))
age = int(input("Enter Age: "))
spending = int(input("Enter Spending habit (1-10): "))


customer = scaler.transform(
    [[income,age,spending]]
)


result = model.predict(customer)


if result[0][0] > 0.5:
    print("Purchase Decision: YES")
else:
    print("Purchase Decision: NO")