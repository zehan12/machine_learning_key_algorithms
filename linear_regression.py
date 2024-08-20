import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Sample data
# Reshaping to make it a 2 Dimension array
stamps_bought = np.array([ 1, 3, 5, 7, 9 ]).reshape(( -1, 1 ))

amount_spent = np.array([ 2, 6, 8, 12, 18 ])


# Creating a Linear Regression Model
model = LinearRegression()

# Training the Model
model.fit( stamps_bought, amount_spent )

# Prediction
next_month_stamps = 10
predicted_spend = model.predict([[next_month_stamps]])

# Plotting
plt.scatter( stamps_bought, amount_spent, color="blue" )
plt.plot(stamps_bought, model.predict(stamps_bought), color="red")
plt.title("Stamps Bought vs Amount Spent")
plt.xlabel("Stamp Bought")
plt.ylabel("Amount Spent ($)")
plt.grid(True)
plt.show()

# Display Prediction
print(f"If Zehan buys {next_month_stamps} stamps next month, they will likely spend ${predicted_spend[0]:.2f}")