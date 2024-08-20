import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data
pages = np.array([ 100, 150, 200, 250, 300, 350, 400, 450, 500 ]).reshape( -1, 1 )
# Like : 1, Dislike : 0
likes = np.array([ 0, 1, 1, 1, 0, 0, 0, 0, 0 ])

# Creating a Logistic Regression Model
model = LogisticRegression()

# Training the Model
model.fit(pages, likes)

# Predictions
predict_book_pages = 260
predicted_likes = model.predict([[predict_book_pages]])


# Plotting
plt.scatter( pages, likes, color="forestgreen" )
plt.plot( pages, model.predict_proba(pages)[:,1], color="darkred")
plt.title("Book Pages vs Like/Dislike")
plt.xlabel("Number of Pages")
plt.ylabel("Likelihood of Liking")
plt.axvline( x=predict_book_pages, color='green', linestyle='--' )
plt.axhline( y=0.5, color="grey", linestyle="--" )
plt.show()

# Displaying Prediction
print(f"Zehan will {'like' if predicted_likes[0] == 1 else 'not like'} a book of {predict_book_pages} pages.")