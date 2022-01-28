import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from process_data import plot_heatmap, plot_cost, plot_result, plot_single, process_data
from MLR import gradient_descent, predict, r2score

df = pd.read_csv("life_expectancy.csv")
Y_VALUE = "Life expectancy "

#plot_heatmap(df)
#plot_single(df, Y_VALUE)

X, y = process_data(df, Y_VALUE)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.95, random_state=42) #random state = the same split every run

w, b, cost_list, epoch_list = gradient_descent(x=X_train, y=Y_train, w=np.zeros(X_train.shape[1]), b=0, learning_rate=0.001, epochs=50000)
y_pred = predict(X_test, w, b)

plot_result(X, y, X_test, y_pred)
plot_cost(epoch_list, cost_list)
# print("Algorithm score is: ", r2score(y_pred, Y_test))