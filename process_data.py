import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

def plot_heatmap(df):
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True) #rectangular dataset, annot True -> write values
    plt.show()

def plot_single(df, y_col):
    plt.figure()
    j = 1
    for i in range(1, 19):
        plt.subplot(3, 1, j)
        sns.regplot(x=df.columns[i + 3], y=y_col , data=df) # plot life expectancy / other factors plots
        if j == 3: 
            plt.show()
            j = 0
        j+=1

def process_data(df, y_col):
    df = df.fillna(df.mean()) # fill na / nan values with mean value

    labelencoder = LabelEncoder()
    sc = StandardScaler() 

    df['Country'] = labelencoder.fit_transform(df['Country']) # encodes countires as numbers 0-192
    df['Status'] = labelencoder.fit_transform(df['Status']) # encodes status 0 / 1

    X = df.drop(y_col, axis=1) # splitting data into X and y
    X = sc.fit_transform(X) # scaling by removing the mean and dividing by standard deviation so that there's no feature bias
    y = df[y_col]

    #X = shuffle(X)

    return X, y

def plot_result(X, y, X_test, y_pred):
    plt.figure()
    plt.scatter(x=list(range(len(X))), y=y, color="blue")
    plt.scatter(x=list(range(len(X_test))), y=y_pred, color="red")
    plt.show()

def plot_cost(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='epochs', ylabel='cost',
            title='cost function')
    
    ax.grid()
    plt.show()