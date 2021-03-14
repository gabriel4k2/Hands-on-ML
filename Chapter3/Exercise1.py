from utils import *
from sklearn.model_selection import StratifiedShuffleSplit

if __name__ == '__main__':
    mnist_data = fetch_data_sklearn_repo(name='mnist_784')
    X = mnist_data['data']
    Y= mnist_data['target']
    # X_train, Y_train, X_test, Y_test = X[:60000], Y[:60000],
    print(X.shape)
             