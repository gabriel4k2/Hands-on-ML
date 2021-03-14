from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def fetch_data_sklearn_repo(name, return_as_frame=False):
    return fetch_openml(name, version=1, as_frame=return_as_frame)


#Given an array and its index it show the digit (deals with numpy and pandas' dataframe heterogenity)
def show_a_digit_index(arr, index, is_pandas=False):
    if is_pandas: #If is a pandas dataframe
        this_digit = arr.iloc[index]
        this_digit_image = this_digit.values.reshape(28, 28)
    else:
        this_digit = arr[index]
        this_digit_image = this_digit.reshape(28, 28)

    plt.imshow(this_digit_image, cmap='binary')
    plt.axis('off')

#Given an object simply show it
def show_a_digit_obj(image):
    plt.imshow(image, cmap='binary')
    plt.axis('off')
