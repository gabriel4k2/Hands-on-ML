from sklearn.datasets import fetch_openml


def fetch_data_sklearn_repo(name, return_as_frame=False):
    return fetch_openml(name, version=1, as_frame=return_as_frame)
