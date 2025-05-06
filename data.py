from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'] / 255.0  # 0~1 정규화
    y = mnist['target'].astype(np.int64)
    return X, y