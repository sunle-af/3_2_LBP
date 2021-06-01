import numpy as np
import imageio
from PIL import Image
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def get_sar_data(stage, width=50, height=50):
    data_dir = "data/train/" if stage == "train" else "data/test/" if stage == "test" else None
    print("------ " + stage + " ------")
    sub_dir = ["URBAN","RAIL"]
    X = []
    y = []

    for i in range(len(sub_dir)):
        tmp_dir = data_dir + sub_dir[i] + "/"
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".png")]
        print(sub_dir[i], len(img_idx))
        y += [i] * len(img_idx)
        for j in range(len(img_idx)):
            img = np.array(Image.fromarray(imageio.imread((tmp_dir + img_idx[j]))).resize(size=(height,width)))
            img = img[:,:,0]
            X.append(img)

    return np.asarray(X), np.asarray(y)

def data_shuffle(X, y, seed=0):
    data = np.hstack([X, y[:, np.newaxis]])
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]

def mean_wise(X):
    return (X.T - np.mean(X, axis=1)).T

#X stores the image data and y stores the class of the image
X_train, y_train = get_sar_data("train", 50, 50)
X_test, y_test = get_sar_data("test", 50, 50)

# Flattening the image
X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1] * X_train.shape[2]])
X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1] * X_test.shape[2]])

X_train, y_train = data_shuffle(X_train, y_train)
X_test, y_test = data_shuffle(X_test, y_test)

#Feature Scaling
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = mean_wise(X_train)
X_test = mean_wise(X_test)



#Any one of the following 8 models can be chosen

classifier = GaussianNB().fit(X_train,y_train)
#classifier = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.005, \
#                                       max_features="sqrt", max_depth=None, random_state=0).fit(X_train, y_train)
#classifier = SVC(C=1.0, kernel="rbf", max_iter=-1, random_state=0).fit(X_train, y_train)
#classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0).fit(X_train,y_train)
#classifier = KNeighborsClassifier(n_neighbors=10, weights="uniform", algorithm="auto").fit(X_train, y_train)
#classifier = DecisionTreeClassifier(criterion="entropy", max_features=0.8, max_depth=None, random_state=0).fit(X_train, y_train)
#classifier = RandomForestClassifier(n_estimators=1000, max_features="sqrt", min_samples_split=2, \
#                                   max_depth=None, bootstrap=True, oob_score=False, random_state=0, n_jobs=4).fit(X_train, y_train)
#classifier = MLPClassifier(hidden_layer_sizes=1000, activation="logistic", solver="sgd", batch_size=32, \
#                          learning_rate="constant", learning_rate_init=0.1, early_stopping=False, max_iter=1000, random_state=0).fit(X_train, y_train)


print('The accuracy of the classifier on training data is ',classifier.score(X_train, y_train))
print('The accuracy of the classifier on testing data is ',classifier.score(X_test, y_test))
