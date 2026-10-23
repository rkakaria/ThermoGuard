import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

# load, clean, and format data - remove model numbers from laptops as they are all unique
raw_data = pd.read_csv("motherboard.csv")
dataset = raw_data.dropna(axis=0, how='any') 
dataset['ModelName'] = dataset.apply(lambda x: x['ModelName'][:-5], axis = 1)

# define data and target columns
x = dataset.drop(['ProblemDetected'], axis = 1) # data which informs the target value
y = dataset['ProblemDetected'] # target value; stratify based on this column

'''
Create a training, testing, and validation dataset (70/15/15 split).
Results in a training set of 1400 lines and a testing and validation set of 300 lines each.
'''

# split into train and temporary dataframes
x_train, x_temp, y_train, y_temp = train_test_split(x, y, stratify = y, test_size = (0.3), random_state = 1)

# split into test and validate dataframes
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, stratify = y_temp, test_size = (0.5), random_state = 1)

# encode categorical data using OneHot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown = 'ignore').set_output(transform = "pandas")

train_cat = x_train.select_dtypes(exclude = 'number').columns
train_encoded = encoder.fit_transform(x_train[train_cat])
encoded_df = pd.DataFrame(train_encoded, columns = encoder.get_feature_names_out(train_cat))
x_train = x_train.drop(['ModelName'], axis = 1)
x_train = pd.DataFrame(pd.concat([x_train, encoded_df], axis = 1))

test_cat = x_test.select_dtypes(exclude = 'number').columns
test_encoded = encoder.transform(x_test[test_cat])
encoded_df = pd.DataFrame(test_encoded, columns = encoder.get_feature_names_out(test_cat))
x_test = x_test.drop(['ModelName'], axis = 1)
x_test = pd.DataFrame(pd.concat([x_test, encoded_df], axis = 1))

val_cat = x_val.select_dtypes(exclude = 'number').columns
val_encoded = encoder.transform(x_val[val_cat])
encoded_df = pd.DataFrame(val_encoded, columns = encoder.get_feature_names_out(val_cat))
x_val = x_val.drop(['ModelName'], axis = 1)
x_val = pd.DataFrame(pd.concat([x_val, encoded_df], axis = 1))

# scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# encode y-axis using Label Binarizer
# lb = LabelBinarizer()

# y_train = pd.DataFrame(lb.fit_transform(y_train), index = y_train.index)
# y_test = pd.DataFrame(lb.fit_transform(y_test), index = y_test.index)
# y_val = pd.DataFrame(lb.fit_transform(y_val), index = y_val.index)

# train model

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

clf = OneVsRestClassifier(LinearSVC(random_state=None)).fit(x_train, y_train)
y_pred = clf.predict(x_test)


from sklearn import metrics
print("OneVsRest Accuracy:", metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth=2, random_state=0)
dtree.fit(x_train, y_train)
dtree_preds = dtree.predict(x_test)
dtree_acc = accuracy_score(y_test, dtree_preds)

print("Decision Tree Accuracy:", dtree_acc)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
knn_preds = knn.predict(x_test)
knn_acc = accuracy_score(y_test, knn_preds)

print("KNN Accuracy:", knn_acc)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
nb_preds = nb.predict(x_test)
nb_acc = accuracy_score(y_test, nb_preds)

print("Naive Bayes Accuracy:", nb_acc)