import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

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
encoder = OneHotEncoder(sparse_output=False).set_output(transform = "pandas")

train_cat = x_train.select_dtypes(exclude = 'number').columns
train_encoded = encoder.fit_transform(x_train[train_cat])
encoded_df = pd.DataFrame(train_encoded, columns = encoder.get_feature_names_out(train_cat))
x_train = x_train.drop(['ModelName'], axis = 1)
x_train = pd.concat([x_train, encoded_df], axis = 1)

test_cat = x_test.select_dtypes(exclude = 'number').columns
test_encoded = encoder.fit_transform(x_test[test_cat])
encoded_df = pd.DataFrame(test_encoded, columns = encoder.get_feature_names_out(test_cat))
x_test = x_test.drop(['ModelName'], axis = 1)
x_test = pd.concat([x_test, encoded_df], axis = 1)

val_cat = x_val.select_dtypes(exclude = 'number').columns
val_encoded = encoder.fit_transform(x_val[val_cat])
encoded_df = pd.DataFrame(val_encoded, columns = encoder.get_feature_names_out(val_cat))
x_val = x_val.drop(['ModelName'], axis = 1)
x_val = pd.concat([x_val, encoded_df], axis = 1)

# encode y-axis using Label Binarizer
lb = LabelBinarizer()

y_train = pd.DataFrame(lb.fit_transform(y_train), index = y_train.index)
y_test = pd.DataFrame(lb.fit_transform(y_test), index = y_test.index)
y_val = pd.DataFrame(lb.fit_transform(y_val), index = y_val.index)