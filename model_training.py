import pandas as pd 
from sklearn.model_selection import train_test_split

# load and clean data
raw_data = pd.read_csv("motherboard.csv")
dataset = raw_data.dropna(axis=0, how='any') 

# define data and target columns
x = dataset.drop(['ProblemDetected'], axis = 1).values # data which informs the target value
y = dataset['ProblemDetected'] # target value; stratify based on this column

'''
Create a training, testing, and validation dataset (70/15/15 split).
Results in a training set of 1400 lines and a testing and validation set of 300 lines each.
'''

# split into train and temporary dataframes
x_train, x_temp, y_train, y_temp = train_test_split(x, y, stratify = y, test_size = (0.3), random_state = 1)

# split into test and validate dataframes
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, stratify = y_temp, test_size = (0.5), random_state = 1)