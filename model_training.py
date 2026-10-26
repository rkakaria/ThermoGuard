import pandas as pd 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle

# load, clean, and format data - remove timestamps
raw_data = pd.read_csv("Rotating_equipment_fault_data.csv")
dataset = raw_data.dropna(axis=0, how='any') 
dataset = dataset.drop(['Timestamp'], axis = 1)

# define data and target columns
x = dataset.drop(['Fault_Type'], axis = 1) # data which informs the target value
y = dataset['Fault_Type'] # target value; stratify based on this column

# create pipeline to scale data and select model type
model = make_pipeline(StandardScaler(), GaussianNB())

# perform stratified cross-validation and determine accuracy
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, x, y, cv = cv)

print("CV Scores:", scores)
print("Mean Score:", scores.mean())

# save model
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))