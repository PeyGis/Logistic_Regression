import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



##Import helper modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


##Import custom models
from model.column_selector import ColumnsSelector
from model.categorical_encoder import CategoricalEncoder
from model.categorical_pipeline import CategoricalImputer



##Load dataset
columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", 
"occupation", "relationship","race", "sex", "capital-gain", "capital-loss", 
"hours-per-week", "native-country", "income"]

training_data = pd.read_csv('data/adult.data', names=columns, sep=' *, *', na_values='?', engine='python')
testing_data = pd.read_csv('data/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')

print(testing_data.info())

##Handling Numerical variables

numerical_data = training_data.select_dtypes(include=['int64'])
#numerical_data.hist()
#plt.show()


##Handleing Categorical variables

categorical_data = training_data.select_dtypes(include=['object'])
#print(categorical_data.describe())

sns.countplot(y='workClass', hue='income', data = categorical_data)
sns.countplot(y='occupation', hue='income', data = categorical_data)
#plt.show()


#Numerical Data Pipeline, include scaling the numerical entries

numerical_pipeline = Pipeline(steps=[
    ("num_attr_selector", ColumnsSelector(type='int64')),
    ("scaler", StandardScaler())
])

##Categorical pipeline
categorical_pipeline = Pipeline(steps=[
    ("cat_attr_selector", ColumnsSelector(type='object')),
    ("cat_imputer", CategoricalImputer(columns=
          ['workClass','occupation', 'native-country'])),
    ("encoder", CategoricalEncoder(dropFirst=True))
])

#Mergin full pipeline
full_pipeline = FeatureUnion([("numerical_pipeline", numerical_pipeline), 
                ("categorical_pipeline", categorical_pipeline)])

#Drop columns we dont need
training_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
testing_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)

#Training the model by creating a copy
train_copy = training_data.copy()
train_copy["income"] = train_copy["income"].apply(lambda x:0 if x=='<=50K' else 1)
X_train = train_copy.drop('income', axis =1)
Y_train = train_copy['income']


#Pass the training dataset to the full pipeline and train the model
X_train_processed = full_pipeline.fit_transform(X_train)
model = LogisticRegression(random_state=0)
model.fit(X_train_processed, Y_train)

#Test the data
test_copy = testing_data.copy()
test_copy["income"] = test_copy["income"].apply(lambda x:0 if x=='<=50K.' else 1)
X_test = test_copy.drop('income', axis =1)
Y_test = test_copy['income']


#Transform the test data too
X_test_processed = full_pipeline.transform(X_test)

predicted_classes = model.predict(X_test_processed)

accuracy = accuracy_score(Y_test,predicted_classes)
parameters = model.coef_
print(parameters)
print(accuracy)

#Plot a confussion matrix with seaborn
cfm = confusion_matrix(predicted_classes, Y_test.values)
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
#plt.show()


##Save the mode
file_name = 'logistic_regression.sav'
pk.dump(model, open(file_name, 'wb'))

##Load the model

new_model = pk.load(open(file_name, 'rb'))

predicted_saved = new_model.predict(X_test_processed)

accuracy_2 = accuracy_score(Y_test, predicted_saved)
print(accuracy_2)