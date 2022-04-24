import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

regressor = ""
featureset = 9
if featureset == 4:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)']
if featureset == 5:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'LUMO (eV)']
if featureset == 6:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'Band Gap']
if featureset == 7:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'EA (eV)']
if featureset == 8:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV) (est)', 'LUMO (eV) (est)', 'Band Gap (est)', 'EA (eV)']
if featureset == 9:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'LUMO (eV)', 'Band Gap', 'EA (eV)']

# molecule_train = pd.read_excel("../01_Dataset/preprocessed.xlsx", sheet_name="Train")['Structure']
# train = pd.read_excel("../01_Dataset/preprocessed.xlsx", sheet_name="Train")[features]
# molecule_test = pd.read_excel("../01_Dataset/preprocessed.xlsx", sheet_name="Test")['Structure']
# test = pd.read_excel("../01_Dataset/preprocessed.xlsx", sheet_name="Test")[features]

molecule_train = pd.read_excel("../../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")['Structure']
train = pd.read_excel("../../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")[features]
molecule_test = pd.read_excel("../../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")['Structure']
test = pd.read_excel("../../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")[features]


#specify feature column names
feature_cols = train.columns[:-1]
#print(feature_cols)
feature_names = train.columns.values
feature_out = train.columns[-1]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train[feature_out]

X_test = test.loc[:, feature_cols]
y_test = test[feature_out]

#normalizing 
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

if regressor == "MLP":
    tuned_parameters = {"hidden_layer_sizes": [(50,),(100,)], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.0001,0.05], 'learning_rate': ['constant', 'adaptive'],}

    clf = GridSearchCV(MLPRegressor(), tuned_parameters, n_jobs= 4, cv=5)
elif regressor == "SVR":
    tuned_parameters = {'kernel' : ('poly', 'rbf', 'sigmoid'), 'C' : [1,5,10], 'degree' : [3,8], 'coef0' : [0.01,10,0.5], 'gamma' : ('auto','scale'),}

    clf = GridSearchCV(SVR(), tuned_parameters, n_jobs= 2, cv=5)
else:
    # tuned_parameters = [{'kernel':["linear","rbf"],'alpha': [0.01]}]
    tuned_parameters = [{'kernel':["linear","rbf"],'alpha': np.logspace(-3,0,100)}]

    clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=5)

clf.fit(X_train, y_train)

print(clf.best_params_)

#prediction
y_pred_train = clf.predict(X_train)
print("---MSE+R2+%error of Training Set---")
print(mean_squared_error(y_train, y_pred_train))
print(r2_score(y_train, y_pred_train))
print(mean_absolute_percentage_error(y_train, y_pred_train))
#np.savetxt("res/krr_train_pred.csv", y_pred_train, delimiter=",")
df_train = pd.DataFrame(data = {'molecule': molecule_train, 'ML' + str(featureset):y_pred_train, features[-1] +  "(est)": y_train})
with pd.ExcelWriter(str(featureset) + '.xlsx', engine="openpyxl") as writer:  
    df_train.to_excel(writer, sheet_name='Train')

y_pred = clf.predict(X_test)
print("---MSE+R2+%error of Testing Set---")
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
print(mean_absolute_percentage_error(y_test, y_pred))
#np.savetxt("res/krr_test_pred.csv", y_pred, delimiter=",")
df_test = pd.DataFrame(data = {'molecule': molecule_test, 'ML' + str(featureset):y_pred, features[-1] + " (est)": y_test})
with pd.ExcelWriter(str(featureset) + '.xlsx', engine="openpyxl", mode='a') as writer: 
    df_test.to_excel(writer, sheet_name = "Test")