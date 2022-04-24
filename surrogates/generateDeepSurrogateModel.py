import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

featureset = 14
if featureset ==1:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'RP (V) - DFT']
if featureset ==2:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV) (est)', 'RP (V) - DFT']
if featureset ==3:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'LUMO (eV) (est)', 'RP (V) - DFT']
if featureset ==4:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'Band Gap (est)', 'RP (V) - DFT']
if featureset ==5:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV) (est)', 'LUMO (eV) (est)', 'Band Gap (est)', 'RP (V) - DFT']
if featureset ==6:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'EA (eV) (est)', 'RP (V) - DFT']
if featureset ==7:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'EA (eV) (est2)', 'RP (V) - DFT']
if featureset ==8:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV) (est)', 'LUMO (eV) (est)', 'Band Gap (est)', 'EA (eV) (est)', 'EA (eV) (est2)', 'EA (eV) (est3)', 'RP (V) - DFT']
if featureset ==9:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'RP (V) - DFT']
if featureset ==10:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'LUMO (eV)', 'RP (V) - DFT']
if featureset ==11:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'Band Gap', 'RP (V) - DFT']
if featureset ==12:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'LUMO (eV)', 'Band Gap', 'RP (V) - DFT']
if featureset ==13:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'EA (eV)', 'RP (V) - DFT']
if featureset ==14:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'LUMO (eV)', 'Band Gap', 'EA (eV)', 'RP (V) - DFT']
if featureset ==15:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'EA (eV) (est3)', 'RP (V) - DFT']
if featureset ==16:
    features = ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'LUMO (eV)', 'Band Gap', 'EA (eV) (est)', 'RP (V) - DFT']

# molecule_train = pd.read_excel("../01_Dataset/preprocessed.xlsx", sheet_name="Train")['Structure']
# train = pd.read_excel("../01_Dataset/preprocessed.xlsx", sheet_name="Train")[features]
# molecule_test = pd.read_excel("../01_Dataset/preprocessed.xlsx", sheet_name="Test")['Structure']
# test = pd.read_excel("../01_Dataset/preprocessed.xlsx", sheet_name="Test")[features]

molecule_train = pd.read_excel("../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")['Structure']
train = pd.read_excel("../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")[features]
molecule_test = pd.read_excel("../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")['Structure']
test = pd.read_excel("../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")[features]

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

tuned_parameters = [{'kernel':["linear","rbf"],'alpha': [0.01]}]
#tuned_parameters = [{'kernel':["linear","rbf"],'alpha': np.logspace(-3,0,100)}]

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
df_train = pd.DataFrame(data = {'molecule': molecule_train, 'ML' + str(featureset):y_pred_train, 'RP (V) - DFT': y_train})
with pd.ExcelWriter('ML' + str(featureset) + '.xlsx', engine="openpyxl") as writer:  
    df_train.to_excel(writer, sheet_name='Train')

y_pred = clf.predict(X_test)
print("---MSE+R2+%error of Testing Set---")
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
print(mean_absolute_percentage_error(y_test, y_pred))
#np.savetxt("res/krr_test_pred.csv", y_pred, delimiter=",")
df_test = pd.DataFrame(data = {'molecule': molecule_test, 'ML' + str(featureset):y_pred, 'RP (V) - DFT': y_test})
with pd.ExcelWriter('ML' + str(featureset) + '.xlsx', engine="openpyxl", mode='a') as writer: 
    df_test.to_excel(writer, sheet_name = "Test")