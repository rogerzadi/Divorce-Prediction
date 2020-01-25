import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.model_selection import train_test_split

divorce = pd.read_csv('../data/divorce.csv',sep = ';')
divorce.head()

print(divorce.shape)
divorce.Class.value_counts()
for u in divorce.columns:
    print (divorce[u].value_counts())
    
from sklearn.neighbors import KNeighborsClassifier as KNNC

y=divorce.Class
X=divorce.drop(columns=['Class'])

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

knnc=KNNC(n_neighbors=3)
knnc.fit(X_train, y_train)
y_pred=knnc.predict(X_test)

from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score

print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))

from sklearn.svm import SVC

svc=SVC(probability=True).fit(X_train, y_train)

y_pred=svc.predict(X_test)
y_prob=svc.predict_proba(X_test)[::, -1]

print(list(y_prob))

from sklearn.ensemble import RandomForestClassifier as RFC
X_train1, X_test1, y_train1, y_test1=train_test_split(X, y, test_size=0.2)

rf=RFC()
rf.fit(X_train1, y_train1)
y_pred1=rf.predict(X_test1)

feats = {}
for feature, importance in zip(X_train1.columns, rf.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient="index").rename(
    columns={0: "importance"})

imp = importances.sort_values(by="importance", ascending=False)
print(accuracy_score(y_test1, y_pred1))
print(roc_auc_score(y_test1,y_pred1))
print(imp)
print(rf.feature_importances_)

corre=divorce.corr()

print(corre.Class.sort_values(ascending=False))

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print('Training set metrics:')
print('Accuracy:', accuracy_score(y_train, rf.predict(X_train)))
print('Precision:', precision_score(y_train, rf.predict(X_train)))
print('Recall:', recall_score(y_train, rf.predict(X_train)))

print('Test set metrics:')
print('Accuracy:', accuracy_score(y_test, rf.predict(X_test)))
print('Precision:', precision_score(y_test, rf.predict(X_test)))
print('Recall:', recall_score(y_test, rf.predict(X_test)))
