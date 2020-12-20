import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics 

df=pd.read_csv("pima-indians-diabetes.csv")
df.head()
x=df.drop(['Outcome'], axis=1)
y=df.Outcome

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# print('Decision Tree Classifier')
# model = DecisionTreeClassifier()
print('RandomForestClassifier')
model = RandomForestClassifier(n_estimators=10, random_state=42,max_features="auto")
# print('KNeighborsClassifier')
# model = KNeighborsClassifier(n_neighbors=9)
# print('SVC')
# model = SVC(kernel="rbf",C=100)
# print('LogisticRegression')
# model = LogisticRegression(max_iter=15,solver = 'sag')


model = model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# print(model.predict([[6,148,72,35,0,33.6,0.627,50]]))
print(model.predict([[5,116,74,0,0,25.6,0.201,30]]))
print('-------------------')


