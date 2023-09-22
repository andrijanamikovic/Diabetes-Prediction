import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

translate_dict_columns = {'gender': 'Пол', 'age': 'Године', 'hypertension': 'Хипертензија',
                          'heart_disease': 'Болести-срца', 'smoking_history': 'Историја-пушења',
                          'bmi': 'Индекс-телесне-масе', 'HbA1c_level': 'HbA1c-ниво',
                          'blood_glucose_level': 'Ниво-глукозе', 'diabetes': 'Дијабетес'}

translate_dict_gender = {'Female': 'Женски', 'Male': 'Мушки'}
translate_dict_smoking = {'never': 'Никад', 'No Info': 'Без информације', 'current': 'Тренутно', 'ever': 'Претходни',
                          'former': 'Претходни', 'not current': 'Претходни'}
data = pd.read_csv('data/diabetes_prediction_dataset.csv')
data.rename(columns=translate_dict_columns, inplace=True)
data['Пол'] = data['Пол'].replace(translate_dict_gender)
data['Историја-пушења'] = data['Историја-пушења'].replace(translate_dict_smoking)

prima = pd.read_csv('data/prima.csv')
prima.info()
print(prima.head())


diabetes_data = data[data['Дијабетес'] == 1]
no_diabetes_data = data[data['Дијабетес'] == 0]
print(len(diabetes_data))
print(len(no_diabetes_data))
no_diabetes_data_random = no_diabetes_data.sample(11000, random_state=42)
data_merged = pd.concat([no_diabetes_data_random, diabetes_data], axis=0)
data_merged = data_merged.sample(frac=1, random_state=42)

diabetes_data = data_merged[data_merged['Дијабетес'] == 1]
no_diabetes_data = data_merged[data_merged['Дијабетес'] == 0]
print(len(diabetes_data))
print(len(no_diabetes_data))

# Data splitting
X = data_merged.drop('Дијабетес', axis=1)
Y = data_merged['Дијабетес']
X1 = data.drop('Дијабетес', axis=1)
Y1 = data['Дијабетес']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)
print(X1_train.shape, X1_test.shape)


# Data encoding
def encode_gender(label):
    if label == 'Женски':
        return 0
    else:
        return 1


def encode_smoking(label):
    if label == 'Без информације':
        return 0
    elif label == 'Никад':
        return 1
    elif label == 'Претходни':
        return 2
    elif label == 'Тренутно':
        return 3
    else:
        return 4


X_train['Пол'] = X_train['Пол'].apply(encode_gender)
X_test['Пол'] = X_test['Пол'].apply(encode_gender)

X_train['Историја-пушења'] = X_train['Историја-пушења'].apply(encode_smoking)
X_test['Историја-пушења'] = X_test['Историја-пушења'].apply(encode_smoking)


X_train.drop(columns=['Пол', 'Историја-пушења'], inplace=True)
X_test.drop(columns=['Пол', 'Историја-пушења'], inplace=True)

X1_train['Пол'] = X1_train['Пол'].apply(encode_gender)
X1_test['Пол'] = X1_test['Пол'].apply(encode_gender)

X1_train['Историја-пушења'] = X1_train['Историја-пушења'].apply(encode_smoking)
X1_test['Историја-пушења'] = X1_test['Историја-пушења'].apply(encode_smoking)


X1_train.drop(columns=['Пол', 'Историја-пушења'], inplace=True)
X1_test.drop(columns=['Пол', 'Историја-пушења'], inplace=True)

adasyn = ADASYN(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X1_train, Y1_train)
X1_train_resampled, y1_train_resampled = adasyn.fit_resample(X_train, Y_train)

print(X_train_resampled.shape)
print(y_train_resampled.shape)

algorithm_name = []
algorithm_score = []
def knn_algotiham_plot():
    knn_score_list = []
    for k in range(1, 20, 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X1_train_resampled, y1_train_resampled)  # ubaci u model
        prediction = knn.predict(X_test)
        knn_score_list.append(knn.score(X_test, Y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 20, 1), knn_score_list, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='green', markersize=10)
    plt.title('Прецизност КНН алгоритма за различите вредности параметра')
    plt.xlabel('K')
    plt.ylabel('Прецизност')
    plt.show()

def logistical_regression():
    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X_train, Y_train)
    algorithm_name.append('Логистичка регресија')
   # algorithm_score.append(ceil(log_reg.score(X_test, Y_test)*100))
    y_pred = log_reg.predict(X_test)
    algorithm_score.append(accuracy_score(Y_test, y_pred)*100)
    print(classification_report(Y_test, y_pred))

def knn_algoritham():
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X1_train_resampled, y1_train_resampled)
    algorithm_name.append('КНН')
   # algorithm_score.append(ceil(knn.score(X_test, Y_test)*100))
    y_pred = knn.predict(X_test)
    algorithm_score.append(accuracy_score(Y_test, y_pred)*100)
    print(classification_report(Y_test, y_pred))

def decision_tree():
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train_resampled, y_train_resampled)
    algorithm_name.append('Стабло одлучивања')
   # algorithm_score.append(ceil(dtree.score(X_test, Y_test)*100))
    y_pred = dtree.predict(X1_test)
    algorithm_score.append(accuracy_score(Y1_test, y_pred)*100)
    print(classification_report(Y1_test, y_pred))


#knn_algotiham_plot()
#print("Logisticka regresija: ")
#logistical_regression()
#print ("Knn algoritam: ")
#knn_algoritham()
#print("Stablo odlucivanja: ")
#decision_tree()

#print(algorithm_score)

plt.figure(figsize=(10, 5))
plt.bar(algorithm_name, algorithm_score, color='#87CEEB')
plt.ylim(80,100)
plt.title("Прецизност модела")
plt.xlabel("Модел")
plt.ylabel("Прецизност")


plt.show()
