import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from math import ceil

# pandas display optionos
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
# colors = [(135, 206, 250), 	(69, 176, 140)]
colors = ['#87CEEB', '#f28500', '#45B08C', '#ff2400']

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

diabetes_data = data[data['Дијабетес'] == 1]
no_diabetes_data = data[data['Дијабетес'] == 0]

data.info()

null_counts = data.isnull().sum()
print(null_counts)

print(data.head())

# Pol
gender_to_include = ['Женски', 'Мушки']
plt.figure(figsize=(5, 3))
sns.set(font_scale=2)
sns.countplot(x='Пол', data=data, palette=colors, order=gender_to_include)
plt.ylabel('Број')
plt.title("Родне карактеристике")
# plt.show()


plt.figure(figsize=(5, 3))
sns.set(font_scale=2)
sns.countplot(x='Пол', data=diabetes_data, palette=colors)
plt.title("Заступљеност пацијената са дијабетесом на основу пола")
plt.ylabel('Број')
# plt.show()

# Samo godine
# plt.figure(figsize=(5,3))
# plt.hist(data['Године'], color= '#87CEEB', label='Сви')
# plt.title('Дистрибуција година')
# plt.xlabel('Године')
# plt.ylabel('Фреквенција')

# Godine
plt.figure(figsize=(5, 3))
plt.hist(data['Године'], color='#87CEEB', label='Сви')
plt.hist(diabetes_data['Године'], color='#f28500', label='Са дијабетесом')
plt.title('Дистрибуција година и приказ заступљености дијабетеса у различитим узрастима')
plt.xlabel('Године')
plt.ylabel('Фреквенција')

# Hipertenzija

labels = ['Без хипертензије', 'Хипертензија']

# hypertension_counts = data['Хипертензија'].value_counts()
# hypertension_percentage = hypertension_counts / len(data) * 100
# plt.figure(figsize=(10,6))
# plt.pie(hypertension_percentage, labels=labels, autopct='%1.1f%%', colors= colors)
# plt.title('Проценат пацијената са Хипертензијом')


hypertension_counts = diabetes_data['Хипертензија'].value_counts()
hypertension_percentage = hypertension_counts / len(diabetes_data) * 100
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].pie(hypertension_percentage, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
            textprops={'fontsize': 10})
axes[0].set_title('Проценат пацијената са хипертензијом и дијабетесом', fontsize=12)

hypertension_counts = no_diabetes_data['Хипертензија'].value_counts()
hypertension_percentage = hypertension_counts / len(no_diabetes_data) * 100
axes[1].pie(hypertension_percentage, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
            textprops={'fontsize': 10})
axes[1].set_title('Проценат пацијената са хипертензијом, а без дијабетеса', fontsize=12)

# Bolesti-srca
labels = ['Са болестима срца', 'Здраво срце']
hear_disease_counts = diabetes_data['Болести-срца'].value_counts()
hear_disease_percentage = hear_disease_counts / len(diabetes_data) * 100
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].pie(hear_disease_percentage, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
            textprops={'fontsize': 10})
axes[0].set_title('Проценат пацијената са болестима срца и дијабетесом', fontsize=12)

hear_disease_counts = no_diabetes_data['Болести-срца'].value_counts()
hear_disease_percentage = hear_disease_counts / len(no_diabetes_data) * 100
axes[1].pie(hear_disease_percentage, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
            textprops={'fontsize': 10})
axes[1].set_title('Проценат пацијената са болестима срца, а без дијабетеса', fontsize=12)

# Pusenje
smoking_data_group = data.groupby('Историја-пушења')['Дијабетес'].mean()
plt.figure(figsize=(10, 5))
plt.bar(smoking_data_group.index, smoking_data_group.values, color='#87CEEB')
plt.title('Пропорција пацијената са дијабетесом у односу на историју пушења')
plt.xlabel('Историја пушења')
plt.ylabel('Проценат пацијената са дијабетесом')

# BMI

plt.figure(figsize=(5, 3))
plt.hist(data['Индекс-телесне-масе'], color='#87CEEB', label='Сви')
plt.hist(diabetes_data['Индекс-телесне-масе'], color='#f28500', label='Са дијабетесом')
plt.title('Дистрибуција индекса телесне масе и приказ заступљености дијабетеса')
plt.xlabel('Индекс-телесне-масе')
plt.ylabel('Фреквенција')

# Hb1Ac to moram drugacije
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.subplots_adjust(wspace=0.3)

axes[0].hist(diabetes_data['HbA1c-ниво'], color='#87CEEB', label='Са дијабетесом')
axes[0].set_title('Дистрибуција HbA1c-нивoa у пацијентима са дијабетесом', fontsize=12)
axes[0].set_xlabel('HbA1c-ниво')
axes[0].set_ylabel('Фреквенција')

axes[1].hist(data['HbA1c-ниво'], color='#87CEEB', label='Без дијабетеса')
axes[1].set_title('Дистрибуција HbA1c-нивoa у пацијентима без дијабетеса', fontsize=12)
axes[1].set_xlabel('HbA1c-ниво')
axes[1].set_ylabel('Фреквенција')

# Nivo-glukoze
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.subplots_adjust(wspace=0.3)

axes[0].hist(diabetes_data['Ниво-глукозе'], color='#87CEEB', label='Са дијабетесом')
axes[0].set_title('Дистрибуција нивоа глукозе у пацијентима са дијабетесом', fontsize=12)
axes[0].set_xlabel('Ниво-глукозе')
axes[0].set_ylabel('Фреквенција')

axes[1].hist(data['HbA1c-ниво'], color='#87CEEB', label='Без дијабетеса')
axes[1].set_title('Дистрибуција нивоа глукозе у пацијентима без дијабетеса', fontsize=12)
axes[1].set_xlabel('Ниво-глукозе')
axes[1].set_ylabel('Фреквенција')

# plt.show()

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


data['Пол'] = data['Пол'].apply(encode_gender)
data['Историја-пушења'] = data['Историја-пушења'].apply(encode_smoking)

# Data splitting
X = data.drop('Дијабетес', axis=1)
Y = data['Дијабетес']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape)

# print(X_train.head())
# print(X_test.head())


plt.figure(figsize=(10, 6))
plt.title('Корелација између атрибута')
ax = sns.heatmap(X_train.corr(), annot=True, cmap='tab20c', fmt='.2f', linewidths=0.2)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
plt.xticks(rotation=15)

corr = data.corr()
target_corr = corr['Дијабетес'].drop('Дијабетес')
target_corr_sorted = target_corr.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.set(font_scale=0.8)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr_sorted.to_frame(), cmap="tab20c", annot=True, fmt='.2f')
plt.title('Корелација са дијабетесом')

X_train.drop(columns=['Пол', 'Историја-пушења', 'Болести-срца'], inplace=True)
X_test.drop(columns=['Пол', 'Историја-пушења', 'Болести-срца'], inplace=True)

# k = np.sqrt(len(data)).astype(int)
# if (k % 2) == 0:
#    k = k + 1
# print(k)
algorithm_name = []
algorithm_score = []
def knn_algotiham_plot():
    knn_score_list = []
    for k in range(1, 20, 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)  # ubaci u model
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
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, Y_train)
    algorithm_name.append('КНН')
   # algorithm_score.append(ceil(knn.score(X_test, Y_test)*100))
    y_pred = knn.predict(X_test)
    algorithm_score.append(accuracy_score(Y_test, y_pred)*100)
    print(classification_report(Y_test, y_pred))

def decision_tree():
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, Y_train)
    algorithm_name.append('Стабло одлучивања')
   # algorithm_score.append(ceil(dtree.score(X_test, Y_test)*100))
    y_pred = dtree.predict(X_test)
    algorithm_score.append(accuracy_score(Y_test, y_pred)*100)
    print(classification_report(Y_test, y_pred))


#knn_algotiham_plot()
print("Logisticka regresija: ")
logistical_regression()
print ("Knn algoritam: ")
knn_algoritham()
print("Stablo odlucivanja: ")
decision_tree()

print(algorithm_score)

plt.figure(figsize=(10, 5))
plt.bar(algorithm_name, algorithm_score, color='#87CEEB')
plt.ylim(90,100)
plt.title("Прецизност модела")
plt.xlabel("Модел")
plt.ylabel("Прецизност")
plt.show()
