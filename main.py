import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#pandas display optionos
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
# colors = [(135, 206, 250), 	(69, 176, 140)]
colors = ['#87CEEB','#f28500', '#45B08C', '#ff2400']

translate_dict_columns = {'gender': 'Пол', 'age': 'Године', 'hypertension' : 'Хипертензија', 'heart_disease':'Болести-срца', 'smoking_history' : 'Историја-пушења',
                  'bmi': 'Индекс-телесне-масе','HbA1c_level' : 'HbA1c-ниво' , 'blood_glucose_level' : 'Ниво-глукозе' , 'diabetes': 'Дијабетес'}

translate_dict_gender = {'Female': 'Женски', 'Male' : 'Мушки'}
translate_dict_smoking = {'never' : 'Никад', 'No Info' : 'Без информације', 'current': 'Тренутно', 'ever': 'Претходни', 'former': 'Претходни', 'not current' : 'Претходни'}
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


#Pol
gender_to_include = ['Женски', 'Мушки']
plt.figure(figsize=(5,3))
sns.set(font_scale=2)
sns.countplot(x='Пол',data=data, palette= colors, order=gender_to_include)
plt.ylabel('Број')
plt.title("Родне карактеристике")
# plt.show()


plt.figure(figsize=(5,3))
sns.set(font_scale=2)
sns.countplot(x='Пол',data=diabetes_data,  palette= colors)
plt.title("Заступљеност пацијената са дијабетесом на основу пола")
plt.ylabel('Број')
# plt.show()

#Samo godine
#plt.figure(figsize=(5,3))
#plt.hist(data['Године'], color= '#87CEEB', label='Сви')
#plt.title('Дистрибуција година')
#plt.xlabel('Године')
#plt.ylabel('Фреквенција')

#Godine
plt.figure(figsize=(5,3))
plt.hist(data['Године'], color= '#87CEEB', label='Сви')
plt.hist(diabetes_data['Године'], color= '#f28500', label='Са дијабетесом')
plt.title('Дистрибуција година и приказ заступљености дијабетеса у различитим узрастима')
plt.xlabel('Године')
plt.ylabel('Фреквенција')


#Hipertenzija

labels = ['Без хипертензије', 'Хипертензија']

#hypertension_counts = data['Хипертензија'].value_counts()
#hypertension_percentage = hypertension_counts / len(data) * 100
#plt.figure(figsize=(10,6))
#plt.pie(hypertension_percentage, labels=labels, autopct='%1.1f%%', colors= colors)
#plt.title('Проценат пацијената са Хипертензијом')


hypertension_counts = diabetes_data['Хипертензија'].value_counts()
hypertension_percentage = hypertension_counts / len(diabetes_data) * 100
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].pie(hypertension_percentage, labels=labels, autopct='%1.1f%%',startangle=90, colors= colors ,  textprops={'fontsize': 10})
axes[0].set_title('Проценат пацијената са хипертензијом и дијабетесом', fontsize= 12)

hypertension_counts = no_diabetes_data['Хипертензија'].value_counts()
hypertension_percentage = hypertension_counts / len(no_diabetes_data) * 100
axes[1].pie(hypertension_percentage, labels=labels, autopct='%1.1f%%',startangle=90, colors= colors,  textprops={'fontsize': 10})
axes[1].set_title('Проценат пацијената са хипертензијом, а без дијабетеса', fontsize= 12)

#Bolesti-srca
labels = ['Са болестима срца', 'Здраво срце']
hear_disease_counts = diabetes_data['Болести-срца'].value_counts()
hear_disease_percentage = hear_disease_counts / len(diabetes_data) * 100
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].pie(hear_disease_percentage, labels=labels, autopct='%1.1f%%',startangle=90, colors= colors ,  textprops={'fontsize': 10})
axes[0].set_title('Проценат пацијената са болестима срца и дијабетесом', fontsize= 12)

hear_disease_counts = no_diabetes_data['Болести-срца'].value_counts()
hear_disease_percentage = hear_disease_counts / len(no_diabetes_data) * 100
axes[1].pie(hear_disease_percentage, labels=labels, autopct='%1.1f%%',startangle=90, colors= colors,  textprops={'fontsize': 10})
axes[1].set_title('Проценат пацијената са болестима срца, а без дијабетеса', fontsize= 12)


#Pusenje
smoking_data_group = data.groupby('Историја-пушења')['Дијабетес'].mean()
plt.figure(figsize=(10,5))
plt.bar(smoking_data_group.index, smoking_data_group.values,  color= '#87CEEB')
plt.title('Пропорција пацијената са дијабетесом у односу на историју пушења')
plt.xlabel('Историја пушења')
plt.ylabel('Проценат пацијената са дијабетесом')

#BMI

plt.figure(figsize=(5,3))
plt.hist(data['Индекс-телесне-масе'], color= '#87CEEB', label='Сви')
plt.hist(diabetes_data['Индекс-телесне-масе'], color= '#f28500', label='Са дијабетесом')
plt.title('Дистрибуција индекса телесне масе и приказ заступљености дијабетеса')
plt.xlabel('Индекс-телесне-масе')
plt.ylabel('Фреквенција')

#Hb1Ac to moram drugacije
fig, axes = plt.subplots(1, 2, figsize=(10, 7))

axes[0].hist(diabetes_data['HbA1c-ниво'], color= '#f28500', label='Са дијабетесом')
axes[0].set_title('Дистрибуција HbA1c-нивa у пацијентима са дијабетесом', fontsize= 12)
axes[0].set_xlabel('HbA1c-ниво')
axes[0].set_ylabel('Фреквенција')

axes[1].hist(data['HbA1c-ниво'], color= '#f28500', label='Без дијабетеса')
axes[1].set_title('Дистрибуција HbA1c-нивa у пацијентима без дијабетеса', fontsize= 12)
axes[1].set_xlabel('HbA1c-ниво')
axes[1].set_ylabel('Фреквенција')

plt.show()


