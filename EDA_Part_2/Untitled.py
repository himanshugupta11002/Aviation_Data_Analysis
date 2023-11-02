#!/usr/bin/env python
# coding: utf-8

# In[97]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.impute import KNNImputer
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier , AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score ,confusion_matrix, classification_report ,roc_curve, auc
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


# In[98]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
Data = [train,test]
data_names=['training data','testing data']

train.shape


# In[99]:


for x in Data:
  x.drop(['Unnamed: 0','id'],axis=1,inplace=True)
train.columns


# In[100]:


test.columns


# In[101]:


def calculate_missing_values(data):
    total_missing = data.isnull().sum()  # Count the missing values in each column
    percent_missing = round((total_missing / len(data)) * 100,3)  # Calculate the percentage

    missing_data = pd.DataFrame({
        'Total Missing': total_missing,
        'Percent Missing': percent_missing
    })

    missing_data = missing_data.sort_values(by='Percent Missing', ascending=False)
    return missing_data


# In[102]:


calculate_missing_values(train)


# In[103]:


calculate_missing_values(train)


# In[104]:


calculate_missing_values(test)


# In[105]:


sns.set_style("darkgrid")


train_null_counts = train.isnull().sum()
test_null_counts = test.isnull().sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.barplot(x=train_null_counts, y=train_null_counts.index, ax=axes[0], color='skyblue')
axes[0].set_title('Null Values in Train Data', fontweight='bold')
axes[0].set_xlabel('Count of Null Values', fontweight='bold')
axes[0].set_ylabel('Columns', fontweight='bold')
axes[0].grid(axis='x', linestyle='--', color='gray', alpha=0.5)
sns.despine(ax=axes[0])

sns.barplot(x=test_null_counts, y=test_null_counts.index, ax=axes[1], color='skyblue')
axes[1].set_title('Null Values in Test Data', fontweight='bold')
axes[1].set_xlabel('Count of Null Values', fontweight='bold')
axes[1].set_ylabel('Columns', fontweight='bold')
axes[1].grid(axis='x', linestyle='--', color='gray', alpha=0.3)
sns.despine(ax=axes[1])

plt.tight_layout()

plt.show()


# In[106]:


train.dropna(subset=['Arrival Delay in Minutes'], inplace=True)


# In[107]:


test.dropna(subset=['Arrival Delay in Minutes'], inplace=True)


# In[108]:


train['Arrival Delay in Minutes'].isnull().any()


# In[109]:


test['Arrival Delay in Minutes'].isnull().any()


# In[110]:


for data , name in zip(Data,data_names):
   print(f'There is {data.duplicated().sum()} duplicated data in {name}')


# In[111]:


def calculate_outliers_percentage(df):

    # Calculate the interquartile range (IQR) for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Calculate the number of outliers in each column
    num_outliers = ((df < lower_bound) | (df > upper_bound)).sum()

    # Calculate the percentage of outliers in each column
    pct_outliers = round(num_outliers / len(df) * 100 , 4)

    return pct_outliers


# In[112]:


outliers_train = pd.DataFrame(calculate_outliers_percentage(train), columns=['% Outliers'])
outliers_train.index.name = 'Column Name'
outliers_train.reset_index(inplace=True)
outliers_train


# In[113]:


colors2 = sns.color_palette(['#1337f5', '#E80000'], 2)
colors1 = sns.color_palette(['#1337f5'], 1)


numerical = train.select_dtypes(exclude='object')
n = len(numerical)

for col in numerical:
  fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 4))
  plt.suptitle(f"Distribution of {col}")
  sns.boxplot(data=train, x=col, ax=ax1, palette=colors1)
  ax1.set_xlabel(None)
  ax1.get_xaxis().set_ticks([])
  sns.histplot(data=train, x=col, ax=ax2, palette=colors1)
  plt.subplots_adjust(hspace=0)
  print("\n")
  plt.show()


# In[114]:


colors2 = sns.color_palette(['#1337f5', '#E80000'], 2)
colors1 = sns.color_palette(['#1337f5'], 1)


numerical = test.select_dtypes(exclude='object')
n = len(numerical)

for col in numerical:
  fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 4))
  plt.suptitle(f"Distribution of {col}")
  sns.boxplot(data=test, x=col, ax=ax1, palette=colors1)
  ax1.set_xlabel(None)
  ax1.get_xaxis().set_ticks([])
  sns.histplot(data=test, x=col, ax=ax2, palette=colors1)
  plt.subplots_adjust(hspace=0)
  print("\n")
  plt.show()


# In[115]:


columns_to_visualize = ['Arrival Delay in Minutes', 'Checkin service', 'Departure Delay in Minutes', 'Flight Distance']

fig, axes = plt.subplots(1, 2, figsize=(18, 6))


sns.boxplot(data=train[columns_to_visualize], ax=axes[0])
axes[0].set_title('Outliers in Train Data',fontweight='bold')
axes[0].set_xlabel('Columns')
axes[0].set_ylabel('Values')
axes[0].set_facecolor('darkgray')

sns.boxplot(data=test[columns_to_visualize], ax=axes[1])
axes[1].set_title('Outliers in Test Data',fontweight='bold')
axes[1].set_xlabel('Columns')
axes[1].set_ylabel('Values')
axes[1].set_facecolor('darkgray')

plt.tight_layout()

plt.show()


# In[116]:


def Outliers(df,col):
  Q1 = df[col].quantile(q=0.25)
  Q3 = df[col].quantile(q=0.75)
  IQR = df[col].apply(stats.iqr)
  df_clean = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).any(axis=1)]
  print("Number of rows before removing outliers:", len(df))
  print("Number of rows after removing outliers:", len(df_clean))


# In[117]:


columns = ['Flight Distance']


# In[118]:


Outliers(train,columns)


# In[119]:


Outliers(test,columns)


# In[120]:


sns.set_style("darkgrid")

sns.histplot(data=train, x='Checkin service',bins=20,alpha= 0.3 , kde=True)

plt.title('Distribution of Checkin Service Before Imputing Outliers')
plt.xlabel('Checkin Service')
plt.ylabel('Count')

plt.show()


# In[121]:


median_value = train['Checkin service'].median()
median_value2 = test['Checkin service'].median()
lower_bound = median_value - 1.5 * (train['Checkin service'].quantile(0.75) - train['Checkin service'].quantile(0.25))
upper_bound = median_value + 1.5 * (train['Checkin service'].quantile(0.75) - train['Checkin service'].quantile(0.25))



# Impute outliers in train data with median value
train.loc[(train['Checkin service'] < lower_bound) | (train['Checkin service'] > upper_bound), 'Checkin service'] = median_value
lower_bound = median_value2 - 1.5 * (test['Checkin service'].quantile(0.75) - test['Checkin service'].quantile(0.25))
upper_bound = median_value2 + 1.5 * (test['Checkin service'].quantile(0.75) - test['Checkin service'].quantile(0.25))



# Impute outliers in test data with median value
test.loc[(test['Checkin service'] < lower_bound) | (test['Checkin service'] > upper_bound), 'Checkin service'] = median_value2
sns.set_style("darkgrid")

sns.histplot(data=train, x='Checkin service',bins=20,alpha= 0.3 , kde=True)

plt.title('Distribution of Checkin Service After Imputing Outliers')
plt.xlabel('Checkin Service')
plt.ylabel('Count')

plt.show()


# In[122]:


column_with_outliers = ['Departure Delay in Minutes']


# In[123]:


# IN TRAIN DATA
imputer = KNNImputer(n_neighbors=5)
imputed_values = imputer.fit_transform(train[column_with_outliers])
train[column_with_outliers] = imputed_values



# IN TEST DATA
imputer = KNNImputer(n_neighbors=5)
imputed_values = imputer.fit_transform(test[column_with_outliers])
test[column_with_outliers] = imputed_values


# In[124]:


le = LabelEncoder()
col_encoded = le.fit_transform(train['satisfaction'])
train['satisfaction'] = col_encoded
train['satisfaction'].unique()


# In[125]:


correlation=train.corr()
plt.figure(figsize=(14,8))
sns.heatmap(correlation,annot=True,fmt='.2f',annot_kws={'size': 10},linewidths=0.5,cmap='Blues')
plt.title("Data correlations")


# In[126]:


drop_columns = ['Gender','Arrival Delay in Minutes','Gate location','Departure/Arrival time convenient']
train.drop(drop_columns,axis=1,inplace=True)
train.head()


# In[127]:


test.drop(drop_columns,axis=1,inplace=True)
test.head()


# In[128]:


encoder = OneHotEncoder()
columns_to_encode = ['Customer Type', 'Type of Travel', 'Class']


#FOR TRAIN DATA
encoder = OneHotEncoder(sparse=False)
encoded_columns = encoder.fit_transform(train[columns_to_encode])
encoded_column_names = []


for i, column in enumerate(columns_to_encode):
    categories = encoder.categories_[i]
    encoded_column_names.extend([column + '_' + str(category) for category in categories])
train.drop(columns_to_encode, axis=1, inplace=True)
train[encoded_column_names] = encoded_columns
train.head()


# In[129]:


#FOR TEST DATA
encoder = OneHotEncoder(sparse=False)
encoded_columns = encoder.fit_transform(test[columns_to_encode])
encoded_column_names = []
for i, column in enumerate(columns_to_encode):
    categories = encoder.categories_[i]
    encoded_column_names.extend([column + '_' + str(category) for category in categories])
test.drop(columns_to_encode, axis=1, inplace=True)
test[encoded_column_names] = encoded_columns
test.head()


# In[130]:


columns_to_scale = ['Age', 'Flight Distance', 'Inflight wifi service',
       'Ease of Online booking', 'Food and drink', 'Online boarding',
       'Seat comfort', 'Inflight entertainment', 'On-board service',
       'Leg room service', 'Baggage handling', 'Checkin service',
       'Inflight service', 'Cleanliness', 'Departure Delay in Minutes']
scaler = StandardScaler()
# IN TRAIN DATA
scaled_values = scaler.fit_transform(train[columns_to_scale])
train[columns_to_scale] = scaled_values
aggregated_train = train[columns_to_scale].agg(['mean', 'min', 'max', 'median', 'std']).style.background_gradient(cmap='Blues')
aggregated_train


# In[131]:


# IN TEST DATA
scaled_values = scaler.fit_transform(test[columns_to_scale])
test[columns_to_scale] = scaled_values
aggregated_train = train[columns_to_scale].agg(['mean', 'min', 'max', 'median', 'std']).style.background_gradient(cmap='Blues')
aggregated_train


# In[132]:


X_train = train.drop(["satisfaction"], axis = 1)
y_train = train["satisfaction"]

X_test = test.drop(["satisfaction"], axis = 1)
y_test = test["satisfaction"]

y_test = y_test.replace({"satisfied":1, "neutral or dissatisfied":0})

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)


# In[133]:


custom_palette = ["#0072B2", "#ADD8E6"]

plt.figure(figsize=(10, 8))

sns.set_palette(custom_palette)
ax = sns.countplot(x='satisfaction', data=train)

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points',
                fontsize=12,
                color='black')

plt.title('Class Distribution in Train Data', fontweight='bold', fontsize=16)
plt.xlabel('Satisfaction', fontsize=15,fontweight='bold')
plt.ylabel('Count', fontsize=15,fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.show()


# In[134]:


# imbalance data

ROS=RandomOverSampler(random_state=42)
# train
train,y_train=ROS.fit_resample(train,y_train)

# Check before and after ===>overSample
from collections import Counter
print("Updata dataset Train: ",Counter(y_train))


# In[135]:


custom_palette = ["#0072B2", "#ADD8E6"]

plt.figure(figsize=(10, 8))

sns.set_palette(custom_palette)
ax = sns.countplot(x='satisfaction', data=train)

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points',
                fontsize=12,
                color='black')

plt.title('Class Distribution in Train Data After', fontweight='bold', fontsize=16)
plt.xlabel('Satisfaction', fontsize=15,fontweight='bold')
plt.ylabel('Count', fontsize=15,fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.show()


# In[136]:


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
def plot_roc_auc(model, X_test, y_test, y_pred=None):
  if(y_pred is None):
    y_pred =  model.decision_function(X_test)

  fpr, tpr, thresholds = roc_curve(y_test, y_pred)

  roc_auc = auc(fpr, tpr)

  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.axis('tight')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  return fpr, tpr, thresholds
def plot_roc_auc2(model, X_test, y_test, y_pred=None):
    if y_pred is None:
        y_pred = model.predict_proba(X_test)[:, 1]  # Use the predicted probabilities of the positive class

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return fpr, tpr, thresholds


# In[137]:


lr_model = LogisticRegression()
accuracy_scores = []

# Performing k-fold cross-validation
for train_index, val_index in k_fold.split(X_train):
    # Splitting the data into training and validation sets for the current fold
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Fitting the logistic regression model on the training data
    lr_model.fit(X_train_fold, y_train_fold)

    # Predicting the labels for the validation data
    ylr_val_pred = lr_model.predict(X_val_fold)

    # Calculating the accuracy score for the current fold
    accuracy = accuracy_score(y_val_fold, ylr_val_pred)
    accuracy_scores.append(accuracy)
avglr_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the KNN model using k-fold cross-validation is: {:.2f}%".format(
    avglr_accuracy * 100))


# In[138]:


classification_rep = classification_report(y_val_fold, ylr_val_pred)
print(classification_rep)


# In[139]:


cm = confusion_matrix(y_val_fold, ylr_val_pred)
make_confusion_matrix(cm)


# In[140]:


plot_roc_auc2(lr_model, X_test, y_test, y_pred=None)


# In[141]:


xgboost_model = XGBClassifier()
accuracy_scores = []


for train_index, val_index in k_fold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    xgboost_model.fit(X_train_fold, y_train_fold)

    yxg_val_pred = xgboost_model.predict(X_val_fold)

    xgscore = accuracy_score(y_val_fold, yxg_val_pred)
    accuracy_scores.append(xgscore)
avgXGBoost_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the XGBoost model using k-fold cross-validation is: {:.2f}%".format(
    avgXGBoost_accuracy * 100))


# In[142]:


classification_rep = classification_report(y_val_fold, yxg_val_pred)
print(classification_rep)


# In[143]:


cm = confusion_matrix(y_val_fold, yxg_val_pred)
make_confusion_matrix(cm)


# In[144]:


plot_roc_auc2(xgboost_model, X_test, y_test, y_pred=None)


# In[145]:


random_forest_model = RandomForestClassifier()
accuracy_scores = []

for train_index, val_index in k_fold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    random_forest_model.fit(X_train_fold, y_train_fold)

    yrf_val_pred = random_forest_model.predict(X_val_fold)

    rfscore = accuracy_score(y_val_fold, yrf_val_pred)
    accuracy_scores.append(rfscore)
avgRF_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the RF model using k-fold cross-validation is: {:.2f}%".format(
    avgRF_accuracy * 100))


# In[146]:


classification_rep = classification_report(y_val_fold, yrf_val_pred)
print(classification_rep)


# In[147]:


cm = confusion_matrix(y_val_fold, yrf_val_pred)
make_confusion_matrix(cm)


# In[148]:


plot_roc_auc2(random_forest_model, X_test, y_test, y_pred=None)


# In[149]:


knn_model= KNeighborsClassifier()
accuracy_scores = []

for train_index, val_index in k_fold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    knn_model.fit(X_train_fold, y_train_fold)

    yknn_val_pred = knn_model.predict(X_val_fold)

    knnscore = accuracy_score(y_val_fold, yknn_val_pred)
    accuracy_scores.append(knnscore)
accuracy_scores


# In[150]:


avgknn_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the KNN model using k-fold cross-validation is: {:.2f}%".format(
    avgknn_accuracy * 100))


# In[151]:


classification_rep = classification_report(y_val_fold, yknn_val_pred)
print(classification_rep)


# In[152]:


cm = confusion_matrix(y_val_fold, yknn_val_pred)
make_confusion_matrix(cm)


# In[153]:


plot_roc_auc2(knn_model, X_test, y_test, y_pred=None)


# In[154]:


dt_model = DecisionTreeClassifier()
accuracy_scores = []

for train_index, val_index in k_fold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    dt_model.fit(X_train_fold, y_train_fold)

    ydt_val_pred = dt_model.predict(X_val_fold)

    dtscore = accuracy_score(y_val_fold, ydt_val_pred)
    accuracy_scores.append(dtscore)
accuracy_scores


# In[155]:


avgdt_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the DT model using k-fold cross-validation is: {:.2f}%".format(
    avgdt_accuracy * 100))


# In[156]:


classification_rep = classification_report(y_val_fold, ydt_val_pred)
print(classification_rep)


# In[157]:


cm = confusion_matrix(y_val_fold, ydt_val_pred)
make_confusion_matrix(cm)


# In[158]:


plot_roc_auc2(dt_model, X_test, y_test, y_pred=None)


# In[180]:


svm_model = SVC()
accuracy_scores = []

for train_index, val_index in k_fold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    svm_model.fit(X_train_fold, y_train_fold)

    ysvm_val_pred = svm_model.predict(X_val_fold)

    accuracy_fold = accuracy_score(y_val_fold, ysvm_val_pred)
    accuracy_scores.append(accuracy_fold)
accuracy_scores


# In[160]:


avgsvm_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the SVM model using k-fold cross-validation is: {:.2f}%".format(
    avgsvm_accuracy * 100))


# In[161]:


classification_rep = classification_report(y_val_fold, ysvm_val_pred)
print(classification_rep)


# In[162]:


cm = confusion_matrix(y_val_fold, ysvm_val_pred)
make_confusion_matrix(cm)


# In[163]:


plot_roc_auc(svm_model, X_test, y_test, y_pred=None)


# In[164]:


nb_model = GaussianNB()
accuracy_scores = []

for train_index, val_index in k_fold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    nb_model.fit(X_train_fold, y_train_fold)

    ynb_val_pred = nb_model.predict(X_val_fold)

    accuracy_fold = accuracy_score(y_val_fold, ynb_val_pred)
    accuracy_scores.append(accuracy_fold)
accuracy_scores


# In[165]:


avgnb_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the Naive Bayes model using k-fold cross-validation is: {:.2f}%".format(
    avgnb_accuracy * 100))


# In[166]:


classification_rep = classification_report(y_val_fold, ynb_val_pred)
print(classification_rep)


# In[167]:


cm = confusion_matrix(y_val_fold, ynb_val_pred)
make_confusion_matrix(cm)


# In[168]:


plot_roc_auc2(nb_model, X_test, y_test, y_pred=None)


# In[169]:


bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
accuracy_scores = []

for train_index, val_index in k_fold.split(X_train):

    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    bagging_model.fit(X_train_fold, y_train_fold)

    y_val_pred = bagging_model.predict(X_val_fold)

    accuracy = accuracy_score(y_val_fold, y_val_pred)
    accuracy_scores.append(accuracy)
accuracy_scores


# In[170]:


avgBagging_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the Bagging Classifier using k-fold cross-validation is: {:.2f}%".format(
    avgBagging_accuracy * 100))


# In[171]:


classification_rep = classification_report(y_val_fold, y_val_pred)
print(classification_rep)


# In[172]:


cm = confusion_matrix(y_val_fold, y_val_pred)
make_confusion_matrix(cm)


# In[173]:


plot_roc_auc2(bagging_model, X_test, y_test, y_pred=None)


# In[174]:


adaboost_model  = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=4)
accuracy_scores = []

for train_index, val_index in k_fold.split(X_train):

    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    adaboost_model.fit(X_train_fold, y_train_fold)

    y_val_pred = adaboost_model.predict(X_val_fold)

    accuracy = accuracy_score(y_val_fold, y_val_pred)
    accuracy_scores.append(accuracy)
accuracy_scores


# In[175]:


avgAbaboosting_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("The average accuracy of the Bagging Classifier using k-fold cross-validation is: {:.2f}%".format(
    avgAbaboosting_accuracy * 100))


# In[176]:


classification_rep = classification_report(y_val_fold, y_val_pred)
print(classification_rep)


# In[177]:


cm = confusion_matrix(y_val_fold, y_val_pred)
make_confusion_matrix(cm)


# In[178]:


plot_roc_auc2(adaboost_model, X_test, y_test, y_pred=None)


# In[179]:


model_accuracies = {}
model_accuracies['LogisticRegression'] = avglr_accuracy
model_accuracies['RandomForestClassifier'] = avgRF_accuracy
model_accuracies['KNeighborsClassifier'] = avgknn_accuracy
model_accuracies['DecisionTreeClassifier'] = avgdt_accuracy
model_accuracies['SVM'] = avgsvm_accuracy
model_accuracies['XGBoost'] = avgXGBoost_accuracy
model_accuracies['Naive Bayes'] = avgnb_accuracy
model_accuracies['Bagging Classifier'] = avgBagging_accuracy
model_accuracies['AdaBoosting Classifier'] = avgAbaboosting_accuracy
results = pd.DataFrame(model_accuracies.items(), columns=['Model', 'Accuracy'])
results = results.sort_values(by='Accuracy', ascending=False).style.background_gradient(cmap='Blues')
results


# In[ ]:




