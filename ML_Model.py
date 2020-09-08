#Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Import Dataset
df = pd.read_excel(r'C:\Users\nsid4\Desktop\A1.xlsx') #Replace the path with Local Directory

#Check for missing data
df.isna().sum() #Missing data only in link_clicks

#Get insights in link_clicks
df.link_clicks.describe() #Since more than 75%ile of the data is 0 we will replace the missing values with 0 as well

#Replace the missing values
df.link_clicks.fillna(value = 0, inplace = True)

#We will drop Phase and Product since they have a single value across all Rows
df.drop(['product', 'phase'], axis = 1, inplace = True)


#See the number of unique values in each column
df.device.value_counts()
df.campaign_platform.value_counts()
df.campaign_type.value_counts()
df.communication_medium.value_counts()
df.subchannel.value_counts()
df.audience_type.value_counts()
df.creative_type.value_counts()
df.creative_name.value_counts()
df.age.value_counts()


#We will drop more features which do not help us much while creating a model
df_modified = df.drop(['impressions', 'clicks', 'link_clicks', 'Date'], axis = 1)

#Let us create Dummy variables for all the categorical variables in our dataset
campaign_platform_dummy = pd.get_dummies(df_modified.campaign_platform)
campaign_type_dummy = pd.get_dummies(df_modified.campaign_type)
communication_medium_dummy = pd.get_dummies(df_modified.communication_medium)
subchannel_dummy = pd.get_dummies(df_modified.subchannel)
#audience_type_dummy = pd.get_dummies(df_modified.audience_type)
creative_type_dummy = pd.get_dummies(df_modified.creative_type)
creative_name_dummy = pd.get_dummies(df_modified.creative_name)
device_dummy = pd.get_dummies(df_modified.device)
age_dummy = pd.get_dummies(df_modified.age)

#Inorder to prevent dummy variable trap we will have to drop N-1 feature from each of the dummy variables
campaign_platform_dummy = campaign_platform_dummy.drop(columns = 'Google Ads')
campaign_type_dummy = campaign_type_dummy.drop(campaign_type_dummy.columns[[0]], axis = 1)
communication_medium_dummy = communication_medium_dummy.drop(communication_medium_dummy.columns[[0]], axis = 1)
subchannel_dummy = subchannel_dummy.drop(subchannel_dummy.columns[[0]], axis = 1)
#audience_type_dummy = audience_type_dummy.drop(audience_type_dummy.columns[[0]], axis = 1)
creative_type_dummy = creative_type_dummy.drop(creative_type_dummy.columns[[0]], axis = 1)
creative_name_dummy = creative_name_dummy.drop(creative_name_dummy.columns[[0]], axis = 1)
device_dummy = device_dummy.drop(device_dummy.columns[[0]], axis = 1)
age_dummy = age_dummy.drop(age_dummy.columns[[6]], axis = 1)

#Lets us drop the original columns and concat the dummy variables to our dataset
df_modified = df_modified.drop(df_modified.columns[[0,1,2,3,5,6,7,8]], axis = 1)
df_modified = pd.concat([df_modified, campaign_platform_dummy, campaign_type_dummy, communication_medium_dummy, subchannel_dummy, creative_type_dummy, creative_name_dummy, device_dummy, age_dummy], axis = 1)

le = LabelEncoder()
aud_type_encoded = le.fit_transform(df.audience_type)
aud_type_encoded = pd.DataFrame(aud_type_encoded, columns=['Audience_Type'])

df_modified = pd.concat([df_modified, aud_type_encoded], axis = 1)
df_modified['Audience_Type'] = df_modified['Audience_Type'].map({0:'Audience 4', 1:'Audience 1', 2:'Audience 2', 3:'Audience 3'}) 
df_modified = df_modified.drop(df_modified.columns[[0]], axis = 1)

#Let us Split our data into Test and Train
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(pd.DataFrame(df_modified.iloc[:, :-1]), pd.DataFrame(df_modified.Audience_Type), train_size = 0.90, random_state = 18)

#Let us create our model
from sklearn.svm import SVC
svcmodel = SVC(kernel = 'rbf', gamma = 'auto', random_state = 18)
svcmodel.fit(x_train, y_train.values.ravel())
svcmodel.score(x_test, y_test)

#Check for custom values
svcmodel.predict([[3435.0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0]]) #Input your test values here

