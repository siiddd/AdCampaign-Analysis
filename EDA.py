#Import Packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#Import the Dataset 
df = pd.read_excel('AdCampaignData.xlsx')

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

#Lets us Create another dataset df_filtered with no '_ values

le = LabelEncoder()
aud_type_encoded = le.fit_transform(df.audience_type)
aud_type_encoded = pd.DataFrame(aud_type_encoded, columns=['Values'])

creative_type_encoded = le.fit_transform(df.creative_type)
creative_type_encoded = pd.DataFrame(creative_type_encoded, columns = ['Values1'])

df_filtered = [df, aud_type_encoded]
df_filtered = pd.concat(df_filtered, axis = 1)
df_filtered['Values'] = df_filtered['Values'].map({0:'Audience 4', 1:'Audience 1', 2:'Audience 2', 3:'Audience 3'}) 
df_filtered = pd.concat([df_filtered, creative_type_encoded], axis = 1)
df_filtered['Values1'] = df_filtered['Values1'].map({0:'Google Content', 1:'Carousal', 2:'Image'}) 



#Lets Visualize our Data
sns.barplot(x = "campaign_platform", y = "spends", hue = "Values", data = df_filtered).set_title('Platform vs Spend')
sns.barplot(x = "device", y = "spends", data = df_filtered).set_title('Device vs Spend')
df_age = df_filtered[df_filtered.age != 'Undetermined'] 
sns.barplot(x = "age", y = "spends", data = df_age).set_title('Age vs Spend')
sns.barplot(x = "Values1", y = "spends", data = df_filtered).set(xlabel = 'Content Type', title = 'Content Type vs Spend')
sns.barplot(x = "Date", y = "spends", date = df_filtered) 
df_filtered[['Date','spends']].set_index('Date').plot()
df_filtered[['Date','clicks']].set_index('Date').plot()




