# ODD2023-Datascience-Ex06
# AIM:
To read the given data and perform Feature Transformation process and save the data to a file.
# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
# ALGORITHM:
### STEP 1:
Read the given Data
### STEP 2:
Clean the Data Set using Data Cleaning Process
### STEP 3:
Apply Feature Transformation techniques to all the features of the data set
### STEP 4:
Print the transformed features
# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
# OUTPUT:
## Original data:
![274235104-c06332e3-2093-4674-b862-ea5fd699a378](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/f7ab76c8-09d4-4860-966d-54c285606b84)

## Data information:
![274235405-22b7a6f3-8af3-4c8a-8a6e-57f43d9bd12f](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/e316357c-229c-411b-b00b-5ef6546a185e)

## Data describe:
![274235595-33133b16-e895-419f-a38e-6c2616f93dd6](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/951a6b04-24ba-4297-8169-3844da200641)

## Before transformation:
![274235740-8e207720-fa46-4a0d-8d54-9dcd99e32c95](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/7eab3ff5-e852-4573-873a-94d97b9b3fe9)
![274235885-3dd44293-5295-4ca8-aea0-1cd36a011aab](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/7314e39f-db22-478b-ac86-7afab1d5f016)
![274236003-419d60ba-f137-4ee4-968c-b6e6479883ac](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/b09b3c07-23cf-4763-b616-8c9f68f5c366)
![274236100-a65c904d-1b29-4896-ad91-c95edc0ba0ce](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/084f455d-aa31-4232-8b74-c22dda673e0f)

## Log transformation:
![274236218-f5dd7244-b004-4311-b598-4a499c85969f](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/0d9c3250-da99-4279-9bea-10ab36957047)

## Reciprocal transformation:
![274236776-a36cbcac-fa50-4836-b1af-c1fcc4ae89f2](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/136b50b5-2e88-439e-8fbd-5f0815583cc4)

## Square root transformation:
![274236994-e4e36eb0-26ea-49a4-9089-acc48da671c9](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/7988964e-8b4f-4c1d-a101-c650fec4f522)
![274237882-5cdf222d-b13f-4b69-9749-9ef9427d7ccd](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/17eb2903-0b00-4ba6-bc4c-db20e745236e)

## Power transformation:
![274238438-1c78172d-9337-4f19-a89f-078c2320dd78](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/062d9ab8-269a-4a75-a444-e8472158dcc0)
![274238505-b26bae30-af3d-46a4-9448-2dbaabd66564](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/d98c4bc1-0948-42a7-9f56-236016885d2d)

## Quantile transformation:
![274238666-c571497c-130a-4055-9656-2b52c5695cf4](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex06/assets/119393589/f0b7bf15-a6bd-46d9-80a7-eac715a96263)

# RESULT:
Thus feature transformation is done for the given dataset.











