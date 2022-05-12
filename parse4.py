import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

def AvgBucketData(df):
    weighted_df_avg = pd.DataFrame({"Mother's Delivery Weight":[],
                                    "Infant Birth Weight 14":[],
                                    "Region Code":[]})
    for row in df.iterrows():
        for i in range(int(row[1]["Percentage of Births"])):
            mom_del_weight = row[1]["Mother's Delivery Weight"]
            if mom_del_weight == "100 - 149 lbs":
                momweight = (100+149)/2
            elif mom_del_weight == "150 - 199 lbs":
                momweight = (150+199)/2
            elif mom_del_weight == "200 - 249 lbs":
                momweight=(200+249)/2
            elif mom_del_weight == "250 - 299 lbs":
                momweight=(250+299)/2
            elif mom_del_weight == "300 - 349 lbs":
                momweight=(300+349)/2
            elif mom_del_weight == "350 - 400 lbs":
                momweight=(350+400)/2
            infant_weight = row[1]["Infant Birth Weight 14"]
            if infant_weight == "499 grams or less":
                inf_weight = (499)/2
            elif infant_weight == "500 - 749 grams":
                inf_weight = (500+749)/2
            elif infant_weight == "750 - 999 grams":
                inf_weight = (750+999)/2
            elif infant_weight == "1000 - 1249 grams":
                inf_weight = (1000+1249)/2
            elif infant_weight == "1250 - 1499 grams":
                inf_weight = (1250+1499)/2
            elif infant_weight == "1500 - 1999 grams":
                inf_weight = (1500+1999)/2
            elif infant_weight == "2000 - 2499 grams":
                inf_weight = (2000+2499)/2
            elif infant_weight == "2500 - 2999 grams":
                inf_weight = (2500+2999)/2
            elif infant_weight == "3000 - 3499 grams":
                inf_weight = (3000+3499)/2
            elif infant_weight == "3500 - 4000 grams":
                inf_weight = (3500+4000)/2
            elif infant_weight == "4000 - 4499 grams":
                inf_weight == (4000+4499)/2
            elif infant_weight == "4500 - 5000 grams":
                inf_weight = (4500+5000)/2
            elif infant_weight == "5000 - 8165 grams":
                inf_weight = (5000+8165)/2

            if row[1]['Census Region of Residence Code']=="CENS-R1":
                region = 1
            elif row[1]['Census Region of Residence Code']=="CENS-R2":
                region = 2
            elif row[1]['Census Region of Residence Code']=="CENS-R3":
                region = 3
            elif row[1]['Census Region of Residence Code']=="CENS-R4":
                region = 4
            weighted_df_avg.loc[len(weighted_df_avg.index)]=[momweight,inf_weight,region]
    return weighted_df_avg

#reading data csv.
df = pd.read_csv('StatsCSV.csv')
del df["Notes"]

births = df["Births"].tolist()
#separates the regions into distinct graphs
dfne=df.loc[df['Census Region of Residence']=='Census Region 1: Northeast']
dfmw=df.loc[df['Census Region of Residence']=='Census Region 2: Midwest']
dfs= df.loc[df['Census Region of Residence']=='Census Region 3: South']
dfw=df.loc[df['Census Region of Residence']=='Census Region 4: West']
#finds the total amount of births per region for weighting data points later
dfne_birth_total=dfne['Births'].sum()
dfmw_birth_total=dfmw['Births'].sum()
dfs_birth_total=dfs['Births'].sum()
dfw_birth_total=dfw['Births'].sum()

#Dataframes and data calc for US-NORTHEAST
dfne_percentage_birth_array=[]
for row in dfne.iterrows():
    dfne_percentage_birth_array.append(((row[1]['Births'])/dfne_birth_total)*100)
dfne['Percentage of Births'] = dfne_percentage_birth_array
dfne=dfne.sort_values(by=['Infant Birth Weight 14 Code'])
avg_dfne=AvgBucketData(dfne)

#Dataframes and data calc for US-MIDWEST
dfmw_percentage_birth_array=[]
for row in dfmw.iterrows():
    dfmw_percentage_birth_array.append(((row[1]['Births'])/dfmw_birth_total)*100)
dfmw['Percentage of Births'] = dfmw_percentage_birth_array
dfmw=dfmw.sort_values(by=['Infant Birth Weight 14 Code'])
avg_dfmw = AvgBucketData(dfmw)


#Dataframes and data calc for US-SOUTH
dfs_percentage_birth_array=[]
for row in dfs.iterrows():
    dfs_percentage_birth_array.append(((row[1]['Births'])/dfs_birth_total)*100)
dfs['Percentage of Births'] = dfs_percentage_birth_array
dfs=dfs.sort_values(by=['Infant Birth Weight 14 Code'])
avg_dfs=AvgBucketData(dfs)


#Dataframes and data calc for US-WEST
dfw_percentage_birth_array=[]
for row in dfw.iterrows():
    dfw_percentage_birth_array.append(((row[1]['Births'])/dfw_birth_total)*100)
dfw['Percentage of Births'] = dfw_percentage_birth_array
dfw=dfw.sort_values(by=['Infant Birth Weight 14 Code'])
avg_dfw = AvgBucketData(dfw)

#combines the four average dataframes for each region into one large dataframe
frames = [avg_dfne, avg_dfmw, avg_dfs,avg_dfw]
result = pd.concat(frames)

#Mother's Weight Hypothesis Testing
#calculates the means
df_mean = np.mean(result["Mother's Delivery Weight"])
print("Mean weight",df_mean)

#runs a t test on the inputs and produces a p value
tstat, p_val = ttest_1samp(result["Mother's Delivery Weight"],result["Mother's Delivery Weight"].size)
print("p-values",p_val)

if p_val < 0.01:
    print("We reject that Mother's Weight doesn't have an effect on Infant Weight at alpha =0.01")
else:
    print("We accept that Mother's Weight doesn't have an effect on Infant Weight at alpha =0.01")

#Region Hypothesis Testing
#calculates the means
df_mean2 = np.mean(result["Region Code"])
print("Mean Region",df_mean2)

tstat2, p_val2 = ttest_1samp(result["Region Code"],result["Region Code"].size)
print("p-values",p_val2)

if p_val2 < 0.01:
    print("We reject that Region doesn't have an effect on infant weight at alpha=0.01")
else:
    print("We accept that Region doesn't have an effect on infant weight at alpah =0.01")
