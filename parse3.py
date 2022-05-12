from audioop import avgpp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
from mpl_toolkits import mplot3d

def func(x,a,b):
    y=a*x+b
    return y 

#this function will return a dataframe with an exact bucket average 
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

#this function will return dataframe with juttered values for a bucket
# def estimateData(df):
    weighted_df_est = pd.DataFrame({"Mother's Delivery Weight":[],
                              "Infant Birth Weight 14":[],
                              "Region Code":[]})
    for row in df.iterrows():
        for i in range(int(row[1]["Percentage of Births"])):
            #Groups Mother's Delivery Weight into rough ranges
            mom_del_weight=row[1]["Mother's Delivery Weight"]
            if mom_del_weight == "100 - 149 lbs":
                momweight = random.randint(120,130)
            elif mom_del_weight == "150 - 199 lbs":
                momweight = random.randint(170,180)
            elif mom_del_weight == "200 - 249 lbs":
                momweight=random.randint(220,230)
            elif mom_del_weight == "250 - 299 lbs":
                momweight=random.randint(270,280)
            elif mom_del_weight == "300 - 349 lbs":
                momweight=random.randint(320,330)
            elif mom_del_weight == "350 - 400 lbs":
                momweight=random.randint(370,380)
            #Groups Infant Weight into Rough Ranges
            infant_weight = row[1]["Infant Birth Weight 14"]
            if infant_weight == "499 grams or less":
                inf_weight = random.randint(200,300)
            elif infant_weight == "500 - 749 grams":
                inf_weight = random.randint(575,675)
            elif infant_weight == "750 - 999 grams":
                inf_weight = random.randint(825,925)
            elif infant_weight == "1000 - 1249 grams":
                inf_weight = random.randint(1100,1200)
            elif infant_weight == "1250 - 1499 grams":
                inf_weight = random.randint(1300,1400)
            elif infant_weight == "1500 - 1999 grams":
                inf_weight = random.randint(1700,1800)
            elif infant_weight == "2000 - 2499 grams":
                inf_weight = random.randint(2200,2300)
            elif infant_weight == "2500 - 2999 grams":
                inf_weight = random.randint(2700,2800)
            elif infant_weight == "3000 - 3499 grams":
                inf_weight = random.randint(3200,3300)
            elif infant_weight == "3500 - 4000 grams":
                inf_weight = random.randint(3700,3800)
            elif infant_weight == "4000 - 4499 grams":
                inf_weight == random.randint(4200,4300)
            elif infant_weight == "4500 - 5000 grams":
                inf_weight = random.randint(4700,4800)
            elif infant_weight == "5000 - 8165 grams":
                inf_weight = random.randint(5500,5600)
            
            if row[1]['Census Region of Residence Code']=="CENS-R1":
                region = 1
            elif row[1]['Census Region of Residence Code']=="CENS-R2":
                region = 2
            elif row[1]['Census Region of Residence Code']=="CENS-R3":
                region = 3
            elif row[1]['Census Regopm pf Resodemce Code']=="CENS-R4":
                region = 4
            weighted_df_est.loc[len(weighted_df_est.index)]=[momweight,inf_weight,region]
    return weighted_df_est

#this function converts the specific dataframe column into a numpy array, returns two arrays (xdata and ydata)
def toNumpyArray(df):
    xdat=[]
    ydat=[]
    for row in df.iterrows():
        ydat.append(row[1]["Mother's Delivery Weight"])
        xdat.append(row[1]["Infant Birth Weight 14"])
    xdata = np.array(xdat)
    ydata = np.array(ydat)
    return xdata, ydata


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
rowX=[]
for row in result.iterrows():
    rowX.append(1)
result["C"]=rowX

ydata = np.array(result['Infant Birth Weight 14'])
ydata=np.vstack(ydata)
print(ydata)

xframe = pd.DataFrame({"C":[],"Mother's Delivery Weight":[],"Region Code":[]})
for row in result.iterrows():
    xframe.loc[len(xframe.index)]=[row[1]["C"],row[1]["Mother's Delivery Weight"],row[1]["Region Code"]]

print(xframe)

xdata = xframe.to_numpy()
print(xdata)

xtranspose = np.transpose(xdata)
x_prod = np.matmul(xtranspose, xdata)
x_inv = np.linalg.inv(x_prod)
print(x_inv)

xprody = np.matmul(np.transpose(xdata),ydata)
beta = np.matmul(x_inv,xprody)
print("Beta:",beta)

fig = plt.figure()
ax = plt.axes(projection='3d')

for row in result.iterrows():
    xs = row[1]["Mother's Delivery Weight"]
    zs = row[1]["Infant Birth Weight 14"]
    ys = row[1]["Region Code"]
    ax.scatter(xs,ys,zs)
plt.show()