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

#organizes data to be crunched for beta values
rowX=[]
for row in result.iterrows():
    rowX.append(1)
result["C"]=rowX
ydata = np.array(result['Infant Birth Weight 14'])
ydata=np.vstack(ydata)

xframe = pd.DataFrame({"C":[],"Mother's Delivery Weight":[],"Region Code":[]})
for row in result.iterrows():
    xframe.loc[len(xframe.index)]=[row[1]["C"],row[1]["Mother's Delivery Weight"],row[1]["Region Code"]]
#calculates beta values
xdata = xframe.to_numpy()
xtranspose = np.transpose(xdata)
x_prod = np.matmul(xtranspose, xdata)
x_inv = np.linalg.inv(x_prod)
xprody = np.matmul(np.transpose(xdata),ydata)
beta = np.matmul(x_inv,xprody)
print("Beta:",beta)
beta1 = beta[0]
beta2 = beta[1]
beta3 = beta[2]


#drawing to graph
from matplotlib import cm
x1_data = result["Mother's Delivery Weight"].to_numpy()
x2_data = result["Region Code"].to_numpy()
y1_data = result["Infant Birth Weight 14"].to_numpy()

X_mom, X_region = np.meshgrid(x1_data, x2_data)
def regression(x1, x2):
    return beta1+(x1*beta2)+(x2*beta3) #beta 2 corresponds to mother's weight, beta3 corresponds to region code
reg_vec = np.vectorize(regression)
y_reg = reg_vec(X_mom, X_region)

#drawing the graph
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("Mother's Delivery Weight")
ax.set_ylabel("Region Code")
ax.set_zlabel("Infant Birth Weight 14")
ax.scatter(x1_data, x2_data, y1_data)
ax.plot_surface(X_mom, X_region, y_reg)
plt.show()

def summation(array_a):
    sum=0
    for i in array_a:
        sum+=i
    return sum

#calculating SST
ybarsum = summation(list(result["Infant Birth Weight 14"]))
ybar=ybarsum/(len(list(result["Infant Birth Weight 14"]))-1)
y_total_difference = []
for i in list(result["Infant Birth Weight 14"]):
    y_total_result = (i-ybar)**2
    y_total_difference.append(y_total_result)
y_sst = summation(y_total_difference)
print("SST",y_sst)

#calculating SSR
xvalue = list(result["Mother's Delivery Weight"])
zvalue = list(result["Region Code"])

pre_SSR_list=[]
for i in range(len(list(result["Infant Birth Weight 14"]))):
    yhat = float(beta1 + beta2*xvalue[i] + beta3+zvalue[i])
    yhat_ybar_diff = yhat - ybar
    pre_SSR = yhat_ybar_diff**2
    pre_SSR_list.append(pre_SSR)
y_ssr = summation(pre_SSR_list)
print("SSR",y_ssr)

i=0
pre_SSE_list = []
for item in list(result["Infant Birth Weight 14"]):
    yhat = float(beta1 + beta2*xvalue[i] + beta3+zvalue[i])
    yhat_y_diff = yhat - float(item)
    pre_SSE = yhat_y_diff**2
    pre_SSE_list.append(pre_SSE)
    i+=1
y_sse = summation(pre_SSE_list)
print("SSE",y_sse)

R_squared = y_ssr/y_sst
print("R-squared",R_squared*100,"% (",R_squared,")")

MSE = y_sse/(len(list(result["Infant Birth Weight 14"]))-(2+1))
print("MSE (Error Variance):",MSE,"grams")

reg_std_error=np.sqrt(MSE)
print("Regression Standard Error (omega):",reg_std_error, "grams")