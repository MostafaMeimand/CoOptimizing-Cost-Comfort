#%% importing requirements
#%%% Reading libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import gurobipy as gp
from gurobipy import GRB
import random
import pwlf
import warnings
warnings.filterwarnings("ignore")

#%%% Importing electricity price
Electricity_Price = pd.read_excel("C:/Users/mosta/Desktop/ReactiveController_Commercial_9.0/DDRC implementation.xlsx",
                                  sheet_name = "Electricity Price")
Electricity_Price = Electricity_Price["Elecricity Price"]/1000
Electricity_Price = Electricity_Price[0:96]
#%%% Importing thermal comfort profiles
Profiles_Dataset = pd.read_csv("G:\Shared drives\_Research Repository - Mostafa Meimand\Working Papers\++HVAC controller integrated with Personal Thermal Comfort and Real Time Price\ComfortProfiling\min_profiles.csv")
for i in range(1,31):
    Profiles_Dataset["Probability" + str(i)] = round(Profiles_Dataset["Probability" + str(i)],2)

#%% Assigning agents to different zones
Nagent = 5
Agents = [8,2,15,7,13] #random.sample(range(1,31), Nagent)
print(Agents)
#%%
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(8)])
plt.axhline(y=0.5, color='green', ls='--')
plt.axvline(x = 27.80, c = 'r', ls=':')
plt.axvline(x = 22.18, c = 'r', ls=':')
plt.legend(["Comfort profile","line of comfort threshold","upper bound = 27.80", "lower bound = 22.18"])
plt.tight_layout()
plt.ylabel("Temperature")
plt.xlabel("Probability")
# plt.savefig("_comfort_profile.png",dpi = 1800)
#%%% Plotting the profiles at each zone
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[0])])
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[1])])
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[2])])
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[3])])
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[4])])
plt.legend(["Zone 0","Zone 1","Zone 2","Zone 3","Zone 4"])
# plt.savefig("1_comfort_profiles.png",dpi = 1800)

#%%
n_segments = 4 # best number based on different tests
mu = 0.49
def approx_PWLF(Zone): # approaximating objective function and constraints
    my_pwlf = pwlf.PiecewiseLinFit(Profiles_Dataset["Temperature"], Profiles_Dataset["Probability" + str(Agents[Zone])])
    breaks = my_pwlf.fit(n_segments).round(2)
    y_breaks = my_pwlf.predict(breaks).round(2)
    
    maxConstraint = Profiles_Dataset["Temperature"][Profiles_Dataset["Probability" + str(Agents[Zone])].round(2) == mu].max()
    minConstraint = Profiles_Dataset["Temperature"][Profiles_Dataset["Probability" + str(Agents[Zone])].round(2) == mu].min()
    
    return breaks, y_breaks, maxConstraint, minConstraint
   
def ReadExcel(name):
    Output = pd.read_csv("C:/Users/mosta/Desktop/ReactiveController_Commercial_9.0/" + name + ".csv")
    
    # Adding dates to the dataset
    delimiters = " ", ":", "/"
    regexPattern = '|'.join(map(re.escape, delimiters))
    Output["Month"] = None
    Output["Day"] = None
    Output["Hour"] = None
    Output["Minutes"] = None
    
    for i in range(Output.shape[0]):
      Output["Month"][i] = int(re.split(regexPattern,Output["Date/Time"][i])[1])
      Output["Day"][i] = int(re.split(regexPattern,Output["Date/Time"][i])[2])
      Output["Hour"][i] = int(re.split(regexPattern,Output["Date/Time"][i])[4])
      Output["Minutes"][i] = int(re.split(regexPattern,Output["Date/Time"][i])[5])
    
    Output = Output[Output["Day"] == 1]
    Output["PSZ-AC:1:Air System Total Cooling Energy [J](TimeStep)"] *= 2.77778e-7
    Output["PSZ-AC:2:Air System Total Cooling Energy [J](TimeStep)"] *= 2.77778e-7
    Output["PSZ-AC:3:Air System Total Cooling Energy [J](TimeStep)"] *= 2.77778e-7
    Output["PSZ-AC:4:Air System Total Cooling Energy [J](TimeStep)"] *= 2.77778e-7
    Output["PSZ-AC:5:Air System Total Cooling Energy [J](TimeStep) "] *= 2.77778e-7

    Output["time"] = Output.index
    return Output

def textGenerator(Zone):
    k = 0
    String = "EnergyManagementSystem:Program," + "\n"
    String += "MyComputedCoolingSetpointProg_" + str(Zone + 1) + "," + "\n"
    String += "IF (Hour == " + str(X["Hours"][k]) + ") && (Minute  <=  " + str(X["Minutes"][k]) + "),  Set myCLGSETP_SCH_Override_" + str(Zone + 1) + " = " + str(X["Zone " + str(Zone)][k]) + "," 
    
    for i in range(1,96):
        String += "ELSEIF (Hour == " + str(X["Hours"][k]) + ") && (Minute  <=  " + str(X["Minutes"][k]) + "),  Set myCLGSETP_SCH_Override_" + str(Zone + 1) + " = " + str(X["Zone " + str(Zone)][k]) + "," 
        k += 1
    String += "ENDIF;"
    return String

def CoSimulation():
    text = open("OfficeSmall_main.txt").read()
    NextFile = open("OfficeSmall_1.IDF","wt")
    NextFile.write(text[:300896] + '\n' + '\n' + textGenerator(0) + '\n' + '\n' + textGenerator(1) + '\n'
                   + '\n' + textGenerator(2) + '\n' + '\n' + textGenerator(3) + '\n' + '\n' + textGenerator(4) + '\n'
                   + '\n' +  text[300899:])
    NextFile.close()
    os.system("energyplus -w USA_TX_Austin-Mueller.Muni.AP.722540_TMY3.epw -r OfficeSmall_1.idf")

#%% Preparing X for a day
X = ReadExcel("OfficeSmall_main")

X["Previous Temperature_0"] = None
X["Previous Temperature_0"][0] = 23

X["Previous Temperature_1"] = None
X["Previous Temperature_1"][0] = 23

X["Previous Temperature_2"] = None
X["Previous Temperature_2"][0] = 23

X["Previous Temperature_3"] = None
X["Previous Temperature_3"][0] = 23

X["Previous Temperature_4"] = None
X["Previous Temperature_4"][0] = 23

X["Index"] = X.index
X["Zone 0"] = 29.44
X["Zone 1"] = 29.44
X["Zone 1"][34:36] = 20
X["Zone 2"] = 29.44
X["Zone 3"] = 29.44
X["Zone 4"] = 29.44
X["Minutes"] = X["Index"] % 4 * 15 + 15
temp = []
for i in range(0,24):
    temp.append([i] * 4)
X["Hours"] = np.reshape(temp, (1,96))[0]

#%% First CoSimulation to just to get the states
CoSimulation()
X["Previous Temperature_0"] = ReadExcel("eplusout")["CORE_ZN:Zone Mean Air Temperature [C](TimeStep)"]
X["Previous Temperature_1"] = ReadExcel("eplusout")["PERIMETER_ZN_1:Zone Mean Air Temperature [C](TimeStep)"]
X["Previous Temperature_2"] = ReadExcel("eplusout")["PERIMETER_ZN_2:Zone Mean Air Temperature [C](TimeStep)"]
X["Previous Temperature_3"] = ReadExcel("eplusout")["PERIMETER_ZN_3:Zone Mean Air Temperature [C](TimeStep)"]
X["Previous Temperature_4"] = ReadExcel("eplusout")["PERIMETER_ZN_4:Zone Mean Air Temperature [C](TimeStep)"]

x_values = []
#%% Controling for a day
# I want to control zone 1 while other zones are fixed on 29.44 which is setback
for timestep in range(35,72):
    eta = 1
    ##########################################################################
    #### Problem formulation for zone 1, which is a premeter zone
    model = gp.Model("optim")
    z = model.addVar(name="z") # value of the objective function
    x = model.addVar(name="x") # next temperature
    delta_u = model.addVar(name="delta_u",lb=-20, ub=+20)
    # setpoint of the building
    # Adding constraints
    model.addConstr(x == -0.00185 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep] + 
                    0.0076 * X["Previous Temperature_1"][timestep] + delta_u * 0.4430 - 0.1351 + 
                    X["Previous Temperature_1"][timestep])

    my_pwlf = approx_PWLF(1)
    model.addConstr(x <= my_pwlf[2])
    model.addConstr(x >= my_pwlf[3])
    
    # Auxilary varialbes for the second term
    x1 = model.addVar(name="x1") 
    x2 = model.addVar(name="x2")
    x3 = model.addVar(name="x3")
    x4 = model.addVar(name="x4")
    x5 = model.addVar(name="x5")
    model.addConstr(x == my_pwlf[0][0] * x1 + my_pwlf[0][1] * x2 + my_pwlf[0][2] * x3 + my_pwlf[0][3] * x4 + my_pwlf[0][4] * x5)
    model.addConstr(x1 + x2 + x3 + x4 + x5 == 1)
    model.addSOS(GRB.SOS_TYPE2, [x1, x2 , x3, x4, x5])
    
    # defining opjective function       
    model.addConstr(z == (0.0664 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep]
                          + 0.0288 * X["PERIMETER_ZN_1:Zone Mean Air Temperature [C](TimeStep)"][timestep] 
                          - 0.3228 * (X["Zone 1"][timestep] + delta_u) + 7.0210) * Electricity_Price[timestep]
                          - eta * (my_pwlf[1][0] * x1 + my_pwlf[1][1] * x2 + my_pwlf[1][2] * x3 + 
                                   my_pwlf[1][3] * x4 + my_pwlf[1][4] * x5)
                          + 10000)
    
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()
    
    X["Zone 1"][timestep + 1] = round(model.getVars()[2].x,2) + X["Zone 1"][timestep]
    x_values.append(model.getVars()[1].x)
    
    CoSimulation()
    X["Previous Temperature_0"] = ReadExcel("eplusout")["CORE_ZN:Zone Mean Air Temperature [C](TimeStep)"]
    X["Previous Temperature_1"] = ReadExcel("eplusout")["PERIMETER_ZN_1:Zone Mean Air Temperature [C](TimeStep)"]
    X["Previous Temperature_2"] = ReadExcel("eplusout")["PERIMETER_ZN_2:Zone Mean Air Temperature [C](TimeStep)"]
    X["Previous Temperature_3"] = ReadExcel("eplusout")["PERIMETER_ZN_3:Zone Mean Air Temperature [C](TimeStep)"]
    X["Previous Temperature_4"] = ReadExcel("eplusout")["PERIMETER_ZN_4:Zone Mean Air Temperature [C](TimeStep)"]

#%%
X.to_csv("X_FirstandSecond_SettDif_2.csv")

#%% Results evaluation, inside temperature, setpoint, outdoor temperature
# create figure and axis objects with subplots()
# X = pd.read_csv("X_firstandSecond.csv")
fig,ax = plt.subplots()
ax.plot(X['Previous Temperature_1'].groupby(np.arange(96)//4).mean(), color="grey", marker="o")
ax.step(range(24),X['Zone 1'].groupby(np.arange(96)//4).mean(),color="blue",marker="o", where = "mid")
ax.set_xlabel("hour",fontsize=14)
ax.set_ylabel("Temperature",fontsize=14)
ax.legend(["Indoor temperature","Setpoint"],loc='upper left')
ax.set_xticks(range(24))
ax.set_yticks(np.arange(20,30,1))
ax.axhline(y = 20.57, color='r', linestyle='--')
ax.axhline(y = 27.09, color='r', linestyle='--')
# ax.axhline(y = 21, color='g', linestyle='--')
ax.axhline(y = 23.83, c = 'g', ls='--')
# ax.axhline(y = 24, c = 'r', ls='--')
# # twin object for two different y-axis on the sample plot
# ax2=ax.twinx()
# ax2.plot(Electricity_Price.groupby(np.arange(96)//4).mean(),color="g",marker="o")
# ax2.plot(X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(np.arange(96)//4).mean(),color="green",marker="o")
# ax2.tick_params(labelcolor="green")
# ax2.set_ylabel("Electricity Price", fontsize = 14, color = "green")

# fig.savefig('_Zone1_Firstandthird.png',dpi=1800,bbox_inches='tight')
#%%
fig,ax = plt.subplots()
ax.plot(Electricity_Price.groupby(np.arange(96)//4).mean(),color="blue",marker="o")
ax.set_xlabel("hour",fontsize=14)
ax.set_ylabel("Electricity Price",fontsize=14, c = "b")
ax.set_xticks(range(24))
# ax.tick_params(ylabelcolor = "blue")
ax2=ax.twinx()
ax2.plot(X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(np.arange(96)//4).mean(),color="green",marker="o")
ax2.tick_params(labelcolor="green")
ax2.set_ylabel("Outdoor temperature", c = "g",fontsize=14)
# plt.savefig("Outdoor and price",dpi = 1800)
#%%
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
#%% Result Evaluation
Dataset = ReadExcel("1_OfficeSmall_[4, 5, 6, 8, 9]")

# Total Energy usage
Energy = 0
Energy = (Dataset['PSZ-AC:1:Air System Electricity Energy [J](TimeStep)'].sum() + Dataset['PSZ-AC:2:Air System Electricity Energy [J](TimeStep)'].sum() + 
    Dataset['PSZ-AC:3:Air System Electricity Energy [J](TimeStep)'].sum() + Dataset['PSZ-AC:4:Air System Electricity Energy [J](TimeStep)'].sum() + 
    Dataset['PSZ-AC:5:Air System Electricity Energy [J](TimeStep) '].sum())
print("Energy: ")
print(Energy)

# Electricity cost
Cost = (Dataset['PSZ-AC:1:Air System Electricity Energy [J](TimeStep)'] + Dataset['PSZ-AC:2:Air System Electricity Energy [J](TimeStep)'] +
    Dataset['PSZ-AC:3:Air System Electricity Energy [J](TimeStep)'] + Dataset['PSZ-AC:4:Air System Electricity Energy [J](TimeStep)'] + 
    Dataset['PSZ-AC:5:Air System Electricity Energy [J](TimeStep) ']) 
Cost *= Electricity_Price
print("Cost: ")
print(Cost.sum())

# Peak reduction
Peak = 0
Peak += Dataset['PSZ-AC:1:Air System Electricity Energy [J](TimeStep)'][52:64].sum()
Peak += Dataset['PSZ-AC:2:Air System Electricity Energy [J](TimeStep)'][52:64].sum()
Peak += Dataset['PSZ-AC:3:Air System Electricity Energy [J](TimeStep)'][52:64].sum()
Peak += Dataset['PSZ-AC:4:Air System Electricity Energy [J](TimeStep)'][52:64].sum()
Peak += Dataset['PSZ-AC:5:Air System Electricity Energy [J](TimeStep) '][52:64].sum()
print("Peak: ")
print(Peak)

# Comfort values
Dataset["Dataset_Comfort_1"] = None
Dataset["Dataset_Comfort_2"] = None
Dataset["Dataset_Comfort_3"] = None
Dataset["Dataset_Comfort_4"] = None
Dataset["Dataset_Comfort_5"] = None


Agents = [4, 5, 6, 8, 9]

for t in range(96):
    temp = np.round(Dataset['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'][t],1)
    i = Agents[0]
    if (temp > 30 or temp < 18):
        Dataset["Dataset_Comfort_1"][t] = 0
    else:
        Dataset["Dataset_Comfort_1"][t] = float(Profiles_Dataset[str("Probability" + str(i))][Profiles_Dataset[str("Temperature" + str(i))] == temp])

    temp = np.round(Dataset['PERIMETER_ZN_1:Zone Mean Air Temperature [C](TimeStep)'][t],1)
    i = Agents[1]
    if (temp > 30 or temp < 18):
        Dataset["Dataset_Comfort_2"][t] = 0
    else:
        Dataset["Dataset_Comfort_2"][t] = float(Profiles_Dataset[str("Probability" + str(i))][Profiles_Dataset[str("Temperature" + str(i))] == temp])
      
    temp = np.round(Dataset['PERIMETER_ZN_2:Zone Mean Air Temperature [C](TimeStep)'][t],1)
    i = Agents[2]
    if (temp > 30 or temp < 18):
        Dataset["Dataset_Comfort_3"][t] = 0
    else:
        Dataset["Dataset_Comfort_3"][t] = float(Profiles_Dataset[str("Probability" + str(i))][Profiles_Dataset[str("Temperature" + str(i))] == temp])
    
    temp = np.round(Dataset['PERIMETER_ZN_3:Zone Mean Air Temperature [C](TimeStep)'][t],1)
    i = Agents[3]
    if (temp > 30 or temp < 18):
        Dataset["Dataset_Comfort_4"][t] = 0
    else:
        Dataset["Dataset_Comfort_4"][t] = float(Profiles_Dataset[str("Probability" + str(i))][Profiles_Dataset[str("Temperature" + str(i))] == temp])
    
    temp = np.round(Dataset['PERIMETER_ZN_4:Zone Mean Air Temperature [C](TimeStep)'][t],1)
    i = Agents[4]
    if (temp > 30 or temp < 18):
        Dataset["Dataset_Comfort_5"][t] = 0
    else:
        Dataset["Dataset_Comfort_5"][t] = float(Profiles_Dataset[str("Probability" + str(i))][Profiles_Dataset[str("Temperature" + str(i))] == temp])

Comfort = (Dataset["Dataset_Comfort_1"].sum() + Dataset["Dataset_Comfort_2"].sum() + Dataset["Dataset_Comfort_3"].sum() + Dataset["Dataset_Comfort_4"].sum() +
Dataset["Dataset_Comfort_5"].sum())/96/5
print("Comfort: ")
print(Comfort)

#%%
Dataset = ReadExcel("1_OfficeSmall_[1, 8, 10, 14, 15]")
plt.plot(Dataset['PSZ-AC:1:Air System Electricity Energy [J](TimeStep)'].groupby(np.arange(96)//4).mean(),color="blue",marker="o")
Dataset = ReadExcel("OfficeSmall_22.77")
plt.plot(Dataset['PSZ-AC:1:Air System Electricity Energy [J](TimeStep)'].groupby(np.arange(96)//4).sum(), color="red", marker="o")

# fig,ax = plt.subplots()
# temp = ReadExcel("1_OfficeSmall_[1, 8, 10, 14, 15]")
# ax.step(range(24),temp['PSZ-AC:1:Air System Electricity Energy [J](TimeStep)'].groupby(np.arange(96)//4).mean(),color="blue",marker="o", where = "mid")
# ax.set_xlabel("hour",fontsize=14)
# ax.set_ylabel("Thermostat setpoint",fontsize=14)
# # twin object for two different y-axis on the sample plot
# temp = ReadExcel("OfficeSmall_22.77")
# ax2=ax.twinx()
# ax2.plot(temp['PSZ-AC:2:Air System Electricity Energy [J](TimeStep)'].groupby(np.arange(96)//4).sum(), color="red", marker="o")
# ax2.tick_params(labelcolor="red")
# ax2.set_xticks(range(24))
# ax2.set_ylabel("Energy usage (kWh)", fontsize = 14, color = "red")

# fig.savefig('1_1.jpg',format='jpeg',dpi=1000,bbox_inches='tight')

#%%% Plotting the profiles at each zone
plt.plot(Profiles_Dataset["Temperature" + str(Agents[0])],Profiles_Dataset["Probability" + str(Agents[0])])
plt.plot(Profiles_Dataset["Temperature" + str(Agents[0])],Profiles_Dataset["Probability" + str(Agents[1])])
plt.plot(Profiles_Dataset["Temperature" + str(Agents[0])],Profiles_Dataset["Probability" + str(Agents[2])])
plt.plot(Profiles_Dataset["Temperature" + str(Agents[0])],Profiles_Dataset["Probability" + str(Agents[3])])
plt.plot(Profiles_Dataset["Temperature" + str(Agents[0])],Profiles_Dataset["Probability" + str(Agents[4])])
plt.legend(["Zone 1","Zone 2","Zone 3","Zone 4","Zone 5"])

#%%
plt.plot(Profiles_Dataset["Temperature" + str(Agents[0][0])],Profiles_Dataset["Probability8"])
plt.xlabel("Temperature")
plt.ylabel("Comfort probability")
plt.title("Comfort probability for subject #8")
plt.savefig("1.png", dpi = 2000)

#%%
#%%
plt.plot(Electricity_Price.groupby(np.arange(96)//4).mean(),color="green",marker="o")
plt.xlabel("Time")
plt.ylabel("Electricity Price ($/kWh)")
plt.title("Electricity price")
plt.savefig('1_1.jpg',format='jpeg',dpi=1000,bbox_inches='tight')

#%% Plotting objective function
X = ReadExcel("OfficeSmall_main")

timestep = 2
u = np.linspace(18,30,1201)



z = (0.0616 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep] + 
                          0.3391 * X["PERIMETER_ZN_1:Zone Mean Air Temperature [C](TimeStep)"][timestep] - 
                          0.3725 * u) * Electricity_Price[timestep]
plt.plot(u,z)

    
# fig, ax = plt.subplots(1,2,figsize=(10,4))
# ax[0].plot(u,z)
# ax[0].legend(["Objective function-" + str(Agents[0]) + " & " + str(Agents[1]) + " & "  + str(Agents[2]) + " & " + str(Agents[3])])

# ax[1].plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[0])])
# ax[1].plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[1])])
# ax[1].plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[2])])
# ax[1].plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents[3])])
# ax[1].legend(["Profile " + str(Agents[0]),"Profile " + str(Agents[1]),"Profile " + str(Agents[2]),"Profile " + str(Agents[3])])

# plt.savefig("26.png",dpi = 1800)

#%% Controlling for other zones

    #### Problem formulation for zone 1
    model = gp.Model("optim")
    z = model.addVar(name="z") # value of the objective function
    x = model.addVar(name="x") # next temperature
    u = model.addVar(name="u") # setpoint of the building
    model.addConstr(u >= 16)
    # Adding constraints
    model.addConstr(x == 0.7291 * X["Previous Temperature_0"][timestep] + 0.2167 * u + 
                    0.0417 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep])
    my_pwlf = approx_PWLF(0)
    model.addConstr(x <= my_pwlf[2])
    model.addConstr(x >= my_pwlf[3])
    
    # Auxilary varialbes
    x1 = model.addVar(name="x1") 
    x2 = model.addVar(name="x2")
    x3 = model.addVar(name="x3")
    x4 = model.addVar(name="x4")
    x5 = model.addVar(name="x5")   
    model.addConstr(x == my_pwlf[0][0] * x1 + my_pwlf[0][1] * x2 + my_pwlf[0][2] * x3 + my_pwlf[0][3] * x4 + my_pwlf[0][4] * x5)
    model.addConstr(x1 + x2 + x3 + x4 + x5 == 1)
    model.addSOS(GRB.SOS_TYPE2, [x1, x2 , x3, x4, x5])
    
    # keeping temperature between 21 - 25
    # Lower than 21
    T1_plus = model.addVar(name="T1_plus")
    T1_minus = model.addVar(name="T1_minus")   
    model.addConstr(x - 25 == T1_plus - T1_minus)
    # higher than 21
    T2_plus = model.addVar(name="T2_plus")
    T2_minus = model.addVar(name="T2_minus")   
    model.addConstr(x - 21 == T2_plus - T2_minus)
    
    # defining opjective function   
    model.addConstr(z == ([0.0493, -0.026, -0.0003] @ np.array(X.iloc[0][1:4]) + 0.0172 * X["CORE_ZN:Zone People Occupant Count [](TimeStep)"][timestep] +
                        0.1184 * X["Previous Temperature_1"][timestep] - 0.1199 * u) * Electricity_Price[timestep]
                          - eta * (y_breaks_1[0] * x1 + y_breaks_1[1] * x2 + y_breaks_1[2] * x3 + y_breaks_1[3] * x4  + y_breaks_1[4] * x5) 
                          - eta_prime * (T1_plus + T2_minus)
                          + 10000)
    
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()
    X["CORE_ZN:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"][timestep] = model.getVars()[2].x
    
    
    ##########################################################################
    #### Problem formulation for zone 3
    model = gp.Model("optim")
    z = model.addVar(name="z") # value of the objective function
    x = model.addVar(name="x") # next temperature
    u = model.addVar(name="u") # setpoint of the building
    model.addConstr(u >= 16)
    # Adding constraints
    model.addConstr(x == 0.3305 * X["Previous Temperature_3"][timestep] + 0.6436 * u + 
                    0.0052 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep])
    model.addConstr(x <= maxConstraint_3)
    model.addConstr(x >= ThermalPreference_3)
    
    # Auxilary varialbes
    x1 = model.addVar(name="x1") 
    x2 = model.addVar(name="x2")
    x3 = model.addVar(name="x3")
    x4 = model.addVar(name="x4")
    x5 = model.addVar(name="x5")
    model.addConstr(x == breaks_3[0] * x1 + breaks_3[1] * x2 + breaks_3[2] * x3 + breaks_3[3] * x4 + breaks_3[4] * x5)
    model.addConstr(x1 + x2 + x3 + x4 + x5 == 1)
    model.addSOS(GRB.SOS_TYPE2, [x1, x2 , x3, x4, x5])

    # keeping temperature between 21 - 25
    # Lower than 21
    T1_plus = model.addVar(name="T1_plus")
    T1_minus = model.addVar(name="T1_minus")   
    model.addConstr(x - 25 == T1_plus - T1_minus)
    # higher than 21
    T2_plus = model.addVar(name="T2_plus")
    T2_minus = model.addVar(name="T2_minus")   
    model.addConstr(x - 21 == T2_plus - T2_minus)
    
    # defining opjective function    
    model.addConstr(z == ([.04017, -0.0185, -0.0025] @ np.array(X.iloc[0][1:4]) + 0.0235 * X['PERIMETER_ZN_2:Zone People Occupant Count [](TimeStep)'][timestep] +
                        0.1222 * X["Previous Temperature_3"][timestep] - 0.1204 * u) * Electricity_Price[timestep]
                          - eta * (y_breaks_3[0] * x1 + y_breaks_3[1] * x2 + y_breaks_3[2] * x3 + y_breaks_3[3] * x4 + y_breaks_3[4] * x5) 
                          - eta_prime * (T1_plus + T2_minus)
                          + 10000)
                          
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()
    X["PERIMETER_ZN_2:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"][timestep] = model.getVars()[2].x
    
    ##########################################################################
    #### Problem formulation for zone 4
    model = gp.Model("optim")
    z = model.addVar(name="z") # value of the objective function
    x = model.addVar(name="x") # next temperature
    u = model.addVar(name="u") # setpoint of the building
    model.addConstr(u >= 16)
    # Adding constraints
    model.addConstr(x == 0.3345 * X["Previous Temperature_4"][timestep] + 0.6147 * u + 
                    0.0197 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep])
    model.addConstr(x <= maxConstraint_4)
    model.addConstr(x >= ThermalPreference_4)
    
    # Auxilary varialbes
    x1 = model.addVar(name="x1") 
    x2 = model.addVar(name="x2")
    x3 = model.addVar(name="x3")
    x4 = model.addVar(name="x4")
    x5 = model.addVar(name="x5")
    model.addConstr(x == breaks_4[0] * x1 + breaks_4[1] * x2 + breaks_4[2] * x3 + breaks_4[3] * x4 + breaks_4[4] * x5)
    model.addConstr(x1 + x2 + x3 + x4 + x5 == 1)
    model.addSOS(GRB.SOS_TYPE2, [x1, x2 , x3, x4, x5])
   
    # keeping temperature between 21 - 25
    # Lower than 21
    T1_plus = model.addVar(name="T1_plus")
    T1_minus = model.addVar(name="T1_minus")   
    model.addConstr(x - 25 == T1_plus - T1_minus)
    # higher than 21
    T2_plus = model.addVar(name="T2_plus")
    T2_minus = model.addVar(name="T2_minus")   
    model.addConstr(x - 21 == T2_plus - T2_minus)
    
    # defining opjective function
    model.addConstr(z == ([0.0540, -0.0247, -0.003] @ np.array(X.iloc[0][1:4]) + 0.0215 * X['PERIMETER_ZN_3:Zone People Occupant Count [](TimeStep)'][timestep] +
                        0.1475 * X["Previous Temperature_4"][timestep] - 0.1479 * u) * Electricity_Price[timestep]
                          - eta * (y_breaks_4[0] * x1 + y_breaks_4[1] * x2 + y_breaks_4[2] * x3 + y_breaks_4[3] * x4 + y_breaks_4[4] * x5) 
                          - eta_prime * (T1_plus + T2_minus)
                          + 10000)
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()
    X["PERIMETER_ZN_3:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"][timestep] = model.getVars()[2].x
    
    ##########################################################################
    #### Problem formulation for zone 5
    model = gp.Model("optim")
    z = model.addVar(name="z") # value of the objective function
    x = model.addVar(name="x") # next temperature
    u = model.addVar(name="u") # setpoint of the building
    model.addConstr(u >= 16)
    # Adding constraints
    model.addConstr(x == 0.3915 * X["Previous Temperature_5"][timestep] + 0.5618 * u + 
                    0.0178 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep])
    model.addConstr(x <= maxConstraint_5)
    model.addConstr(x >= ThermalPreference_5)
    
    # Auxilary varialbes
    x1 = model.addVar(name="x1") 
    x2 = model.addVar(name="x2")
    x3 = model.addVar(name="x3")
    x4 = model.addVar(name="x4")
    x5 = model.addVar(name="x5")
    model.addConstr(x == breaks_5[0] * x1 + breaks_5[1] * x2 + breaks_5[2] * x3 + breaks_5[3] * x4 + breaks_5[4] * x5)
    model.addConstr(x1 + x2 + x3 + x4 + x5 == 1)
    model.addSOS(GRB.SOS_TYPE2, [x1, x2 , x3, x4, x5])
   
    # keeping temperature between 21 - 25
    # Lower than 21
    T1_plus = model.addVar(name="T1_plus")
    T1_minus = model.addVar(name="T1_minus")   
    model.addConstr(x - 25 == T1_plus - T1_minus)
    # higher than 21
    T2_plus = model.addVar(name="T2_plus")
    T2_minus = model.addVar(name="T2_minus")   
    model.addConstr(x - 21 == T2_plus - T2_minus)
    
    # defining opjective function
    model.addConstr(z == ([0.0368, -0.0153, -0.0001] @ np.array(X.iloc[0][1:4]) + 0.0107 * X['PERIMETER_ZN_4:Zone People Occupant Count [](TimeStep)'][timestep] +
                    0.0857 * X["Previous Temperature_5"][timestep] - 0.0854 * u) * Electricity_Price[timestep]
                      - eta * (y_breaks_5[0] * x1 + y_breaks_5[1] * x2 + y_breaks_5[2] * x3 + y_breaks_5[3] * x4 + y_breaks_5[4] * x5) 
                      - eta_prime * (T1_plus + T2_minus)
                      + 10000)
        
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()
    X["PERIMETER_ZN_4:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"][timestep] = model.getVars()[2].x
