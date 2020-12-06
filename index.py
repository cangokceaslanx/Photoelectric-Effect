import pandas as pd
import numpy as np
import matplotlib.pyplot as mt
from scipy.stats import linregress
import math
data = pd.read_csv("./dataphoto.csv")
amplifier = np.array(data.Amplifier)
current = np.array(data.I)
voltage = np.array(data.V)
renk = np.array(data.Renk)
scale = np.array(data.Scale)
kirmizivoltage = []
kirmizicurrentpositive = []
kirmizicurrentnegative = []
kirmizivoltagepositive = []
kirmizivoltagenegative = []
def sigmaX(m1,m2,n1,n2,sigmam1,sigmam2):
    sigman1=0.2
    sigman2=0.2
    part1 = ((((1)/(m1-m2))**2)*(sigman2**2))
    part2 = ((((1)/(m1-m2))**2)*(sigman1**2))
    part3 = ((((n2-n1)*(((1)/(m1-m2))**2))**2)*sigmam1**2)
    part4 = ((((n2-n1)*(((1)/(m1-m2))**2))**2)*sigmam2**2)
    return math.sqrt(part1+part2+part3+part4)

def intersection(m1,m2,n1,n2):
    return (n2-n1)/(m1-m2)
stderrorVoltagePositive, stderrorCurrentPositive,stderrorVoltageNegative, stderrorCurrentNegative = [],[],[],[]
for i in range(len(voltage)):
    if(renk[i] == "M"):
        kirmizivoltage.append(voltage[i])
        if(current[i]>=0):
            kirmizicurrentpositive.append(((current[i]*scale[i]))/10**((abs(amplifier[i])+6)-14))
            kirmizivoltagepositive.append(voltage[i])
            stderrorVoltagePositive.append(0.001)
            stderrorCurrentPositive.append(0.2 * 100 /10**((abs(amplifier[i])+6)-14))
        else:
            kirmizicurrentnegative.append(((current[i]*scale[i]))/10**((abs(amplifier[i])+6)-14))
            kirmizivoltagenegative.append(voltage[i])
            stderrorVoltageNegative.append(0.001)
            stderrorCurrentNegative.append(0.2 * 100/10**((abs(amplifier[i])+6)-14))
slope,intercept,rvalue,pvalue,stderr=linregress(kirmizivoltagepositive,kirmizicurrentpositive)
fit=np.polyfit(kirmizivoltagepositive,kirmizicurrentpositive,1)
bfl=np.poly1d(fit)
slope1,intercept1,rvalue1,pvalue1,stderr1=linregress(kirmizivoltagenegative,kirmizicurrentnegative)
fit1=np.polyfit(kirmizivoltagenegative,kirmizicurrentnegative,2)
bfl1=np.poly1d(fit1)
interset = intersection(slope,slope1,intercept,intercept1)
stdVstop = sigmaX(slope,slope1,intercept,intercept1,stderr,stderr1)
Vstopping1 = np.array([0.359,0.317,0.397,0.567,0.844])
Vstopping = Vstopping1 * 1.6
sigmaVstopping = np.array([0.059,0.044,0.059,0.063,0.062])
mercury = [4.29E+14,5.18E+14,5.49E+14,6.19E+14,6.82E+14]
#mt.scatter(mercury,Vstopping)
#mt.errorbar(mercury,Vstopping,yerr=sigmaVstopping,linestyle="None")
slope2,intercept2,rvalue2,pvalue2,stderr2=linregress(mercury,Vstopping)
fit2=np.polyfit(mercury,Vstopping,1)
bfl2=np.poly1d(fit2)
#mt.plot(mercury,bfl2(mercury),color="black")
mt.scatter(kirmizivoltagepositive,kirmizicurrentpositive,color="green")
mt.scatter(kirmizivoltagenegative,kirmizicurrentnegative,color="red")
mt.plot(kirmizivoltagepositive,bfl(kirmizivoltagepositive),color="black")
mt.plot([0.7,1.2],bfl1([0.7,1.2]),color="black")
mt.errorbar(kirmizivoltagepositive,kirmizicurrentpositive,xerr=stderrorVoltagePositive,yerr=stderrorCurrentPositive,marker='s',linestyle="None")
mt.errorbar(kirmizivoltagenegative,kirmizicurrentnegative,xerr=stderrorVoltageNegative,yerr=stderrorCurrentNegative,marker='s',linestyle="None")
h = (0.80e-18 + 1.010e-18)/(5.5*10e14)
mt.grid()
mt.show()