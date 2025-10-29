import numpy as np
import matplotlib.pyplot as plt

# Caracterisicas da cavidade

c = 299792458.0 # m/s

L= 0.15 #m 15cm
FSR = c/(2*L)
r1 = 0.9
R1 = r1**2 
T1 = 1 - R1
t1 = np.sqrt(T1)
r2 = 0.9
R2 = r1**2
T2 = 1 - R2
t2 = np.sqrt(T2)

# A finesse para R ~ 1 é mais ou menos

F = np.pi * np.sqrt(np.sqrt(R1 * R2 ))/(1- np.sqrt(R1 * R2)) # se R1 = R2  F = pi sqrt(R)/ 1-R
# F = FSR/FWHM 
FWHM = FSR/F

# função de tranferencia de refletividade no FPI

def H(omega):
    phi = 2*np.pi * omega/FSR
    return r1 - (t1*t2*r2*np.exp(-2j*phi))/(1- r1*r2*np.exp(-2j*phi))

