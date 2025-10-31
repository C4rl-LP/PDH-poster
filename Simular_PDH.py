import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv 
from scipy.signal import butter, filtfilt
# Características da cavidade
c = 299792458.0  # m/s
L = 0.15         # m (15 cm)
FSR = c / (2 * L)

r1 = 0.97
R1 = r1**2
T1 = 1 - R1
t1 = np.sqrt(T1)

r2 = 0.97
R2 = r2**2
T2 = 1 - R2
t2 = np.sqrt(T2)
F = np.pi * np.sqrt(np.sqrt(R1 * R2)) / (1 - np.sqrt(R1 * R2))
FWHM = FSR / F
Omega_m =  10e6 # mod freq
beta = 0.3                # indice de modulação

# Finesse e largura de linha


# Função de transferência de refletividade no Fabry–Perot
def H(omega):
    phi = 2 * np.pi * omega / FSR
    return r1 - (t1 * t2 * r2 * np.exp(-1j * phi)) / (1 - r1 * r2 * np.exp(-1j * phi))

# Varredura em unidades de FSR
omega = np.linspace(-1.5 * FSR, 1.5 * FSR, 3000)
R = np.abs(H(omega))**2  # Potência refletida normalizada

# Gráfico
plt.figure(figsize=(8,4 ))
plt.plot(omega / FSR, R, 'b')
plt.title('Potência Refletida no Fabry–Perot')
plt.xlabel(f'Deslocamento de frequência (unidades de FSR = {FSR/10**9 :.2f} GHz)')
plt.ylabel('Potência refletida (normalizada) |H(ω)|²')
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.show()

print(FWHM, Omega_m ,FWHM/Omega_m) 

def E_ref(t, om):
    return [H(om)*jv(0, beta)*np.exp(1j*om*t) + H(om+Omega_m)*jv(1, beta)*np.exp(1j*(om+Omega_m)*t) - H(om-Omega_m)*jv(1, beta)*np.exp(1j*(om-Omega_m)*t)]

def P_detect(om, t):
    return np.abs(E_ref(t,om))**2

def mixer_lowpass(signal, t, Omega_m, fc=1e5, fs=None):
    """
    Mistura o sinal com o oscilador local cos(Omega_m t)
    e aplica um filtro passa-baixa Butterworth.
    """
    if fs is None:
        dt = t[1] - t[0]
        fs = 1 / dt
    
    if Omega_m > FWHM:
        lo = np.sin(Omega_m * t)
    else:
        lo = np.cos(Omega_m * t)
    mixed = signal * lo

    # filtro passa-baixa
    b, a = butter(4, fc / (0.5 * fs))
    filtered = filtfilt(b, a, mixed)
    return filtered


# ==============================
# 6. SIMULAÇÃO DO SINAL PDH
# ==============================
fs = 200e6  # taxa de amostragem
t = np.arange(0, 20e-6, 1/fs)  # 20 µs

# Frequência angular da portadora sendo varrida
detunings = np.linspace(-FSR/13, FSR/13, 1000)  # em Hz
PDH_error = []

def erro_aprox(om):
    phi = 2 * np.pi * om / FSR
    numerator = 4 * r1 * r2 * np.sin(phi) * (L / c) * (1 - r1**2) * (1 - r2**2)


    denominator_base = (1 - r1 * r2)**2 + 4 * r1 * r2 * (np.sin(phi))**2
    denominator = denominator_base**2

    return numerator / denominator * Omega_m * beta
 







for delta in detunings:
    omega = 2 * np.pi * ( delta)  # frequência angular
    P = P_detect(omega, t)
    err = mixer_lowpass(P - np.mean(P), t, Omega_m, fc=1e6, fs=fs)
    PDH_error.append(np.mean(err))  # valor DC do sinal filtrado
plt.figure(figsize=(7,5))
#plt.plot(detunings/ FWHM, erro_aprox(detunings), label = 'sla')
plt.plot(detunings / FSR, PDH_error)
plt.legend()
plt.axhline(0, color='k', lw=0.8)
plt.xlabel("Desvio de frequência (Δν / FSR)")
plt.ylabel("Sinal de erro PDH (a.u.)")
plt.title("Simulação do sinal Pound–Drever–Hall")
plt.grid(True)
plt.show()









