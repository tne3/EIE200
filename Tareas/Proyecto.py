import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
La = 0.01  # H
Ra = 0.5   # Ω
J = 0.01   # kg·m²
B = 0.001  # N·m·s/rad
Ka = 0.01  # V·s/rad
Km = 0.01  # N·m/A
Vt = 24.0  # V

# Condiciones iniciales
ia0 = 0.0
wm0 = 0.0
theta0 = 0.0

# Tiempo de simulación
t0 = 0.0
tf = 0.5
dt = 0.01
N = int((tf - t0) / dt)

# Tiempo de simulación para referencia
dt_ref = 0.0001
N_ref = int((tf - t0) / dt_ref)
t_ref = np.linspace(t0, tf, N_ref + 1)

# Definición de funciones
def f_ia(ia, wm, Vt, Ra):
    return (Vt - Ra * ia - Ka * wm) / La

def f_wm(ia, wm):
    return (Km * ia - B * wm) / J

def f_theta(wm):
    return wm

# Método de Ralston
def ralston_step(y, f, dt):
    k1 = f(y)
    k2 = f(y + (2 / 3) * dt * k1)
    return y + dt * (k1 + 3 * k2) / 4

# Métodos numéricos
def rk2_step(y, f, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    return y + dt * k2

def rk3_step(y, f, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + dt * (k1 + 4 * k2) / 6)
    return y + dt * (k1 + 4 * k2 + k3) / 6

def rk4_step(y, f, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Simulación de referencia usando Ralston con un paso de tiempo pequeño
ia_ref = np.zeros(N_ref + 1)
wm_ref = np.zeros(N_ref + 1)
theta_ref = np.zeros(N_ref + 1)
torque_ref = np.zeros(N_ref + 1)
ia_ref[0] = ia0
wm_ref[0] = wm0
theta_ref[0] = theta0
torque_ref[0] = Km * ia0

for i in range(N_ref):
    ia_ref[i + 1] = ralston_step(ia_ref[i], lambda ia: f_ia(ia, wm_ref[i], Vt, Ra), dt_ref)
    wm_ref[i + 1] = ralston_step(wm_ref[i], lambda wm: f_wm(ia_ref[i], wm), dt_ref)
    theta_ref[i + 1] = ralston_step(theta_ref[i], lambda theta: f_theta(wm_ref[i]), dt_ref)
    torque_ref[i + 1] = Km * ia_ref[i + 1]

# Arrays para almacenar soluciones
t = np.linspace(t0, tf, N + 1)
ia_rk2 = np.zeros(N + 1)
wm_rk2 = np.zeros(N + 1)
theta_rk2 = np.zeros(N + 1)
torque_rk2 = np.zeros(N + 1)
ia_rk3 = np.zeros(N + 1)
wm_rk3 = np.zeros(N + 1)
theta_rk3 = np.zeros(N + 1)
torque_rk3 = np.zeros(N + 1)
ia_rk4 = np.zeros(N + 1)
wm_rk4 = np.zeros(N + 1)
theta_rk4 = np.zeros(N + 1)
torque_rk4 = np.zeros(N + 1)

# Condiciones iniciales
ia_rk2[0] = ia0
wm_rk2[0] = wm0
theta_rk2[0] = theta0
torque_rk2[0] = Km * ia0
ia_rk3[0] = ia0
wm_rk3[0] = wm0
theta_rk3[0] = theta0
torque_rk3[0] = Km * ia0
ia_rk4[0] = ia0
wm_rk4[0] = wm0
theta_rk4[0] = theta0
torque_rk4[0] = Km * ia0

# Simulación
for i in range(N):
    ia_rk2[i + 1] = rk2_step(ia_rk2[i], lambda ia: f_ia(ia, wm_rk2[i], Vt, Ra), dt)
    wm_rk2[i + 1] = rk2_step(wm_rk2[i], lambda wm: f_wm(ia_rk2[i], wm), dt)
    theta_rk2[i + 1] = rk2_step(theta_rk2[i], lambda theta: f_theta(wm_rk2[i]), dt)
    torque_rk2[i + 1] = Km * ia_rk2[i + 1]

    ia_rk3[i + 1] = rk3_step(ia_rk3[i], lambda ia: f_ia(ia, wm_rk3[i], Vt, Ra), dt)
    wm_rk3[i + 1] = rk3_step(wm_rk3[i], lambda wm: f_wm(ia_rk3[i], wm), dt)
    theta_rk3[i + 1] = rk3_step(theta_rk3[i], lambda theta: f_theta(wm_rk3[i]), dt)
    torque_rk3[i + 1] = Km * ia_rk3[i + 1]

    ia_rk4[i + 1] = rk4_step(ia_rk4[i], lambda ia: f_ia(ia, wm_rk4[i], Vt, Ra), dt)
    wm_rk4[i + 1] = rk4_step(wm_rk4[i], lambda wm: f_wm(ia_rk4[i], wm), dt)
    theta_rk4[i + 1] = rk4_step(theta_rk4[i], lambda theta: f_theta(wm_rk4[i]), dt)
    torque_rk4[i + 1] = Km * ia_rk4[i + 1]

# Cálculo de errores respecto a la referencia
ia_ref_interp = np.interp(t, t_ref, ia_ref)
wm_ref_interp = np.interp(t, t_ref, wm_ref)
theta_ref_interp = np.interp(t, t_ref, theta_ref)
torque_ref_interp = np.interp(t, t_ref, torque_ref)

error_ia_rk2 = np.abs(ia_rk2 - ia_ref_interp)
error_wm_rk2 = np.abs(wm_rk2 - wm_ref_interp)
error_theta_rk2 = np.abs(theta_rk2 - theta_ref_interp)
error_torque_rk2 = np.abs(torque_rk2 - torque_ref_interp)

error_ia_rk3 = np.abs(ia_rk3 - ia_ref_interp)
error_wm_rk3 = np.abs(wm_rk3 - wm_ref_interp)
error_theta_rk3 = np.abs(theta_rk3 - theta_ref_interp)
error_torque_rk3 = np.abs(torque_rk3 - torque_ref_interp)

error_ia_rk4 = np.abs(ia_rk4 - ia_ref_interp)
error_wm_rk4 = np.abs(wm_rk4 - wm_ref_interp)
error_theta_rk4 = np.abs(theta_rk4 - theta_ref_interp)
error_torque_rk4 = np.abs(torque_rk4 - torque_ref_interp)

# Graficar resultados de soluciones
plt.figure(figsize=(12, 10))

# Corriente de Armadura
plt.subplot(4, 1, 1)
plt.plot(t, ia_rk2, label='RK2')
plt.plot(t, ia_rk3, label='RK3')
plt.plot(t, ia_rk4, label='RK4')
plt.plot(t_ref, ia_ref, label='Referencia', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente de Armadura [A]')
plt.title('Corriente de Armadura')
plt.legend()

# Velocidad Angular
plt.subplot(4, 1, 2)
plt.plot(t, wm_rk2, label='RK2')
plt.plot(t, wm_rk3, label='RK3')
plt.plot(t, wm_rk4, label='RK4')
plt.plot(t_ref, wm_ref, label='Referencia', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad Angular [rad/s]')
plt.title('Velocidad Angular')
plt.legend()

# Posición Angular
plt.subplot(4, 1, 3)
plt.plot(t, theta_rk2, label='RK2')
plt.plot(t, theta_rk3, label='RK3')
plt.plot(t, theta_rk4, label='RK4')
plt.plot(t_ref, theta_ref, label='Referencia', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Posición Angular [rad]')
plt.title('Posición Angular')
plt.legend()

# Torque
plt.subplot(4, 1, 4)
plt.plot(t, torque_rk2, label='RK2')
plt.plot(t, torque_rk3, label='RK3')
plt.plot(t, torque_rk4, label='RK4')
plt.plot(t_ref, torque_ref, label='Referencia', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque [Nm]')
plt.title('Torque')
plt.legend()

plt.tight_layout()
plt.show()

# Graficar errores absolutos comparados con la solución de referencia
plt.figure(figsize=(12, 10))

# Error en Corriente de Armadura
plt.subplot(4, 1, 1)
plt.plot(t, error_ia_rk2, label='RK2')
plt.plot(t, error_ia_rk3, label='RK3')
plt.plot(t, error_ia_rk4, label='RK4')
plt.plot(t, np.abs(ia_ref_interp - ia_ref_interp), label='Error Referencial', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error Absoluto')
plt.title('Error en Corriente de Armadura')
plt.legend()

# Error en Velocidad Angular
plt.subplot(4, 1, 2)
plt.plot(t, error_wm_rk2, label='RK2')
plt.plot(t, error_wm_rk3, label='RK3')
plt.plot(t, error_wm_rk4, label='RK4')
plt.plot(t, np.abs(wm_ref_interp - wm_ref_interp), label='Error Referencial', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error Absoluto')
plt.title('Error en Velocidad Angular')
plt.legend()

# Error en Posición Angular
plt.subplot(4, 1, 3)
plt.plot(t, error_theta_rk2, label='RK2')
plt.plot(t, error_theta_rk3, label='RK3')
plt.plot(t, error_theta_rk4, label='RK4')
plt.plot(t, np.abs(theta_ref_interp - theta_ref_interp), label='Error Referencial', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error Absoluto')
plt.title('Error en Posición Angular')
plt.legend()

# Error en Torque
plt.subplot(4, 1, 4)
plt.plot(t, error_torque_rk2, label='RK2')
plt.plot(t, error_torque_rk3, label='RK3')
plt.plot(t, error_torque_rk4, label='RK4')
plt.plot(t, np.abs(torque_ref_interp - torque_ref_interp), label='Error Referencial', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error Absoluto')
plt.title('Error en Torque')
plt.legend()

plt.tight_layout()
plt.show()


# Cálculo de métricas de error (error absoluto medio)
mean_error_ia_rk2 = np.mean(error_ia_rk2)
mean_error_wm_rk2 = np.mean(error_wm_rk2)
mean_error_theta_rk2 = np.mean(error_theta_rk2)
mean_error_torque_rk2 = np.mean(error_torque_rk2)

mean_error_ia_rk3 = np.mean(error_ia_rk3)
mean_error_wm_rk3 = np.mean(error_wm_rk3)
mean_error_theta_rk3 = np.mean(error_theta_rk3)
mean_error_torque_rk3 = np.mean(error_torque_rk3)

mean_error_ia_rk4 = np.mean(error_ia_rk4)
mean_error_wm_rk4 = np.mean(error_wm_rk4)
mean_error_theta_rk4 = np.mean(error_theta_rk4)
mean_error_torque_rk4 = np.mean(error_torque_rk4)

# Imprimir métricas de error
print("Métricas de error absoluto medio:")
print(f"RK2 - Corriente de Armadura: {mean_error_ia_rk2:.4f}")
print(f"RK2 - Velocidad Angular: {mean_error_wm_rk2:.4f}")
print(f"RK2 - Posición Angular: {mean_error_theta_rk2:.4f}")
print(f"RK2 - Torque: {mean_error_torque_rk2:.4f}")

print(f"RK3 - Corriente de Armadura: {mean_error_ia_rk3:.4f}")
print(f"RK3 - Velocidad Angular: {mean_error_wm_rk3:.4f}")
print(f"RK3 - Posición Angular: {mean_error_theta_rk3:.4f}")
print(f"RK3 - Torque: {mean_error_torque_rk3:.4f}")

print(f"RK4 - Corriente de Armadura: {mean_error_ia_rk4:.4f}")
print(f"RK4 - Velocidad Angular: {mean_error_wm_rk4:.4f}")
print(f"RK4 - Posición Angular: {mean_error_theta_rk4:.4f}")
print(f"RK4 - Torque: {mean_error_torque_rk4:.4f}")

# Análisis de sensibilidad
Vt_values = np.linspace(20, 28, 5)
Ra_values = np.linspace(0.3, 0.7, 5)

sensitivity_results = {
    'Vt': [],
    'Ra': [],
    'ia_rk2': [],
    'wm_rk2': [],
    'theta_rk2': [],
    'torque_rk2': [],
    'ia_rk3': [],
    'wm_rk3': [],
    'theta_rk3': [],
    'torque_rk3': [],
    'ia_rk4': [],
    'wm_rk4': [],
    'theta_rk4': [],
    'torque_rk4': []
}

for Vt_var in Vt_values:
    for Ra_var in Ra_values:
        ia_rk2 = np.zeros(N + 1)
        wm_rk2 = np.zeros(N + 1)
        theta_rk2 = np.zeros(N + 1)
        torque_rk2 = np.zeros(N + 1)
        ia_rk3 = np.zeros(N + 1)
        wm_rk3 = np.zeros(N + 1)
        theta_rk3 = np.zeros(N + 1)
        torque_rk3 = np.zeros(N + 1)
        ia_rk4 = np.zeros(N + 1)
        wm_rk4 = np.zeros(N + 1)
        theta_rk4 = np.zeros(N + 1)
        torque_rk4 = np.zeros(N + 1)

        ia_rk2[0] = ia0
        wm_rk2[0] = wm0
        theta_rk2[0] = theta0
        torque_rk2[0] = Km * ia0
        ia_rk3[0] = ia0
        wm_rk3[0] = wm0
        theta_rk3[0] = theta0
        torque_rk3[0] = Km * ia0
        ia_rk4[0] = ia0
        wm_rk4[0] = wm0
        theta_rk4[0] = theta0
        torque_rk4[0] = Km * ia0

        for i in range(N):
            ia_rk2[i + 1] = rk2_step(ia_rk2[i], lambda ia: f_ia(ia, wm_rk2[i], Vt_var, Ra_var), dt)
            wm_rk2[i + 1] = rk2_step(wm_rk2[i], lambda wm: f_wm(ia_rk2[i], wm), dt)
            theta_rk2[i + 1] = rk2_step(theta_rk2[i], lambda theta: f_theta(wm_rk2[i]), dt)
            torque_rk2[i + 1] = Km * ia_rk2[i + 1]

            ia_rk3[i + 1] = rk3_step(ia_rk3[i], lambda ia: f_ia(ia, wm_rk3[i], Vt_var, Ra_var), dt)
            wm_rk3[i + 1] = rk3_step(wm_rk3[i], lambda wm: f_wm(ia_rk3[i], wm), dt)
            theta_rk3[i + 1] = rk3_step(theta_rk3[i], lambda theta: f_theta(wm_rk3[i]), dt)
            torque_rk3[i + 1] = Km * ia_rk3[i + 1]

            ia_rk4[i + 1] = rk4_step(ia_rk4[i], lambda ia: f_ia(ia, wm_rk4[i], Vt_var, Ra_var), dt)
            wm_rk4[i + 1] = rk4_step(wm_rk4[i], lambda wm: f_wm(ia_rk4[i], wm), dt)
            theta_rk4[i + 1] = rk4_step(theta_rk4[i], lambda theta: f_theta(wm_rk4[i]), dt)
            torque_rk4[i + 1] = Km * ia_rk4[i + 1]

        sensitivity_results['Vt'].append(Vt_var)
        sensitivity_results['Ra'].append(Ra_var)
        sensitivity_results['ia_rk2'].append(ia_rk2)
        sensitivity_results['wm_rk2'].append(wm_rk2)
        sensitivity_results['theta_rk2'].append(theta_rk2)
        sensitivity_results['torque_rk2'].append(torque_rk2)
        sensitivity_results['ia_rk3'].append(ia_rk3)
        sensitivity_results['wm_rk3'].append(wm_rk3)
        sensitivity_results['theta_rk3'].append(theta_rk3)
        sensitivity_results['torque_rk3'].append(torque_rk3)
        sensitivity_results['ia_rk4'].append(ia_rk4)
        sensitivity_results['wm_rk4'].append(wm_rk4)
        sensitivity_results['theta_rk4'].append(theta_rk4)
        sensitivity_results['torque_rk4'].append(torque_rk4)

# Graficar resultados de sensibilidad
plt.figure(figsize=(12, 12))

# Sensibilidad de ia con respecto a Vt
plt.subplot(3, 3, 1)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['ia_rk2'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente de Armadura [A]')
plt.title('Sensibilidad de Corriente de Armadura respecto a Vt (RK2)')
plt.legend()

plt.subplot(3, 3, 2)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['ia_rk3'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente de Armadura [A]')
plt.title('Sensibilidad de Corriente de Armadura respecto a Vt (RK3)')
plt.legend()

plt.subplot(3, 3, 3)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['ia_rk4'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente de Armadura [A]')
plt.title('Sensibilidad de Corriente de Armadura respecto a Vt (RK4)')
plt.legend()

# Sensibilidad de wm con respecto a Vt
plt.subplot(3, 3, 4)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['wm_rk2'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad Angular [rad/s]')
plt.title('Sensibilidad de Velocidad Angular respecto a Vt (RK2)')
plt.legend()

plt.subplot(3, 3, 5)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['wm_rk3'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad Angular [rad/s]')
plt.title('Sensibilidad de Velocidad Angular respecto a Vt (RK3)')
plt.legend()

plt.subplot(3, 3, 6)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['wm_rk4'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad Angular [rad/s]')
plt.title('Sensibilidad de Velocidad Angular respecto a Vt (RK4)')
plt.legend()

# Sensibilidad de ia con respecto a Ra
plt.subplot(3, 3, 7)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['ia_rk2'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente de Armadura [A]')
plt.title('Sensibilidad de Corriente de Armadura respecto a Ra (RK2)')
plt.legend()

plt.subplot(3, 3, 8)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['ia_rk3'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente de Armadura [A]')
plt.title('Sensibilidad de Corriente de Armadura respecto a Ra (RK3)')
plt.legend()

plt.subplot(3, 3, 9)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['ia_rk4'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente de Armadura [A]')
plt.title('Sensibilidad de Corriente de Armadura respecto a Ra (RK4)')
plt.legend()

plt.tight_layout()
plt.show()

# Sensibilidad de wm con respecto a Ra
plt.figure(figsize=(12, 12))

plt.subplot(3, 3, 1)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['wm_rk2'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad Angular [rad/s]')
plt.title('Sensibilidad de Velocidad Angular respecto a Ra (RK2)')
plt.legend()

plt.subplot(3, 3, 2)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['wm_rk3'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad Angular [rad/s]')
plt.title('Sensibilidad de Velocidad Angular respecto a Ra (RK3)')
plt.legend()

plt.subplot(3, 3, 3)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['wm_rk4'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad Angular [rad/s]')
plt.title('Sensibilidad de Velocidad Angular respecto a Ra (RK4)')
plt.legend()

# Sensibilidad del torque con respecto a Vt
plt.subplot(3, 3, 4)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['torque_rk2'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque [Nm]')
plt.title('Sensibilidad del Torque respecto a Vt (RK2)')
plt.legend()

plt.subplot(3, 3, 5)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['torque_rk3'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque [Nm]')
plt.title('Sensibilidad del Torque respecto a Vt (RK3)')
plt.legend()

plt.subplot(3, 3, 6)
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_results['torque_rk4'][i * len(Ra_values)], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque [Nm]')
plt.title('Sensibilidad del Torque respecto a Vt (RK4)')
plt.legend()

# Sensibilidad del torque con respecto a Ra
plt.subplot(3, 3, 7)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['torque_rk2'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque [Nm]')
plt.title('Sensibilidad del Torque respecto a Ra (RK2)')
plt.legend()

plt.subplot(3, 3, 8)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['torque_rk3'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque [Nm]')
plt.title('Sensibilidad del Torque respecto a Ra (RK3)')
plt.legend()

plt.subplot(3, 3, 9)
for j, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_results['torque_rk4'][j], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque [Nm]')
plt.title('Sensibilidad del Torque respecto a Ra (RK4)')
plt.legend()

plt.tight_layout()
plt.show()

# Sensibilidad respecto a Vt y Ra
Vt_values = np.arange(20.0, 30.0, 2.0)
Ra_values = [0.3, 0.399999999997, 0.5, 0.6, 0.7]

sensitivity_vt_rk4 = []
sensitivity_ra_rk4 = []

# Sensibilidad con respecto a Vt
for Vt_var in Vt_values:
    theta_sens = np.zeros(N + 1)
    ia_sens = np.zeros(N + 1)
    wm_sens = np.zeros(N + 1)
    
    ia_sens[0] = ia0
    wm_sens[0] = wm0
    theta_sens[0] = theta0
    
    for i in range(N):
        ia_sens[i + 1] = rk4_step(ia_sens[i], lambda ia: f_ia(ia, wm_sens[i], Vt_var, Ra), dt)
        wm_sens[i + 1] = rk4_step(wm_sens[i], lambda wm: f_wm(ia_sens[i], wm), dt)
        theta_sens[i + 1] = rk4_step(theta_sens[i], lambda theta: f_theta(wm_sens[i]), dt)
    
    sensitivity_vt_rk4.append(theta_sens)

# Sensibilidad con respecto a Ra
for Ra_var in Ra_values:
    theta_sens = np.zeros(N + 1)
    ia_sens = np.zeros(N + 1)
    wm_sens = np.zeros(N + 1)
    
    ia_sens[0] = ia0
    wm_sens[0] = wm0
    theta_sens[0] = theta0
    
    for i in range(N):
        ia_sens[i + 1] = rk4_step(ia_sens[i], lambda ia: f_ia(ia, wm_sens[i], Vt, Ra_var), dt)
        wm_sens[i + 1] = rk4_step(wm_sens[i], lambda wm: f_wm(ia_sens[i], wm), dt)
        theta_sens[i + 1] = rk4_step(theta_sens[i], lambda theta: f_theta(wm_sens[i]), dt)
    
    sensitivity_ra_rk4.append(theta_sens)

# Graficar resultados por separado

# Sensibilidad respecto a Vt (RK4)
plt.figure(figsize=(8, 6))
for i, Vt_var in enumerate(Vt_values):
    plt.plot(t, sensitivity_vt_rk4[i], label=f'Vt={Vt_var}V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Posición Angular [rad]')
plt.title('Sensibilidad de Posición Angular respecto a Vt (RK4)')
plt.legend()
plt.grid(True)
plt.show()

# Sensibilidad respecto a Ra (RK4)
plt.figure(figsize=(8, 6))
for i, Ra_var in enumerate(Ra_values):
    plt.plot(t, sensitivity_ra_rk4[i], label=f'Ra={Ra_var}Ω')
plt.xlabel('Tiempo [s]')
plt.ylabel('Posición Angular [rad]')
plt.title('Sensibilidad de Posición Angular respecto a Ra (RK4)')
plt.legend()
plt.grid(True)

plt.show()
# Arrays para almacenar soluciones
t = np.linspace(t0, tf, N + 1)