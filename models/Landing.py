import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def run_simulation():

    def system_dynamics(v, t, m, c, g, F0, a):
        dvdt = (m*g-F0*np.exp(-a*t)-c*v)/m
        return dvdt # Возвращаем производные
    def oil_pendulum():
        # Параметры
        m, c, g = 80.0, 15.0, 9.81
        F0, a = 500, 0.2
        
        # Начальное состояние
        v0 = 0
        v_as = g*m/c
        
        # Сетка времени (теперь это просто точки, в которых мы хотим ПОЛУЧИТЬ ответ)
        t = np.linspace(0, 30, 10000)

        # 2. РЕШАЕМ в одну строку
        # Аргументы: функция, нач. состояние, время, дополнительные параметры (args)
        states = odeint(system_dynamics, v0, t, args=(m, c, g, F0, a))        

        # states — это матрица, где колонки — это X, Y и Z
        v = states[:, 0]

        v_analytic=g*m/c * (1-np.exp(-c*t/m))+F0/(c-a*m)*(np.exp(-c*t/m)-np.exp(-t*a))

        # Визуализация
        plt.figure(figsize=(12, 6))
        plt.plot(t, v, color='blue', label='Скорость (v)')
        plt.plot(t, v_analytic, label='Аналитика (v)', alpha=0.7)
        plt.axvline(y=v_as, color='r', linestyle='--', label='Максимальная скорость (ас.)')
        plt.title("Десант")
        plt.xlabel("Время (t)")
        plt.ylabel("Скорость (x)")
        plt.grid(True)
        plt.legend()
        plt.show()

    oil_pendulum()