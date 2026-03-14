import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def run_simulation():

    def system_dynamics(state, t, m, c, k, F0, omega):
        x,v = state # Распаковываем текущее состояние
        dvdt = 1/m * (F0 * np.cos(omega * t) - c*v - k*x)
        dxdt = v
        return [dxdt, dvdt] # Возвращаем производные
    def oil_pendulum():
        # Параметры
        m, c, k = 1.0, 0.3, 1.0
        F0, omega = 1.0, 1.2
        
        # Начальное состояние
        state0 = [1.0, 0.0]
        
        # Сетка времени (теперь это просто точки, в которых мы хотим ПОЛУЧИТЬ ответ)
        t = np.linspace(0, 50, 10000)

        # 2. РЕШАЕМ в одну строку
        # Аргументы: функция, нач. состояние, время, дополнительные параметры (args)
        states = odeint(system_dynamics, state0, t, args=(m, c, k, F0, omega))        

        # states — это матрица, где колонки — это X, Y и Z
        x = states[:, 0]
        v = states[:, 1]

        x0=state0[0]
        v0=state0[1]

        #Аналитическое решение
        A = F0 * (k - m*omega**2) / ((k - m*omega**2)**2 + (c*omega)**2)
        B = F0 * (c*omega) / ((k - m*omega**2)**2 + (c*omega)**2)
        xch = A * np.cos(omega * t) + B * np.sin(omega * t)

        D = (c**2 - 4*k*m) / (m**2)

        if D>0:
            k1=-(c-np.sqrt(c**2-4*k*m))/(2*m)
            k2=-(c+np.sqrt(c**2-4*k*m))/(2*m)
            c1=x0-A-(v0-B*omega+k1*x0-A*k1)/(k2-k1)
            c2=(v0-B*omega+k1*x0-A*k1)/(k2-k1)
            xo=c1*np.exp(k1*t)+c2*np.exp(k2*t)
        elif D == 0:
            k12=-c/(2*m)
            c1=x0-A
            c2=v0-omega*B-k1*(x0-A)
            xo=np.exp(k12*t)*(c1+c2*t)
        elif D<0:
            alpha=-c/(2*m)
            betha=np.sqrt(4*k*m-c**2)/(2*m)
            c1=x0-A
            c2=(v0-omega*B-alpha*(x0-A))/betha
            xo=np.exp(alpha*t)*(c1*np.cos(betha*t)+c2*np.sin(betha*t))

        x_analytic=xo+xch

        # Визуализация
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.3)

        # 1. График координаты от времени
        ax1.plot(t, x, color='blue', label='Смещение (x)')
        ax1.plot(t, x_analytic, label='Аналитика (x)', alpha=0.7)
        ax1.set_title("Изменение координаты во времени")
        ax1.set_xlabel("Время (t)")
        ax1.set_ylabel("Координата (x)")
        ax1.grid(True)
        ax1.legend()

        # 2. Фазовый портрет (v от x)
        ax2.plot(x, v, color='magenta', lw=1)
        ax2.set_title("Фазовый портрет (скорость от координаты)")
        ax2.set_xlabel("Координата (x)")
        ax2.set_ylabel("Скорость (v)")
        ax2.grid(True)
        plt.show()

    oil_pendulum()