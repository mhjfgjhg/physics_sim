import numpy as np
import matplotlib.pyplot as plt

def run_simulation():
    def SIRS():
        N=1000
        beta_norm=0.4
        beta_karantin = beta_norm/3
        gamma=0.1
        ksi=0.1
        mu=0.005
        t_kar=20
        t_outkar=80
        dt_kar=5
        dt=0.01
        t_max=100
        S=[N-1]
        I=[1]
        R=[0]
        D=[0]
        time_steps=np.arange(0,t_max,dt)

        for _ in time_steps[:-1]:
            if _ <t_kar:
                beta = beta_norm
            elif _ >t_kar and _<t_kar+dt_kar:
                beta = beta_norm+((beta_karantin-beta_norm)/dt_kar) * (_-t_kar)
            elif _>t_kar+dt_kar and _<t_outkar:
                beta = beta_karantin
            elif _>=t_outkar and _<=t_outkar+dt_kar:
                beta = beta_karantin+((beta_norm-beta_karantin)/dt_kar) * (_-t_outkar)
            else:
                beta = beta_norm

            s_curr = S[-1]
            i_curr = I[-1]
            r_curr = R[-1]
            d_curr = D[-1]

            dS=-beta*s_curr*i_curr / N + ksi * r_curr
            dI=beta*s_curr*i_curr / N - gamma * i_curr - mu * i_curr
            dR=gamma*i_curr - ksi*r_curr
            dD=mu * i_curr

            S.append(s_curr + dS*dt)
            I.append(i_curr + dI*dt)
            R.append(r_curr + dR*dt)
            D.append(d_curr + dD*dt)

        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, S, label='Здоровье S', color='blue')
        plt.plot(time_steps, I, label='Больные I', color='red')
        plt.plot(time_steps, R, label='Выздоровевшие R', color='green')
        plt.plot(time_steps, D, label='Умершие D', color='black')
        plt.axvline(x=t_kar, color='g', linestyle='--', label='Карантин')
        plt.axvline(x=t_outkar, color='r', linestyle='--', label='Карантин')

        #plt.plot(S,I)
        plt.title("Модель пандемии SIR")
        #plt.xlabel("Здоровые")
        #plt.ylabel("Больные")
        plt.xlabel("Дни")
        plt.ylabel("Кол-во людей")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    SIRS()