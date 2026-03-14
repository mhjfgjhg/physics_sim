import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

try:
    from models import Lorenz_Euler, Lorenz, Parashute, Predator_Prey, SIRS_D, Oil_pendulum, Landing  # добавь свои названия
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедись, что в папке models есть файл init.py")
    sys.exit(1)

def main():
    print("Выберите модель для запуска:")
    print("0. Выход")
    print("1. Аттрактор Лоренца (через Эйлера) (3D)")
    print("2. Аттрактор Лоренца (3D)")
    print("3. Парашютист (2D)")
    print("4. Модель Волк-Заяц (2D)")
    print("5. Модель SIRS-SIRD")
    print("6. Маятник в масле")
    print("7. Десант")

    choice = input("\nВведите номер: ")

    if choice == '0':
        print("Пока!")
    elif choice == '1':
        Lorenz_Euler.run_simulation() # Убедись, что в файле lorenz.py расчет обернут в функцию
    elif choice == '2':
        Lorenz.run_simulation()
    elif choice == '3':
        Parashute.run_simulation()
    elif choice == '4':
        Predator_Prey.run_simulation()
    elif choice == '5':
        SIRS_D.run_simulation()
    elif choice == '6':
        Oil_pendulum.run_simulation()
    elif choice == '7':
        Landing.run_simulation()
    else:
        print("Неверный ввод.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма принудительно остановлена.")
        sys.exit(0)