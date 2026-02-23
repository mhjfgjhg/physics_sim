# Physics Mathematical Simulation

A set of training mathematical models for physical problems. Developed for the purpose of in-depth study of differential equations and mathematical modeling using the example of physical problems.

## Phisics Models

### Lorenz Attractor
A 3D simulation Lorenz sysem "Butterfly Effect"
On the *Euler method* and *Runge-Kutti*

The system is composed of three ordinary differential equations:

$$
\begin{aligned}
\frac{dx}{dt} &= \sigma(y - x) \\
\frac{dy}{dt} &= x(\rho - z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{aligned}
$$

<img width="941" height="985" alt="lorenz" src="https://github.com/user-attachments/assets/3f2c034e-5a6a-4193-8ac7-4376087ae0b0" />

### Predator-Prey Dynamics
Classic simple ecological model
On the *Runge-Kutti*
<img width="941" height="985" alt="Predator-Prey" src="https://github.com/user-attachments/assets/617af8d5-eb7a-4b83-b1e4-b11dda0d30cf" />

### SIRS/SIRD Epidemic Models
Epidemic models whis immunity, quarantine, death
On the *Euler method*

$$
\begin{aligned}
\frac{dS}{dt} &= -\frac{\beta S I}{N} + \xi R \\
\frac{dI}{dt} &= \frac{\beta S I}{N} - \gamma I - \mu I \\
\frac{dR}{dt} &= \gamma I - \xi R \\
\frac{dD}{dt} &= \mu I
\end{aligned}
$$

<img width="941" height="985" alt="SIRS-D" src="https://github.com/user-attachments/assets/47c4f0b4-bc19-4bf0-9edd-b52f86fc24d7" />

### Parashute Jump Simulation
A base physical model of free fall with gynamic drag coef
On the *Euler method*
<img width="941" height="985" alt="parashute" src="https://github.com/user-attachments/assets/c8d94027-86c5-474d-a608-6537fac7216c" />

## Tech Stack
- Language: Python 3.12+
- Environment: Managed via uv (Fast Python package manager)
- Libraries: NumPy, SciPy, Matplotlib
- OS: Arch Linux
- Tested on: Arch, Debian

## Quick Start
If you have uv installed, simply run:
```bash
git clone https://github.com/mhjfgjhg/physics_sim.git
cd physics_sim
puthon3 main.py
# Or through uv
#uv sync
#uv run main.py
