# Bølgelignigen simulasjon i 1D
# -------- Pakkeimporter --------
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- Parametre ----------
length_m: float = 1.0          # L: strengens lengde [m]
wave_speed: float = 1.0        # c: bølgefart [m/s]
duration_s: float = 3.0        # total simuleringstid [s]
method: str = "finite_difference"  # "finite_difference" eller "fourier"

# ---------- Numeriske parametre ----------
num_points: int = 401          # antall rompunkter (inkl. endepunkter)
cfl: float = 0.95              # CFL = c*dt/dx (≤ ~1 for stabilitet ved FD)

# ---------- For Fourier-metoden ----------
num_modes: int = 60            # antall normalmodi i seriesummen

# ---------- Plot/animasjon ----------
fps: int = 60                  # frames per second
line_width: float = 2.0        # tykkelse på kurven

# ---------- Initialbetingelser ----------
initial_shape_type: str = "pluck"
initial_velocity_zero: bool = True   # True = slipp streng fra ro (g(x)=0)

# ---------- Initialbetingelser for "pluck" ----------
pluck_position: float = 0.3    # som andel av L (0–1)
pluck_width: float = 0.15      # som andel av L (0–1)


# ---------- Initialbetingelser ----------
def initial_shape(x: np.ndarray) -> np.ndarray:
    """Startform f(x) for strengen ved t=0."""
    if initial_shape_type == "sine1":
        return np.sin(np.pi * x / length_m)  # ren første modus
    elif initial_shape_type == "pluck":
        # trekant-bump rundt pluck_position*L med bredde pluck_width*L
        center = pluck_position * length_m
        width = pluck_width * length_m
        return np.clip(1.0 - np.abs(x - center) / width, 0.0, 1.0)
    else:
        # enkel egendefinert variant (kan endres fritt)
        return np.exp(-((x - 0.5 * length_m) ** 2) / (0.05 * length_m) ** 2)

def initial_velocity(x: np.ndarray) -> np.ndarray:
    """Starthastighet g(x) = u_t(x,0)."""
    if initial_velocity_zero:
        return np.zeros_like(x)
    # eksempel: liten “dytt”
    return 0.2 * np.sin(2 * np.pi * x / length_m)


# ---------- Diskret rom/tid ----------
x: np.ndarray = np.linspace(0.0, length_m, num_points)
dx: float = x[1] - x[0]
dt: float = cfl * dx / wave_speed
num_steps: int = int(np.ceil(duration_s / dt))
t: np.ndarray = np.arange(num_steps) * dt


# ---------- Metode A: Finitt differanse (leapfrog) ----------
def simulate_fd() -> np.ndarray:
    """Simulerer med 2. ordens finitte differanser i tid/rom (leapfrog)."""
    lam2 = (wave_speed * dt / dx) ** 2

    u_prev = initial_shape(x)          # u(x, 0)
    v0 = initial_velocity(x)           # g(x) = u_t(x, 0)
    u_curr = np.empty_like(u_prev)
    u_next = np.empty_like(u_prev)

    # Faste ender
    u_prev[0] = 0.0
    u_prev[-1] = 0.0

    # Første steg (Taylor-kick)
    u_curr[1:-1] = (u_prev[1:-1] + dt * v0[1:-1]
                    + 0.5 * lam2 * (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]))
    u_curr[0] = 0.0
    u_curr[-1] = 0.0

    frames = np.empty((num_steps, num_points))
    frames[0, :] = u_prev
    frames[1, :] = u_curr

    # Leapfrog
    for n in range(1, num_steps - 1):
        u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1]
                        + lam2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]))
        u_next[0] = 0.0
        u_next[-1] = 0.0

        frames[n + 1, :] = u_next
        # “roter” pekerne uten ekstra kopiering
        u_prev, u_curr, u_next = u_curr, u_next, u_prev

    return frames


# ---------- Metode B: Analytisk Fourierserie ----------
def simulate_fourier() -> np.ndarray:
    """Simulerer via separasjon av variable og Fourier-sinusserier."""
    f = initial_shape(x)
    g = initial_velocity(x)

    n = np.arange(1, num_modes + 1)[:, None]        # (N,1)
    sin_basis = np.sin(np.pi * n * x[None, :] / length_m)  # (N, Nx)
    omega_n = wave_speed * np.pi * np.arange(1, num_modes + 1) / length_m  # (N,)

    # Koefisienter (trapesintegrasjon)
    B = (2.0 / length_m) * np.trapz(f[None, :] * sin_basis, x, axis=1)
    B_star = (2.0 / (length_m * omega_n)) * np.trapz(g[None, :] * sin_basis, x, axis=1)

    frames = np.zeros((num_steps, num_points))
    for k, tk in enumerate(t):
        coeff_t = B * np.cos(omega_n * tk) + B_star * np.sin(omega_n * tk)
        frames[k, :] = coeff_t @ sin_basis  # (N,) @ (N, Nx) -> (Nx,)

    frames[:, 0] = 0.0
    frames[:, -1] = 0.0
    return frames


# ---------- Plot/animasjon ----------
def animate(frames: np.ndarray, title: str):
    """Animerer løsning u(x,t)."""
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, frames[0], lw=line_width)
    ax.set_xlim(0.0, length_m)
    y_pad = 0.1 + 0.1 * np.max(np.abs(frames[0]))
    ax.set_ylim(frames.min() - y_pad, frames.max() + y_pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("u(x,t) [arb. enhet]")
    ax.set_title(title)
    time_txt = ax.text(0.02, 0.92, f"t = 0.000 s", transform=ax.transAxes)

    step_stride = max(1, int(len(frames) / (duration_s * fps)))

    def init():
        line.set_ydata(frames[0])
        time_txt.set_text(f"t = {0.0:.3f} s")
        return line, time_txt

    def update(i):
        line.set_ydata(frames[i])
        time_txt.set_text(f"t = {i*dt:.3f} s")
        return line, time_txt

    # Viktig: lagre animasjonen i en variabel
    anim = FuncAnimation(
        fig,
        update,
        frames=range(0, len(frames), step_stride),
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    plt.show()
    return anim

 
# ---------- Kjør ----------
if __name__ == "__main__":
    if method == "finite_difference":
        U = simulate_fd()
        anim = animate(U, "Bølgeligning - Finitt differanse")
    elif method == "fourier":
        U = simulate_fourier()
        anim = animate(U, f"Bølgeligning - Fourierserie (N={num_modes})")
    else:
        raise ValueError("Ukjent metode. Bruk 'finite_difference' eller 'fourier'.")