# -------- Pakkeimporter --------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- Parametre ----------
length_m: float = 1.0
wave_speed: float = 1.0
duration_s: float = 3.0

# ---------- Numeriske parametre ----------
num_points: int = 401

# ---------- For Fourier-metoden ----------
num_modes: int = 60

# ---------- Plot/animasjon ----------
fps: int = 60
line_width: float = 2.0

# ---------- Initialbetingelser ----------
initial_shape_type: str = "pluck"
initial_velocity_zero: bool = True

# ---------- Initialbetingelser for "pluck" ----------
pluck_position: float = 0.3
pluck_width: float = 0.15

# ---------- Initialbetingelser ----------
def initial_shape(x: np.ndarray) -> np.ndarray:
    if initial_shape_type == "sine1":
        return np.sin(np.pi * x / length_m)
    elif initial_shape_type == "pluck":
        center = pluck_position * length_m
        width = pluck_width * length_m
        return np.clip(1.0 - np.abs(x - center) / width, 0.0, 1.0)
    else:
        return np.exp(-((x - 0.5 * length_m) ** 2) / (0.05 * length_m) ** 2)

def initial_velocity(x: np.ndarray) -> np.ndarray:
    if initial_velocity_zero:
        return np.zeros_like(x)
    return 0.2 * np.sin(2 * np.pi * x / length_m)

# ---------- Diskret rom/tid ----------
x: np.ndarray = np.linspace(0.0, length_m, num_points)
dx: float = x[1] - x[0]           # beholdes for info, ikke brukt til dt
dt: float = 1.0 / fps             # tidssteg styres av ønsket animasjonsrate
num_steps: int = int(duration_s * fps) + 1
t: np.ndarray = np.arange(num_steps) * dt

# ---------- Metode: Analytisk Fourierserie ----------
def simulate_fourier() -> np.ndarray:
    f = initial_shape(x)
    g = initial_velocity(x)

    n = np.arange(1, num_modes + 1)[:, None]                         # (N,1)
    sin_basis = np.sin(np.pi * n * x[None, :] / length_m)            # (N, Nx)
    omega_n = wave_speed * np.pi * np.arange(1, num_modes + 1) / length_m  # (N,)

    B = (2.0 / length_m) * np.trapz(f[None, :] * sin_basis, x, axis=1)
    B_star = (2.0 / (length_m * omega_n)) * np.trapz(g[None, :] * sin_basis, x, axis=1)

    frames = np.zeros((num_steps, num_points))
    for k, tk in enumerate(t):
        coeff_t = B * np.cos(omega_n * tk) + B_star * np.sin(omega_n * tk)
        frames[k, :] = coeff_t @ sin_basis

    frames[:, 0] = 0.0
    frames[:, -1] = 0.0
    return frames

# ---------- Plot/animasjon ----------
def animate(frames: np.ndarray, title: str):
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, frames[0], lw=line_width)
    ax.set_xlim(0.0, length_m)
    y_pad = 0.1 + 0.1 * np.max(np.abs(frames[0]))
    ax.set_ylim(frames.min() - y_pad, frames.max() + y_pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("u(x,t) [arb. enhet]")
    ax.set_title(title)
    time_txt = ax.text(0.02, 0.92, f"t = 0.000 s", transform=ax.transAxes)

    def init():
        line.set_ydata(frames[0])
        time_txt.set_text(f"t = {0.0:.3f} s")
        return line, time_txt

    def update(i):
        line.set_ydata(frames[i])
        time_txt.set_text(f"t = {i*dt:.3f} s")
        return line, time_txt

    anim = FuncAnimation(
        fig,
        update,
        frames=range(0, len(frames), 1),
        init_func=init,
        interval=1000 / fps,    # ekte fps
        blit=True,
    )
    plt.show()
    return anim

# ---------- Kjør ----------
if __name__ == "__main__":
    U = simulate_fourier()
    anim = animate(U, f"Bølgeligning - Fourierserie (N={num_modes})")
