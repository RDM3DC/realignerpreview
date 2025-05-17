import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Arc path setup
r = 1.0
theta = np.linspace(0, np.pi/2, 100)
x_arc = r * np.cos(np.pi/2 - theta)
y_arc = r * np.sin(np.pi/2 - theta)

# Chord (G1) approximation
num_segments = 5
theta_chord = np.linspace(0, np.pi/2, num_segments + 1)
x_chord = r * np.cos(np.pi/2 - theta_chord)
y_chord = r * np.sin(np.pi/2 - theta_chord)

# Matplotlib setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.set_title("Tracing the Curve (Slow + Controlled)")
line_arc, = ax.plot([], [], 'b-', label='Adaptive π (Arc)')
line_chord, = ax.plot([], [], 'r--', label='Chord Segments')
point, = ax.plot([], [], 'ko', markersize=5)
ax.legend()

def init():
    line_arc.set_data([], [])
    line_chord.set_data([], [])
    point.set_data([], [])
    return line_arc, line_chord, point

def update(frame):
    point.set_data([x_arc[frame]], [y_arc[frame]])
    line_arc.set_data(x_arc[:frame+1], y_arc[:frame+1])
    if frame == len(x_arc) - 1:
        line_chord.set_data(x_chord, y_chord)
    return line_arc, line_chord, point

ani = animation.FuncAnimation(
    fig, update, frames=len(x_arc),
    init_func=init, blit=True, interval=120, repeat=False
)

# Save as .gif
ani.save("adaptive_pi_arc_vs_chord_slow.gif", writer='pillow', fps=10)

# Or display live:
plt.show()
