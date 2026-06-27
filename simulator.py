import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# ============================================================
# Three-Body Simulator with Test Planet and Temperature Model
# ============================================================

st.set_page_config(page_title="Three-Body Simulator", layout="wide")

st.title("Three-Body Problem Simulator with Planet Temperature")
st.write(
    "This simulator shows three massive bodies interacting gravitationally. "
    "A small planet follows their gravitational field but does not affect them."
)

# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.header("Simulation Controls")

G = st.sidebar.slider("Gravity constant", 0.1, 5.0, 1.0, 0.1)
t_max = st.sidebar.slider("Simulation time", 5, 100, 30)
steps = st.sidebar.slider("Steps", 200, 5000, 1200)
trace_length = st.sidebar.slider("Trace length", 20, 1000, 200)
frame_step = st.sidebar.slider("Animation frame step", 1, 20, 2)
speed = st.sidebar.slider("Animation speed ms/frame", 10, 300, 40)

st.sidebar.header("Mass Controls")

m1 = st.sidebar.slider("Mass 1", 0.1, 5.0, 1.0)
m2 = st.sidebar.slider("Mass 2", 0.1, 5.0, 1.0)
m3 = st.sidebar.slider("Mass 3", 0.1, 5.0, 1.0)

# Planet is a test particle:
# it feels gravity from the three bodies but does not affect them.
masses = np.array([m1, m2, m3, 0.0])

st.sidebar.header("Planet Temperature Model")

base_temp = st.sidebar.slider("Base temperature", -100, 100, 20)
heating_strength = st.sidebar.slider("Heating strength", 1, 300, 80)
temp_scale = st.sidebar.slider("Distance softening", 0.1, 5.0, 1.0)

run = st.sidebar.button("Run Simulation")
reset = st.sidebar.button("Reset")

if reset:
    st.session_state.clear()
    st.rerun()

# ============================================================
# Visual style
# ============================================================

colors = [
    "#1f77b4",   # Mass 1 blue
    "#ff7f0e",   # Mass 2 orange
    "#2ca02c",   # Mass 3 green
    "#9467bd"    # Planet purple
]

names = ["Mass 1", "Mass 2", "Mass 3", "Planet"]


def mass_to_size(m):
    """
    Convert mass to marker size.
    Uses logarithmic scaling so large masses do not become too large visually.
    """
    return 8 + 12 * np.log2(m + 1)


def temperature_to_color(temp):
    """
    Convert planet temperature to a simple color.
    Blue = cold, purple = medium, red/yellow = hot.
    """
    if temp < 0:
        return "#1f77b4"  # cold blue
    elif temp < 50:
        return "#9467bd"  # mild purple
    elif temp < 120:
        return "#ff7f0e"  # warm orange
    else:
        return "#d62728"  # hot red


def calculate_planet_temperature(planet_pos, body_positions, body_masses):
    """
    Simple visual temperature model.

    Temperature increases when the planet is close to massive bodies.

    This is not a real astrophysical temperature model.
    It is designed for visualization only.
    """
    temp = base_temp

    for j in range(3):
        r = np.linalg.norm(planet_pos - body_positions[j])
        temp += heating_strength * body_masses[j] / (r**2 + temp_scale)

    return temp


# ============================================================
# Initial conditions
# ============================================================

# Initial positions: 3 massive bodies + 1 small planet
positions0 = np.array([
    [-1.0, 0.0],    # Mass 1
    [1.0, 0.0],     # Mass 2
    [0.0, 0.8],     # Mass 3
    [0.0, -1.5]     # Planet
])

# Initial velocities
velocities0 = np.array([
    [0.0, 0.35],    # Mass 1
    [0.0, -0.35],   # Mass 2
    [0.45, 0.0],    # Mass 3
    [0.8, 0.0]      # Planet
])

y0 = np.concatenate([
    positions0.flatten(),
    velocities0.flatten()
])


# ============================================================
# Physics engine
# ============================================================

def equations(t, y):
    positions = y[:8].reshape(4, 2)
    velocities = y[8:].reshape(4, 2)

    accelerations = np.zeros((4, 2))

    for i in range(4):
        for j in range(3):  # only Mass 1, 2, 3 generate gravity
            if i != j:
                r = positions[j] - positions[i]
                distance = np.linalg.norm(r) + 1e-6
                accelerations[i] += G * masses[j] * r / distance**3

    return np.concatenate([
        velocities.flatten(),
        accelerations.flatten()
    ])


def run_simulation():
    t_eval = np.linspace(0, t_max, steps)

    solution = solve_ivp(
        equations,
        (0, t_max),
        y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8
    )

    return solution.y[:8].reshape(4, 2, -1)


# ============================================================
# Run simulation
# ============================================================

if run or "positions" not in st.session_state:
    with st.spinner("Running simulation..."):
        st.session_state.positions = run_simulation()

positions = st.session_state.positions
n_frames = positions.shape[2]

# ============================================================
# Temperature calculation for planet
# ============================================================

planet_temperatures = []

for k in range(n_frames):
    planet_pos = positions[3, :, k]
    body_positions = positions[:3, :, k]
    temp = calculate_planet_temperature(
        planet_pos,
        body_positions,
        masses[:3]
    )
    planet_temperatures.append(temp)

planet_temperatures = np.array(planet_temperatures)

# ============================================================
# Plot bounds
# ============================================================

all_x = positions[:, 0, :].flatten()
all_y = positions[:, 1, :].flatten()

x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)

margin = 0.15 * max(x_max - x_min, y_max - y_min, 1)

# ============================================================
# Create animation frames
# ============================================================

frames = []

for frame in range(1, n_frames, frame_step):
    safe_frame = min(frame, n_frames - 1)
    start_trace = max(0, safe_frame - trace_length)

    frame_data = []

    # Traces
    for i in range(4):
        frame_data.append(
            go.Scatter(
                x=positions[i, 0, start_trace:safe_frame],
                y=positions[i, 1, start_trace:safe_frame],
                mode="lines",
                line=dict(
                    color=colors[i],
                    width=2 if i < 3 else 1
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Markers
    for i in range(4):
        if i == 3:
            marker_color = temperature_to_color(planet_temperatures[safe_frame])
            marker_size = 9
            hover_text = (
                f"Planet<br>"
                f"Temperature: {planet_temperatures[safe_frame]:.1f}<br>"
                f"x: {positions[i, 0, safe_frame]:.2f}<br>"
                f"y: {positions[i, 1, safe_frame]:.2f}"
            )
        else:
            marker_color = colors[i]
            marker_size = mass_to_size(masses[i])
            hover_text = (
                f"{names[i]}<br>"
                f"Mass: {masses[i]:.2f}<br>"
                f"x: {positions[i, 0, safe_frame]:.2f}<br>"
                f"y: {positions[i, 1, safe_frame]:.2f}"
            )

        frame_data.append(
            go.Scatter(
                x=[positions[i, 0, safe_frame]],
                y=[positions[i, 1, safe_frame]],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=marker_color
                ),
                name=names[i],
                legendgroup=names[i],
                showlegend=True,
                hovertext=hover_text,
                hoverinfo="text"
            )
        )

    frames.append(go.Frame(data=frame_data, name=str(safe_frame)))


# ============================================================
# Initial frame
# ============================================================

initial_data = []

# Empty traces at frame 0
for i in range(4):
    initial_data.append(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(
                color=colors[i],
                width=2 if i < 3 else 1
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

# Markers at frame 0
for i in range(4):
    if i == 3:
        marker_color = temperature_to_color(planet_temperatures[0])
        marker_size = 9
        hover_text = (
            f"Planet<br>"
            f"Temperature: {planet_temperatures[0]:.1f}<br>"
            f"x: {positions[i, 0, 0]:.2f}<br>"
            f"y: {positions[i, 1, 0]:.2f}"
        )
    else:
        marker_color = colors[i]
        marker_size = mass_to_size(masses[i])
        hover_text = (
            f"{names[i]}<br>"
            f"Mass: {masses[i]:.2f}<br>"
            f"x: {positions[i, 0, 0]:.2f}<br>"
            f"y: {positions[i, 1, 0]:.2f}"
        )

    initial_data.append(
        go.Scatter(
            x=[positions[i, 0, 0]],
            y=[positions[i, 1, 0]],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=marker_color
            ),
            name=names[i],
            legendgroup=names[i],
            showlegend=True,
            hovertext=hover_text,
            hoverinfo="text"
        )
    )


# ============================================================
# Build Plotly figure
# ============================================================

fig = go.Figure(
    data=initial_data,
    frames=frames
)

slider_steps = []

for frame in range(1, n_frames, frame_step):
    safe_frame = min(frame, n_frames - 1)
    slider_steps.append(
        dict(
            method="animate",
            args=[
                [str(safe_frame)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=str(safe_frame)
        )
    )

fig.update_layout(
    width=700,
    height=700,
    title="Three-Body Motion with Test Planet Temperature",
    xaxis=dict(
        range=[x_min - margin, x_max + margin],
        title="x"
    ),
    yaxis=dict(
        range=[y_min - margin, y_max + margin],
        title="y",
        scaleanchor="x",
        scaleratio=1
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=speed, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )
                    ]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate"
                        )
                    ]
                )
            ]
        )
    ],
    sliders=[
        dict(
            steps=slider_steps,
            currentvalue=dict(prefix="Frame: ")
        )
    ],
    legend=dict(
        title="Objects"
    )
)

st.plotly_chart(fig, use_container_width=False)

# ============================================================
# Temperature summary
# ============================================================

st.subheader("Planet Temperature Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Minimum temperature", f"{np.min(planet_temperatures):.1f}")
col2.metric("Average temperature", f"{np.mean(planet_temperatures):.1f}")
col3.metric("Maximum temperature", f"{np.max(planet_temperatures):.1f}")

temp_fig = go.Figure()

temp_fig.add_trace(
    go.Scatter(
        x=np.arange(n_frames),
        y=planet_temperatures,
        mode="lines",
        name="Planet temperature"
    )
)

temp_fig.update_layout(
    width=700,
    height=300,
    title="Planet Temperature over Time",
    xaxis_title="Frame",
    yaxis_title="Temperature"
)

st.plotly_chart(temp_fig, use_container_width=False)

st.caption(
    "Temperature is a simplified visual model. "
    "It increases when the planet gets closer to the three massive bodies."
)
