import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Three-Body Simulator", layout="wide")

st.title("Three-Body Problem Simulator with Planet")

st.sidebar.header("Controls")

G = st.sidebar.slider("Gravity constant", 0.1, 5.0, 1.0, 0.1)
t_max = st.sidebar.slider("Simulation time", 5, 100, 30)
steps = st.sidebar.slider("Steps", 200, 3000, 800)
trace_length = st.sidebar.slider("Trace length", 20, 600, 200)
frame_step = st.sidebar.slider("Animation frame step", 1, 20, 4)
speed = st.sidebar.slider("Animation speed ms/frame", 10, 200, 40)

preset = st.sidebar.selectbox(
    "Preset",
    [
        "Random",
        "Stable Planet",
        "Random Binary Star",
        "Random Figure Eight",
        "Random Chaotic"
    ]
)

run = st.sidebar.button("Run Simulation")
reset = st.sidebar.button("Reset")
new_random = st.sidebar.button("🎲 New Random System")

if reset or new_random:
    st.session_state.clear()
    st.rerun()

colors = [
    "#1f77b4",   # Mass 1
    "#ff7f0e",   # Mass 2
    "#2ca02c",   # Mass 3
    "#9467bd"    # Planet
]

names = ["Mass 1", "Mass 2", "Mass 3", "Planet"]


def mass_to_size(m):
    """
    Convert mass to marker size.
    Square-root scaling keeps large masses visible but not too large.
    """
    return 12 + 6 * np.sqrt(m)


def get_initial_conditions(preset_name):
    """
    Return initial positions, velocities, and masses.

    There are 4 objects:
    - Object 0: Mass 1
    - Object 1: Mass 2
    - Object 2: Mass 3
    - Object 3: Planet

    The planet is a test particle with mass = 0.
    It feels gravity from the three masses, but it does not affect them.
    """

    if preset_name == "Stable Planet":
        positions = np.array([
            [0.0, 0.0],
            [2.5, 0.0],
            [-2.5, 0.0],
            [0.0, 5.0]
        ])

        velocities = np.array([
            [0.0, 0.0],
            [0.0, 1.40],
            [0.0, -1.40],
            [-0.95, 0.0]
        ])

        masses = np.array([5.0, 0.3, 0.3, 0.0])

    elif preset_name == "Random Binary Star":
        positions = np.array([
            [-1.5, 0.0],
            [1.5, 0.0],
            np.random.uniform(-4, 4, 2),
            np.random.uniform(-8, 8, 2)
        ])

        velocities = np.array([
            [0.0, np.random.uniform(0.8, 1.2)],
            [0.0, -np.random.uniform(0.8, 1.2)],
            np.random.uniform(-0.5, 0.5, 2),
            np.random.uniform(-1.0, 1.0, 2)
        ])

        masses = np.array([
            np.random.uniform(3.0, 5.0),
            np.random.uniform(3.0, 5.0),
            np.random.uniform(0.2, 1.0),
            0.0
        ])

    elif preset_name == "Random Figure Eight":
        positions = np.array([
            [-0.97000436, 0.24308753],
            [0.97000436, -0.24308753],
            [0.0, 0.0],
            np.random.uniform(-6, 6, 2)
        ])

        velocities = np.array([
            [0.466203685, 0.43236573],
            [0.466203685, 0.43236573],
            [-0.93240737, -0.86473146],
            np.random.uniform(-1.0, 1.0, 2)
        ])

        masses = np.array([
            np.random.uniform(0.8, 1.2),
            np.random.uniform(0.8, 1.2),
            np.random.uniform(0.8, 1.2),
            0.0
        ])

    elif preset_name == "Random Chaotic":
        positions = np.random.uniform(-3, 3, (4, 2))
        velocities = np.random.uniform(-1.2, 1.2, (4, 2))

        masses = np.array([
            np.random.uniform(0.5, 5.0),
            np.random.uniform(0.5, 5.0),
            np.random.uniform(0.5, 5.0),
            0.0
        ])

    else:
        positions = np.random.uniform(-2, 2, (4, 2))
        velocities = np.random.uniform(-0.8, 0.8, (4, 2))

        masses = np.array([
            np.random.uniform(0.5, 5.0),
            np.random.uniform(0.5, 5.0),
            np.random.uniform(0.5, 5.0),
            0.0
        ])

    return positions, velocities, masses


positions0, velocities0, masses = get_initial_conditions(preset)

y0 = np.concatenate([
    positions0.flatten(),
    velocities0.flatten()
])


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


def calculate_planet_temperature(positions):
    """
    Simple planet temperature model.

    This is not a real astrophysics model.
    It is a visual/educational model:
    - closer to massive bodies = hotter
    - farther away = colder
    - thermal inertia smooths fast changes
    """

    planet_positions = positions[3]       # shape: (2, frames)
    body_positions = positions[:3]        # shape: (3, 2, frames)

    equilibrium_temps = []

    for k in range(positions.shape[2]):
        planet = planet_positions[:, k]
        heating = 0.0

        for j in range(3):
            body = body_positions[j, :, k]
            distance = np.linalg.norm(planet - body) + 1e-6
            heating += masses[j] / distance**2

        # Scale heating into a visible temperature range
        temp = 180 + 120 * np.sqrt(heating)
        equilibrium_temps.append(temp)

    equilibrium_temps = np.array(equilibrium_temps)

    # Apply simple thermal inertia
    smooth_temps = np.zeros_like(equilibrium_temps)
    smooth_temps[0] = equilibrium_temps[0]

    inertia = 0.03
    for k in range(1, len(equilibrium_temps)):
        smooth_temps[k] = smooth_temps[k - 1] + inertia * (
            equilibrium_temps[k] - smooth_temps[k - 1]
        )

    return smooth_temps


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


if run or "positions" not in st.session_state:
    with st.spinner("Running simulation..."):
        st.session_state.positions = run_simulation()
        st.session_state.temperatures = calculate_planet_temperature(
            st.session_state.positions
        )

positions = st.session_state.positions
temperatures = st.session_state.temperatures

n_frames = positions.shape[2]

all_x = positions[:, 0, :].flatten()
all_y = positions[:, 1, :].flatten()

x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)

margin = 0.15 * max(x_max - x_min, y_max - y_min, 1)

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
                line=dict(color=colors[i], width=2 if i < 3 else 1),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Markers
    for i in range(4):
        frame_data.append(
            go.Scatter(
                x=[positions[i, 0, safe_frame]],
                y=[positions[i, 1, safe_frame]],
                mode="markers",
                marker=dict(
                    size=mass_to_size(masses[i]) if i < 3 else 8,
                    color=colors[i]
                ),
                name=names[i],
                legendgroup=names[i],
                showlegend=True
            )
        )

    frames.append(go.Frame(data=frame_data, name=str(safe_frame)))


initial_data = []

# Empty traces at the first frame
for i in range(4):
    initial_data.append(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=colors[i], width=2 if i < 3 else 1),
            hoverinfo="skip",
            showlegend=False
        )
    )

# Initial markers
for i in range(4):
    initial_data.append(
        go.Scatter(
            x=[positions[i, 0, 0]],
            y=[positions[i, 1, 0]],
            mode="markers",
            marker=dict(
                size=mass_to_size(masses[i]) if i < 3 else 8,
                color=colors[i]
            ),
            name=names[i],
            legendgroup=names[i],
            showlegend=True
        )
    )

fig = go.Figure(
    data=initial_data,
    frames=frames
)

fig.update_layout(
    width=650,
    height=650,
    title="Three-Body Motion with a Test Planet",
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
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(frame)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0)
                        )
                    ],
                    label=str(frame)
                )
                for frame in range(1, n_frames, frame_step)
            ],
            currentvalue=dict(prefix="Frame: ")
        )
    ]
)

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=False)

with col2:
    st.subheader("Planet Temperature")

    temp_now = temperatures[0]
    temp_min = float(np.min(temperatures))
    temp_max = float(np.max(temperatures))
    temp_mean = float(np.mean(temperatures))

    st.metric("Initial temperature", f"{temp_now:.1f} K")
    st.write(f"Minimum: **{temp_min:.1f} K**")
    st.write(f"Average: **{temp_mean:.1f} K**")
    st.write(f"Maximum: **{temp_max:.1f} K**")

    if 240 <= temp_mean <= 320 and (temp_max - temp_min) < 120:
        st.success("Relatively stable / potentially habitable temperature range")
    elif temp_max - temp_min > 300:
        st.warning("Large temperature swings")
    else:
        st.info("Moderate or cold temperature range")

    temp_fig = go.Figure()
    temp_fig.add_trace(
        go.Scatter(
            x=np.arange(n_frames),
            y=temperatures,
            mode="lines",
            name="Planet temperature"
        )
    )
    temp_fig.update_layout(
        width=420,
        height=300,
        title="Planet Temperature Over Time",
        xaxis_title="Frame",
        yaxis_title="Temperature (K)"
    )
    st.plotly_chart(temp_fig, use_container_width=True)

st.caption(
    "The planet follows the gravity of the three massive bodies but does not affect them. "
    "Temperature is a simplified visual model based on distance from the massive bodies."
)