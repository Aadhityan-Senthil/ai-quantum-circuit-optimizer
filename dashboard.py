"""
============================================================
STREAMLIT DASHBOARD — Visualize the AI optimizing circuits
============================================================

Run this with:  streamlit run dashboard.py

This lets you:
  - Generate random quantum circuits
  - Watch the trained AI optimize them in real-time
  - See the before/after circuit comparison
  - See which optimization rules the agent chose
============================================================
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

from stable_baselines3 import DQN

from quantum_env import (
    QuantumCircuitOptEnv,
    generate_random_circuit,
    compute_unitary,
    matrices_are_equivalent,
    SINGLE_GATES,
)

# ---------------------------------------------------------------------------
# GATE COLORS — for the circuit visualization
# ---------------------------------------------------------------------------
GATE_COLORS = {
    'H': '#6c63ff',   # purple
    'X': '#f59e0b',   # amber
    'Z': '#10b981',   # green
    'I': '#6b7280',   # gray
    'S': '#ec4899',   # pink
    'T': '#3b82f6',   # blue
}

ACTION_NAMES = {
    0: "Cancel H·H / X·X / Z·Z",
    1: "Merge S·S → Z",
    2: "Merge T·T → S",
    3: "Substitute H·X·H → Z",
    4: "Remove Identity (I)",
    5: "Pass (no action)",
}


def draw_circuit(circuit_names, title, ax, highlight_indices=None):
    """
    Draw a quantum circuit as a horizontal sequence of colored gate boxes.
    This is a simplified 1-qubit visualization.
    """
    ax.set_xlim(-0.5, max(len(circuit_names) - 0.5, 1))
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    if len(circuit_names) == 0:
        ax.text(0.5, 0.5, "Empty circuit (fully optimized!)",
                ha='center', va='center', fontsize=11, color='#10b981',
                transform=ax.transAxes)
        return

    # Draw the qubit wire
    ax.plot([-0.3, len(circuit_names) - 0.7], [0.5, 0.5],
            color='#444', linewidth=2, zorder=1)

    # Draw each gate as a box
    for i, name in enumerate(circuit_names):
        color = GATE_COLORS.get(name, '#888')
        alpha = 1.0

        # Highlight if this gate is part of an optimization target
        edge_color = '#fff'
        edge_width = 1.5
        if highlight_indices and i in highlight_indices:
            edge_color = '#f59e0b'
            edge_width = 3

        box = mpatches.FancyBboxPatch(
            (i - 0.3, 0.15), 0.6, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=edge_color,
            linewidth=edge_width, alpha=alpha, zorder=2
        )
        ax.add_patch(box)
        ax.text(i, 0.5, name, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=3)

    # Depth label
    ax.text(len(circuit_names) - 0.5, -0.3, f"Depth: {len(circuit_names)} gates",
            ha='right', va='top', fontsize=9, color='#888')


def find_optimization_targets(circuit_names):
    """Find indices of gates that can be optimized (for highlighting)."""
    targets = []
    self_inverse = {'H', 'X', 'Z'}

    for i in range(len(circuit_names) - 1):
        if circuit_names[i] in self_inverse and circuit_names[i] == circuit_names[i + 1]:
            targets.extend([i, i + 1])
            break
        if circuit_names[i] == 'S' and circuit_names[i + 1] == 'S':
            targets.extend([i, i + 1])
            break
        if circuit_names[i] == 'T' and circuit_names[i + 1] == 'T':
            targets.extend([i, i + 1])
            break

    for i in range(len(circuit_names) - 2):
        if (circuit_names[i] == 'H' and
                circuit_names[i + 1] == 'X' and
                circuit_names[i + 2] == 'H'):
            targets.extend([i, i + 1, i + 2])
            break

    for i in range(len(circuit_names)):
        if circuit_names[i] == 'I':
            targets.append(i)
            break

    return targets


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI Quantum Circuit Optimizer",
        page_icon="⚛️",
        layout="wide"
    )

    # --- Styling ---
    st.markdown("""
    <style>
        .main { background-color: #0f0f1a; color: #e2e2f0; }
        .stButton > button {
            background: linear-gradient(135deg, #6c63ff, #a78bfa);
            color: white; border: none; border-radius: 8px;
            padding: 10px 24px; font-size: 14px; font-weight: 600;
            cursor: pointer;
        }
        .stButton > button:hover { opacity: 0.85; }
        .stSidebar { background-color: #161625 !important; }
        h1 { color: #a78bfa !important; }
        .stMetric { background: #161625; border-radius: 12px; padding: 12px; }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.title("⚛️ AI Quantum Circuit Optimizer")
    st.markdown("A reinforcement learning agent that learns to simplify quantum circuits — reducing gate count while preserving the quantum operation.")

    # --- Sidebar controls ---
    st.sidebar.header("⚙️ Settings")
    circuit_depth = st.sidebar.slider("Circuit Depth (gates)", min_value=6, max_value=20, value=12)
    seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)

    model_loaded = False
    model = None

    st.sidebar.divider()
    st.sidebar.header("🤖 Model")
    model_file = st.sidebar.file_uploader(
        "Upload trained model (.zip)",
        type=['zip'],
        help="Run train.py first to generate dqn_circuit_opt.zip"
    )

    if model_file:
        # Save uploaded file temporarily and load
        with open('/tmp/model.zip', 'wb') as f:
            f.write(model_file.read())
        try:
            model = DQN.load('/tmp/model.zip')
            model_loaded = True
            st.sidebar.success("✓ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")

    # --- Legend ---
    st.sidebar.divider()
    st.sidebar.header("🎨 Gate Legend")
    for name, color in GATE_COLORS.items():
        st.sidebar.markdown(
            f'<span style="display:inline-block; width:14px; height:14px; '
            f'background:{color}; border-radius:3px; margin-right:6px;"></span>{name}',
            unsafe_allow_html=True
        )

    # --- Main content ---
    col1, col2 = st.columns(2)

    # Generate button
    if col1.button("🎲 Generate New Circuit"):
        circuit, names = generate_random_circuit(circuit_depth, seed=seed)
        st.session_state['circuit'] = circuit
        st.session_state['circuit_names'] = names
        st.session_state['original_names'] = list(names)
        st.session_state['action_log'] = []
        st.session_state['step'] = 0

    # Run AI button
    run_ai = col2.button(
        "🤖 Run AI Optimizer",
        disabled=not model_loaded or 'circuit' not in st.session_state,
        help="Upload a trained model first, then generate a circuit"
    )

    # --- Display current circuit ---
    if 'circuit_names' in st.session_state:
        original = st.session_state.get('original_names', [])
        current = st.session_state['circuit_names']

        # Metrics row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Original Depth", f"{len(original)} gates")
        mcol2.metric("Current Depth", f"{len(current)} gates")
        reduction = len(original) - len(current)
        mcol3.metric("Gates Removed", f"{reduction}", delta=f"-{reduction}" if reduction > 0 else "0")
        ratio = len(current) / max(len(original), 1)
        mcol4.metric("Compression", f"{ratio:.0%}")

        # Circuit visualization
        fig, axes = plt.subplots(2, 1, figsize=(max(12, len(original) * 0.7), 4))
        fig.patch.set_facecolor('#0f0f1a')

        # Before
        draw_circuit(original, "📌 Original Circuit", axes[0])
        # After (highlight optimization targets)
        targets = find_optimization_targets(current)
        draw_circuit(current, "⚡ Current / Optimized Circuit", axes[1], highlight_indices=targets)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # --- Run AI optimization step-by-step ---
        if run_ai and model_loaded:
            env = QuantumCircuitOptEnv(circuit_depth=circuit_depth, max_steps=50)
            # Manually set the environment to match current original circuit
            env.circuit = list(st.session_state.get('_orig_circuit', st.session_state['circuit']))
            env.circuit_names = list(st.session_state.get('original_names', []))
            env.target_unitary = compute_unitary(env.circuit)
            env.original_depth = len(env.circuit)
            env.steps_taken = 0

            obs = env._encode_observation()
            action_log = []

            st.markdown("### 📋 AI Optimization Steps")
            step = 0
            done = False

            while not done and step < 50:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                old_names = list(env.circuit_names)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                new_names = list(env.circuit_names)

                if old_names != new_names:
                    step += 1
                    status = f"Step {step}: **{ACTION_NAMES[action]}**"
                    status += f"  |  {len(old_names)} → {len(new_names)} gates  |  Reward: +{reward:.1f}"
                    st.markdown(f"  ✅ {status}")
                    action_log.append(ACTION_NAMES[action])
                elif action == 5:
                    break  # Agent chose to stop

            # Update session state with final result
            st.session_state['circuit_names'] = list(env.circuit_names)
            st.session_state['action_log'] = action_log

            if not action_log:
                st.info("The agent found no further optimizations possible.")

            # Force rerender with final circuit
            st.divider()
            st.markdown(f"**Final result:** {len(original)} → {len(env.circuit_names)} gates "
                        f"({len(original) - len(env.circuit_names)} removed)")

    else:
        st.info("👆 Click **Generate New Circuit** to start.")

    # --- How it works section ---
    with st.expander("📖 How does the AI optimizer work?"):
        st.markdown("""
        **The Reinforcement Learning Agent** learns to optimize quantum circuits through trial and error:

        1. **Observation:** The agent sees the circuit encoded as a numerical vector (one-hot encoding of gate types)
        2. **Action:** It chooses from 6 possible optimization rules to apply
        3. **Reward:** It gets positive reward for removing gates while keeping the circuit equivalent, negative reward for invalid moves
        4. **Learning:** Over tens of thousands of training episodes, the agent learns which patterns to look for and which rules to apply

        **Optimization Rules the Agent Can Learn:**
        - Cancel self-inverse gate pairs (H·H = I, X·X = I, Z·Z = I)
        - Merge sequential gates (S·S = Z, T·T = S)
        - Apply conjugation identities (H·X·H = Z)
        - Remove identity gates

        **Why this matters for real quantum computing:**
        Every gate on a real quantum processor adds noise. Fewer gates = less noise = better results.
        This is one of the most active research areas in quantum computing right now.
        """)


if __name__ == "__main__":
    main()
