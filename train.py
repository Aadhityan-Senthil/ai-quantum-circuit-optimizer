"""
============================================================
RL AGENT TRAINING — DQN learns to optimize quantum circuits
============================================================

WHAT IS DQN?
  Deep Q-Network. The agent learns a "Q-value" for every
  (state, action) pair — basically, it learns to predict
  "how much total reward will I get if I take action A
  in state S?"

  The agent then always picks the action with the highest
  predicted reward. Over time, it gets better at predicting,
  so it gets better at optimizing circuits.

WHY DQN (not PPO)?
  - Simpler to understand and debug for a first project
  - Works well for discrete action spaces (we have 6 actions)
  - Stable Baselines3 has a clean DQN implementation
  - PPO is better for continuous actions — we'll use it later

TRAINING LOOP:
  1. Agent observes the circuit (as a number vector)
  2. Agent picks an action (which optimization rule to try)
  3. Environment applies it, returns reward
  4. Agent learns from the reward
  5. Repeat thousands of times
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for rendering

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from quantum_env import QuantumCircuitOptEnv


# ---------------------------------------------------------------------------
# CUSTOM CALLBACK — Tracks our quantum-specific metrics during training
# ---------------------------------------------------------------------------
class QuantumMetricsCallback(BaseCallback):
    """
    Stable Baselines3 callbacks let you hook into the training loop.
    We use this to log quantum-specific metrics (compression ratio, etc.)
    that the default SB3 logger doesn't know about.

    Also records every action taken so we can later plot how the agent's
    strategy evolves over training (Improvement 4).
    """

    def __init__(self, log_interval=100):
        super().__init__()
        self.log_interval = log_interval
        self.compression_ratios = []
        self.episode_rewards = []
        self.gates_removed_history = []
        self.depth_ratios = []                # (Improvement 2) depth tracking
        self.action_log = []                  # (Improvement 4) every action taken
        self.action_log_by_phase = {          # actions binned into training thirds
            'early':  [], 'mid': [], 'late': []
        }
        self._total_steps_estimate = 80_000   # filled in before .learn()

    def _on_step(self):
        # --- Record the action the agent just took (Improvement 4) ---
        action = int(self.locals['actions'][0]) if 'actions' in self.locals else None
        if action is not None:
            self.action_log.append(action)
            # Bin into thirds of training for phase-based analysis
            progress = self.n_calls / max(self._total_steps_estimate, 1)
            phase = 'early' if progress < 0.33 else ('mid' if progress < 0.66 else 'late')
            self.action_log_by_phase[phase].append(action)

        # Pull the info dict from the most recent step
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'compression_ratio' in info:
                    self.compression_ratios.append(info['compression_ratio'])
                if 'depth_ratio' in info:                          # Improvement 2
                    self.depth_ratios.append(info['depth_ratio'])
                if 'circuit_length' in info and 'original_length' in info:
                    removed = info['original_length'] - info['circuit_length']
                    self.gates_removed_history.append(removed)

        # Log to console every N steps
        if self.n_calls % (self.log_interval * 100) == 0:
            if self.compression_ratios:
                avg_ratio  = np.mean(self.compression_ratios[-200:])
                avg_depth  = np.mean(self.depth_ratios[-200:]) if self.depth_ratios else 0
                avg_removed = np.mean(self.gates_removed_history[-200:])
                print(f"  Step {self.n_calls:>7d} | "
                      f"Compression: {avg_ratio:.2%} | "
                      f"Depth ratio: {avg_depth:.2%} | "
                      f"Gates removed: {avg_removed:.1f}")
        return True


# ---------------------------------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------------------------------

def train_agent(total_timesteps=80_000, circuit_depth=12, model_path="dqn_circuit_opt"):
    """
    Train the DQN agent to optimize quantum circuits.

    Args:
        total_timesteps: How many environment steps to train for.
                         More = better agent, but slower.
        circuit_depth:   How deep the random circuits are.
                         Start small (8-12), increase as agent improves.
        model_path:      Where to save the trained model.
    """
    print("=" * 60)
    print("  TRAINING: AI Quantum Circuit Optimizer")
    print("=" * 60)
    print(f"  Circuit depth:    {circuit_depth} gates")
    print(f"  Training steps:   {total_timesteps:,}")
    print(f"  Agent type:       DQN (Deep Q-Network)")
    print("=" * 60)
    print()

    # --- Create environment ---
    # Monitor wraps it to auto-track episode rewards/lengths
    env = Monitor(QuantumCircuitOptEnv(circuit_depth=circuit_depth, max_steps=50))
    # DummyVecEnv wraps it into the vectorized format SB3 expects
    vec_env = DummyVecEnv([lambda: env])

    # --- Create the DQN agent ---
    # policy="MlpPolicy" means: use a simple feedforward neural network
    # learning_rate: how fast the network updates its weights
    # batch_size: how many experiences to learn from at once
    # gamma: discount factor — how much the agent values future rewards
    #        0.95 = "I care a lot about future rewards"
    # exploration_fraction: how long to explore randomly before exploiting
    # verbose=0 keeps SB3's own logging quiet (we use our callback)
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-3,
        batch_size=64,
        gamma=0.95,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,  # Always keep 5% random exploration
        buffer_size=10_000,  # Fixed: was replay_buffer_size in older versions
        verbose=0,
        seed=42,
    )

    # --- Train with our custom callback ---
    callback = QuantumMetricsCallback(log_interval=100)
    callback._total_steps_estimate = total_timesteps   # needed for phase binning
    print("Training in progress...\n")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # --- Save the trained model ---
    model.save(model_path)
    print(f"\n✓ Model saved to '{model_path}.zip'")

    return model, callback


# ---------------------------------------------------------------------------
# QISKIT TRANSPILER BASELINE (Improvement 3)
# ---------------------------------------------------------------------------
# We try to import Qiskit.  If it's installed we run its transpiler on the
# exact same circuits the RL agent sees, so the comparison is apples-to-apples.
# If Qiskit isn't installed we skip gracefully and just show the RL results.
# ---------------------------------------------------------------------------
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def _names_to_qiskit_circuit(names):
    """Convert our gate-name list into a Qiskit QuantumCircuit (1 qubit)."""
    qc = QuantumCircuit(1)
    gate_map = {'H': qc.h, 'X': qc.x, 'Z': qc.z, 'S': qc.s, 'T': qc.t, 'I': qc.id}
    for name in names:
        gate_map[name](0)
    return qc


def qiskit_optimize(circuit_names, optimization_level=3):
    """
    Run Qiskit's built-in transpiler on a circuit and return the optimised
    gate count.  optimization_level=3 is the most aggressive setting Qiskit
    offers — it's what real quantum teams use before sending to hardware.

    Returns:
        int | None: optimised gate count, or None if Qiskit isn't available.
    """
    if not QISKIT_AVAILABLE:
        return None
    qc = _names_to_qiskit_circuit(circuit_names)
    backend = AerSimulator()
    optimised = transpile(qc, backend=backend, optimization_level=optimization_level)
    # Count only real gates (ignore barriers / ancillas)
    return optimised.count_ops().get('u3', 0) + \
           optimised.count_ops().get('cx', 0) + \
           sum(v for k, v in optimised.count_ops().items()
               if k not in ('barrier', 'measure'))


# ---------------------------------------------------------------------------
# EVALUATION — See how well the trained agent does
# ---------------------------------------------------------------------------

ACTION_NAMES = {
    0: "Cancel H·H/X·X/Z·Z",
    1: "Merge S·S → Z",
    2: "Merge T·T → S",
    3: "Sub H·X·H → Z",
    4: "Remove I",
    5: "Pass",
}


def evaluate_agent(model, num_episodes=50, circuit_depth=12):
    """
    Run the trained agent on fresh circuits and measure performance.
    Also runs the Qiskit transpiler on the same circuits for a fair comparison.
    Records per-episode action sequences for inspectability analysis.
    """
    print("\n" + "=" * 60)
    print("  EVALUATING trained agent")
    print("=" * 60)

    env = QuantumCircuitOptEnv(circuit_depth=circuit_depth, max_steps=50)

    all_compression_ratios = []
    all_gates_removed      = []
    all_rewards            = []
    all_depth_ratios       = []                  # Improvement 2
    all_episode_actions    = []                  # Improvement 4

    # --- Qiskit baseline lists (Improvement 3) ---
    qiskit_gate_counts     = []
    rl_final_gate_counts   = []
    original_gate_counts   = []

    if QISKIT_AVAILABLE:
        print("  Qiskit detected — running transpiler baseline in parallel.\n")
    else:
        print("  Qiskit not installed — skipping transpiler baseline.\n")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        original_len          = env.original_depth
        original_circuit_depth = env.original_circuit_depth
        original_names        = list(env.circuit_names)   # snapshot for Qiskit
        total_reward          = 0
        done                  = False
        episode_actions       = []                        # Improvement 4

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            episode_actions.append(action)                # record every action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        final_len   = info['circuit_length']
        final_depth = info.get('circuit_depth', final_len)
        compression = final_len / original_len
        depth_ratio = final_depth / max(original_circuit_depth, 1)

        all_compression_ratios.append(compression)
        all_gates_removed.append(original_len - final_len)
        all_rewards.append(total_reward)
        all_depth_ratios.append(depth_ratio)
        all_episode_actions.append(episode_actions)

        # --- Qiskit baseline on the SAME original circuit (Improvement 3) ---
        original_gate_counts.append(original_len)
        rl_final_gate_counts.append(final_len)
        if QISKIT_AVAILABLE:
            qiskit_result = qiskit_optimize(original_names)
            qiskit_gate_counts.append(qiskit_result if qiskit_result else original_len)

    # --- Print summary ---
    print(f"  Episodes evaluated:       {num_episodes}")
    print(f"  Avg compression ratio:    {np.mean(all_compression_ratios):.2%}")
    print(f"  Avg depth ratio:          {np.mean(all_depth_ratios):.2%}")
    print(f"  Avg gates removed:        {np.mean(all_gates_removed):.1f}")
    print(f"  Best compression:         {np.min(all_compression_ratios):.2%}")
    print(f"  Avg episode reward:       {np.mean(all_rewards):.2f}")
    print(f"  Best episode reward:      {np.max(all_rewards):.2f}")

    if QISKIT_AVAILABLE and qiskit_gate_counts:
        print(f"\n  --- Qiskit Transpiler Baseline ---")
        print(f"  Avg gates after Qiskit:   {np.mean(qiskit_gate_counts):.1f}")
        print(f"  Avg gates after RL:       {np.mean(rl_final_gate_counts):.1f}")
        print(f"  Avg original gates:       {np.mean(original_gate_counts):.1f}")

    # --- Action frequency across all evaluation episodes (Improvement 4) ---
    flat_actions = [a for ep in all_episode_actions for a in ep]
    action_counts = np.bincount(flat_actions, minlength=6)

    print(f"\n  --- Agent Action Frequencies ---")
    for idx in range(6):
        pct = action_counts[idx] / max(len(flat_actions), 1) * 100
        print(f"    {ACTION_NAMES[idx]:>22s}  →  {action_counts[idx]:>5d}  ({pct:5.1f}%)")

    return {
        'compression_ratios':   all_compression_ratios,
        'gates_removed':        all_gates_removed,
        'rewards':              all_rewards,
        'depth_ratios':         all_depth_ratios,
        'episode_actions':      all_episode_actions,
        'action_counts':        action_counts,
        # Qiskit benchmark
        'original_gate_counts': original_gate_counts,
        'rl_final_gate_counts': rl_final_gate_counts,
        'qiskit_gate_counts':   qiskit_gate_counts,
        'qiskit_available':     QISKIT_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# MAIN — Run training + evaluation + save plots
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- TRAIN ---
    model, callback = train_agent(
        total_timesteps=80_000,
        circuit_depth=12,
    )

    # --- EVALUATE ---
    results = evaluate_agent(model, num_episodes=50, circuit_depth=12)

    # --- PLOT RESULTS (6-panel figure) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("AI Quantum Circuit Optimizer — Full Analysis", fontsize=16, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('#fafafa')

    # ── Panel 1: Compression ratio per evaluation episode ──
    ax = axes[0, 0]
    ax.plot(results['compression_ratios'], color='#6c63ff', linewidth=1.5, alpha=0.8)
    ax.axhline(y=np.mean(results['compression_ratios']),
               color='#f59e0b', linestyle='--', linewidth=1.2,
               label=f"Mean: {np.mean(results['compression_ratios']):.2%}")
    ax.set_ylabel('Compression Ratio (lower = better)')
    ax.set_xlabel('Evaluation Episode')
    ax.set_title('① Gate-Count Compression')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Depth ratio per evaluation episode (Improvement 2) ──
    ax = axes[0, 1]
    ax.plot(results['depth_ratios'], color='#10b981', linewidth=1.5, alpha=0.8)
    ax.axhline(y=np.mean(results['depth_ratios']),
               color='#f59e0b', linestyle='--', linewidth=1.2,
               label=f"Mean: {np.mean(results['depth_ratios']):.2%}")
    ax.set_ylabel('Depth Ratio (lower = better)')
    ax.set_xlabel('Evaluation Episode')
    ax.set_title('② Circuit-Depth Reduction')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: Gates removed per episode ──
    ax = axes[0, 2]
    ax.bar(range(len(results['gates_removed'])),
           results['gates_removed'], color='#a78bfa', alpha=0.6, width=0.8)
    ax.axhline(y=np.mean(results['gates_removed']),
               color='#f59e0b', linestyle='--', linewidth=1.2,
               label=f"Mean: {np.mean(results['gates_removed']):.1f}")
    ax.set_ylabel('Gates Removed')
    ax.set_xlabel('Evaluation Episode')
    ax.set_title('③ Gates Removed per Episode')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Learning curves — compression + depth over training ──
    ax = axes[1, 0]
    window = 50
    if callback.compression_ratios:
        comp_arr   = np.array(callback.compression_ratios)
        comp_smooth = np.convolve(comp_arr, np.ones(window)/window, mode='valid')
        ax.plot(comp_smooth, color='#6c63ff', linewidth=1.5, label='Gate compression')
    if callback.depth_ratios:
        depth_arr  = np.array(callback.depth_ratios)
        depth_smooth = np.convolve(depth_arr, np.ones(window)/window, mode='valid')
        ax.plot(depth_smooth, color='#10b981', linewidth=1.5, label='Depth ratio')
    ax.set_ylabel('Ratio (lower = better)')
    ax.set_xlabel('Training Episode')
    ax.set_title('④ Learning Curves (smoothed)')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 5: RL vs Qiskit transpiler benchmark (Improvement 3) ──
    ax = axes[1, 1]
    episodes = np.arange(len(results['original_gate_counts']))
    ax.plot(episodes, results['original_gate_counts'],
            color='#888', linewidth=1.2, linestyle=':',  label='Original')
    ax.plot(episodes, results['rl_final_gate_counts'],
            color='#6c63ff', linewidth=2,                label='RL Optimizer')
    if results['qiskit_available'] and results['qiskit_gate_counts']:
        ax.plot(episodes, results['qiskit_gate_counts'],
                color='#f59e0b', linewidth=2, linestyle='--', label='Qiskit (level 3)')
    ax.set_ylabel('Gate Count')
    ax.set_xlabel('Evaluation Episode')
    ax.set_title('⑤ RL vs Qiskit Transpiler')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    if not results['qiskit_available']:
        ax.text(0.5, 0.08, 'Install qiskit + qiskit-aer to enable',
                ha='center', fontsize=9, color='#aaa', transform=ax.transAxes,
                style='italic')

    # ── Panel 6: Action-frequency heatmap across training phases (Improvement 4) ──
    ax = axes[1, 2]
    phases     = ['Early', 'Mid', 'Late']
    action_ids = list(range(6))
    action_labels = [ACTION_NAMES[i] for i in action_ids]

    # Build a 6×3 matrix: rows = actions, columns = training phases
    heatmap = np.zeros((6, 3))
    for col, phase in enumerate(['early', 'mid', 'late']):
        phase_actions = callback.action_log_by_phase.get(phase, [])
        if len(phase_actions) > 0:
            counts = np.bincount(phase_actions, minlength=6)
            heatmap[:, col] = counts / counts.sum()   # normalise to probability

    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0, vmax=heatmap.max())
    ax.set_xticks(range(3))
    ax.set_xticklabels(phases, fontsize=10)
    ax.set_yticks(range(6))
    ax.set_yticklabels(action_labels, fontsize=9)
    ax.set_xlabel('Training Phase')
    ax.set_title('⑥ Policy Evolution Over Training')

    # Annotate each cell with the percentage
    for row in range(6):
        for col in range(3):
            val = heatmap[row, col]
            color = 'white' if val > heatmap.max() * 0.6 else 'black'
            ax.text(col, row, f'{val:.0%}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Action probability')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Results plot saved to 'training_results.png'")