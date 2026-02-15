"""
============================================================
QUANTUM CIRCUIT ENVIRONMENT — The "game" the RL agent plays
============================================================

CONCEPT:
  RL needs an "environment" — think of it like a game.
  The agent takes actions, the environment returns a reward.
  Here: the agent's actions = applying gate transformations
        the reward         = how much the circuit shrinks
                              while staying equivalent

WHY THIS MATTERS:
  On real quantum hardware, every gate adds noise.
  Fewer gates = less noise = better results.
  So optimizing circuits is literally the #1 bottleneck
  blocking quantum computers from being useful today.

WHAT THE AGENT LEARNS:
  - Which gates can be removed (redundant pairs)
  - Which gates can be merged (e.g., two H gates cancel)
  - Which gate sequences can be replaced with shorter ones
============================================================
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# QUANTUM GATE DEFINITIONS
# ---------------------------------------------------------------------------
# We represent gates as 2x2 or 4x4 unitary matrices.
# A unitary matrix U satisfies: U† U = I  (reversible, no information loss)
# This is fundamental to quantum computing — all operations are reversible.
# ---------------------------------------------------------------------------

# Hadamard: puts a qubit into superposition  |0⟩ → (|0⟩+|1⟩)/√2
H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                   [1, -1]], dtype=complex)

# Pauli-X: flips a qubit  |0⟩ → |1⟩, |1⟩ → |0⟩  (quantum NOT)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

# Pauli-Z: phase flip  |0⟩ → |0⟩, |1⟩ → -|1⟩
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

# Identity: does nothing (but takes up a gate slot = wastes depth)
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

# S gate (Phase): |0⟩ → |0⟩, |1⟩ → i|1⟩
S = np.array([[1, 0],
              [0, 1j]], dtype=complex)

# T gate: |0⟩ → |0⟩, |1⟩ → e^(iπ/4)|1⟩
T = np.array([[1, 0],
              [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# All single-qubit gates the agent can work with
SINGLE_GATES = {
    'H': H,
    'X': X,
    'Z': Z,
    'I': I,
    'S': S,
    'T': T,
}

# ---------------------------------------------------------------------------
# KEY IDENTITIES THE AGENT SHOULD DISCOVER:
#   H·H = I          (two Hadamards cancel)
#   X·X = I          (two X gates cancel)
#   Z·Z = I          (two Z gates cancel)
#   S·S = Z          (two S gates = one Z gate)
#   T·T = S          (two T gates = one S gate)
#   H·X·H = Z        (Hadamard conjugation)
# ---------------------------------------------------------------------------


def matrices_are_equivalent(A, B, tolerance=1e-6):
    """
    Check if two unitary matrices are equivalent up to global phase.

    WHY global phase?
      In quantum mechanics, |ψ⟩ and e^(iθ)|ψ⟩ are the SAME physical state.
      So U and e^(iθ)·U represent the same quantum operation.
      We have to account for this when comparing circuits.
    """
    if A.shape != B.shape:
        return False

    # Find the first non-zero element to extract the global phase
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i, j]) > tolerance:
                # phase = B[i,j] / A[i,j] — if they're equivalent,
                # this ratio should be the SAME for all elements
                phase = B[i, j] / A[i, j]
                # Check if A * phase ≈ B everywhere
                return np.allclose(A * phase, B, atol=tolerance)
    return np.allclose(A, B, atol=tolerance)


def compute_unitary(circuit):
    """
    Multiply all gate matrices in the circuit together.
    The resulting matrix = the full quantum operation of the circuit.

    Matrix multiplication order: RIGHT to LEFT
      circuit = [G1, G2, G3]  →  U = G3 · G2 · G1
    This is because quantum states are column vectors,
    and we apply gates left-to-right: G1 first, then G2, then G3.
    """
    if len(circuit) == 0:
        return I.copy()

    result = circuit[-1].copy()
    for gate in reversed(circuit[:-1]):
        result = result @ gate
    return result


def compute_circuit_depth(circuit_names):
    """
    Compute the depth of a single-qubit circuit.

    For a single-qubit circuit every gate is sequential so depth == length.
    We factor this into its own function now so that when we extend to
    multi-qubit circuits later (where gates on different qubits can run
    in parallel) the depth calculation lives in one place and just needs
    to be made smarter — without touching the rest of the code.

    Returns:
        int: the circuit depth
    """
    return len(circuit_names)


def generate_random_circuit(depth, seed=None):
    """
    Create a random quantum circuit of a given depth.
    This is what we'll ask the RL agent to optimize.

    We intentionally insert some redundancies so the agent
    has something to find and fix.
    """
    if seed is not None:
        np.random.seed(seed)

    gate_names = list(SINGLE_GATES.keys())
    circuit = []
    circuit_names = []

    for _ in range(depth):
        name = np.random.choice(gate_names)
        circuit.append(SINGLE_GATES[name].copy())
        circuit_names.append(name)

    # Intentionally add some canceling pairs so there's optimization to find
    # This makes the problem solvable for a learning agent
    insert_pairs = np.random.randint(1, max(2, depth // 3))
    for _ in range(insert_pairs):
        pos = np.random.randint(0, len(circuit))
        pair_gate = np.random.choice(['H', 'X', 'Z'])  # self-inverse gates
        circuit.insert(pos, SINGLE_GATES[pair_gate].copy())
        circuit.insert(pos + 1, SINGLE_GATES[pair_gate].copy())
        circuit_names.insert(pos, pair_gate)
        circuit_names.insert(pos + 1, pair_gate)

    return circuit, circuit_names


# ---------------------------------------------------------------------------
# THE GYMNASIUM ENVIRONMENT
# ---------------------------------------------------------------------------
# Gymnasium is the standard RL framework (successor to OpenAI Gym).
# Every RL environment must implement:
#   reset()  → returns initial observation
#   step()   → takes action, returns (obs, reward, done, info)
# ---------------------------------------------------------------------------

class QuantumCircuitOptEnv(gym.Env):
    """
    Environment where the RL agent optimizes a quantum circuit.

    STATE (observation):  A flattened vector encoding the current circuit.
                          Each gate is encoded as a one-hot vector.
                          The agent "sees" the circuit as a sequence of numbers.

    ACTION:               An integer choosing which optimization rule to apply:
                          0 = Try to cancel adjacent identical self-inverse gates
                          1 = Try to merge adjacent S·S → Z
                          2 = Try to merge adjacent T·T → S
                          3 = Try to apply H·X·H → Z substitution
                          4 = Remove any Identity gate
                          5 = Do nothing (pass)

    REWARD:               Positive for reducing gate count while keeping
                          the circuit equivalent. Negative for invalid moves.
    """

    def __init__(self, circuit_depth=12, max_steps=50):
        super().__init__()

        self.circuit_depth = circuit_depth
        self.max_steps = max_steps
        self.max_circuit_len = circuit_depth * 3  # upper bound after insertions

        # One-hot encoding size: one slot per gate type + padding
        self.num_gate_types = len(SINGLE_GATES)
        self.obs_size = self.max_circuit_len * self.num_gate_types

        # Observation: flattened one-hot encoded circuit
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )

        # 6 possible optimization actions
        self.action_space = spaces.Discrete(6)

        # State variables (set in reset)
        self.circuit = None
        self.circuit_names = None
        self.target_unitary = None
        self.original_depth = None
        self.steps_taken = 0

        # Gate name → index mapping for one-hot encoding
        self.gate_to_idx = {name: i for i, name in enumerate(SINGLE_GATES.keys())}

    def _encode_observation(self):
        """Convert the circuit into a fixed-size observation vector."""
        obs = np.zeros(self.obs_size, dtype=np.float32)
        for i, name in enumerate(self.circuit_names):
            if i >= self.max_circuit_len:
                break
            idx = self.gate_to_idx[name]
            obs[i * self.num_gate_types + idx] = 1.0
        return obs

    def reset(self, seed=None, options=None):
        """Generate a new random circuit to optimize."""
        super().reset(seed=seed)
        self.circuit, self.circuit_names = generate_random_circuit(
            self.circuit_depth, seed=seed
        )
        # Lock in the target: the agent must preserve this
        self.target_unitary  = compute_unitary(self.circuit)
        self.original_depth  = len(self.circuit)
        self.original_circuit_depth = compute_circuit_depth(self.circuit_names)  # depth at start
        self.steps_taken = 0
        return self._encode_observation(), {}

    def step(self, action):
        """Apply the chosen optimization action and return results."""
        self.steps_taken += 1
        old_len = len(self.circuit)
        valid_move = False

        # Snapshot the circuit BEFORE we mutate it.
        # If the explicit equivalence check later fails we roll back here.
        old_circuit_snapshot  = [g.copy() for g in self.circuit]
        old_names_snapshot    = list(self.circuit_names)

        # ---------------------------------------------------------------
        # ACTION 0: Cancel adjacent self-inverse gates (H·H, X·X, Z·Z)
        # ---------------------------------------------------------------
        if action == 0:
            self_inverse = {'H', 'X', 'Z'}
            for i in range(len(self.circuit_names) - 1):
                if (self.circuit_names[i] in self_inverse and
                        self.circuit_names[i] == self.circuit_names[i + 1]):
                    # Found a canceling pair — remove both
                    self.circuit.pop(i + 1)
                    self.circuit.pop(i)
                    self.circuit_names.pop(i + 1)
                    self.circuit_names.pop(i)
                    valid_move = True
                    break

        # ---------------------------------------------------------------
        # ACTION 1: Merge S·S → Z
        # ---------------------------------------------------------------
        elif action == 1:
            for i in range(len(self.circuit_names) - 1):
                if self.circuit_names[i] == 'S' and self.circuit_names[i + 1] == 'S':
                    self.circuit.pop(i + 1)
                    self.circuit[i] = Z.copy()
                    self.circuit_names.pop(i + 1)
                    self.circuit_names[i] = 'Z'
                    valid_move = True
                    break

        # ---------------------------------------------------------------
        # ACTION 2: Merge T·T → S
        # ---------------------------------------------------------------
        elif action == 2:
            for i in range(len(self.circuit_names) - 1):
                if self.circuit_names[i] == 'T' and self.circuit_names[i + 1] == 'T':
                    self.circuit.pop(i + 1)
                    self.circuit[i] = S.copy()
                    self.circuit_names.pop(i + 1)
                    self.circuit_names[i] = 'S'
                    valid_move = True
                    break

        # ---------------------------------------------------------------
        # ACTION 3: H·X·H → Z conjugation
        # ---------------------------------------------------------------
        elif action == 3:
            for i in range(len(self.circuit_names) - 2):
                if (self.circuit_names[i] == 'H' and
                        self.circuit_names[i + 1] == 'X' and
                        self.circuit_names[i + 2] == 'H'):
                    # Replace H·X·H with single Z gate
                    self.circuit.pop(i + 2)
                    self.circuit.pop(i + 1)
                    self.circuit[i] = Z.copy()
                    self.circuit_names.pop(i + 2)
                    self.circuit_names.pop(i + 1)
                    self.circuit_names[i] = 'Z'
                    valid_move = True
                    break

        # ---------------------------------------------------------------
        # ACTION 4: Remove Identity gates
        # ---------------------------------------------------------------
        elif action == 4:
            for i in range(len(self.circuit_names)):
                if self.circuit_names[i] == 'I':
                    self.circuit.pop(i)
                    self.circuit_names.pop(i)
                    valid_move = True
                    break

        # ---------------------------------------------------------------
        # ACTION 5: Pass (do nothing)
        # ---------------------------------------------------------------
        elif action == 5:
            valid_move = True  # Always valid, but no reward

        # ---------------------------------------------------------------
        # COMPUTE REWARD
        # ---------------------------------------------------------------
        gates_removed = old_len - len(self.circuit)

        # --- EXPLICIT EQUIVALENCE VERIFICATION (Improvement 1) ---
        # Every time gates are removed we recompute the full unitary of the
        # optimised circuit and compare it to the locked-in target unitary.
        # Comparison is done up to global phase: two unitaries U and V
        # represent the same quantum operation when U = e^(iθ)·V for some θ.
        # This is the rigorous correctness check — not just a safety net,
        # but the ground-truth proof that the optimised circuit is equivalent.
        # We only run it when the circuit length actually changed to avoid
        # paying the O(n) matrix-multiply cost on no-ops.
        equivalence_verified = None          # None = not checked, True/False = result
        if gates_removed > 0:
            current_unitary  = compute_unitary(self.circuit)
            equivalence_verified = matrices_are_equivalent(
                current_unitary, self.target_unitary
            )

        if not valid_move:
            # Agent tried something impossible — penalize
            reward = -0.5
        elif gates_removed > 0:
            if equivalence_verified:
                # GOOD: removed gates AND unitary equivalence confirmed
                reward = float(gates_removed) * 1.0
            else:
                # Equivalence check FAILED — revert the circuit to the state
                # before this step so training isn't poisoned by a bad state.
                # (This should never fire with our rule set, but the explicit
                # check catches it if it ever does.)
                self.circuit      = old_circuit_snapshot
                self.circuit_names = old_names_snapshot
                reward = -2.0
        else:
            # Valid move but nothing changed (e.g., pass)
            reward = -0.1  # Small penalty to discourage doing nothing

        # --- DEPTH BONUS / PENALTY (Improvement 2) ---
        # Real quantum hardware cares about depth just as much as gate count.
        # Depth = the longest sequential path through the circuit.
        # We add a small bonus when depth shrinks and a small penalty when it
        # grows (even if gate count dropped) to nudge the agent toward
        # solutions that actually help on real hardware.
        old_depth = compute_circuit_depth(old_names_snapshot)
        new_depth = compute_circuit_depth(self.circuit_names)
        depth_delta = old_depth - new_depth          # positive = depth shrank

        if gates_removed > 0 and equivalence_verified:
            if depth_delta > 0:
                reward += 0.3 * depth_delta          # bonus: depth decreased
            elif depth_delta < 0:
                reward -= 0.2 * abs(depth_delta)     # penalty: depth increased despite gate removal

        # Episode ends when: no more steps, or circuit is fully optimized
        terminated = (len(self.circuit) == 0) or self.steps_taken >= self.max_steps
        truncated  = self.steps_taken >= self.max_steps

        obs = self._encode_observation()
        info = {
            'circuit_length':          len(self.circuit),
            'original_length':         self.original_depth,
            'compression_ratio':       len(self.circuit) / max(self.original_depth, 1),
            'circuit_names':           list(self.circuit_names),
            # --- depth fields (Improvement 2) ---
            'circuit_depth':           new_depth,
            'original_circuit_depth':  self.original_circuit_depth,
            'depth_ratio':             new_depth / max(self.original_circuit_depth, 1),
            # --- verification flag (Improvement 1) ---
            'equivalence_verified':    equivalence_verified,
        }

        return obs, reward, terminated, truncated, info