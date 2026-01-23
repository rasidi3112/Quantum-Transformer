from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for visualization")

def plot_attention_weights(
    attention: np.ndarray,
    tokens: Optional[List[str]] = None,
    head: int = 0,
    layer: int = 0,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
):

    _check_matplotlib()

    if attention.ndim == 4:
        attn = attention[layer, head]
    elif attention.ndim == 3:
        attn = attention[head]
    else:
        attn = attention

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    if tokens:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Quantum Attention (Head {head}, Layer {layer})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_circuit_diagram(
    n_qubits: int,
    n_layers: int,
    circuit_type: str = "transformer",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
):

    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_qubits):
        ax.axhline(y=i, color='black', linewidth=1)
        ax.text(-1, i, f'q{i}', fontsize=10, va='center', ha='right')

    x_pos = 0
    colors = {'RY': 'lightblue', 'RZ': 'lightgreen', 'CNOT': 'orange'}

    for layer in range(n_layers):

        for q in range(n_qubits):
            for gate, color in [('RY', 'lightblue'), ('RZ', 'lightgreen')]:
                rect = mpatches.Rectangle(
                    (x_pos - 0.2, q - 0.25), 0.4, 0.5,
                    linewidth=1, edgecolor='blue', facecolor=color
                )
                ax.add_patch(rect)
                ax.text(x_pos, q, gate, fontsize=7, ha='center', va='center')
                x_pos += 0.5

        x_pos += 0.3

        for q in range(n_qubits - 1):
            ax.plot([x_pos, x_pos], [q, q + 1], 'k-', linewidth=2)
            ax.plot(x_pos, q, 'ko', markersize=8)
            ax.plot(x_pos, q + 1, 'ko', markersize=12, fillstyle='none')

        x_pos += 0.8

    ax.set_xlim(-1.5, x_pos + 0.5)
    ax.set_ylim(-0.5, n_qubits - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Quantum {circuit_type.title()} Circuit\n({n_qubits} qubits, {n_layers} layers)', fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[dict] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):

    _check_matplotlib()

    n_plots = 1 + (1 if train_metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)

    if val_losses:
        ax.plot(epochs, val_losses, 'r--', label='Validation', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if train_metrics and n_plots > 1:
        ax = axes[1]
        for name, values in train_metrics.items():
            ax.plot(range(1, len(values) + 1), values, label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric')
        ax.set_title('Training Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_quantum_state(
    state: np.ndarray,
    n_qubits: int,
    plot_type: str = "bar",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):

    _check_matplotlib()

    dim = 2 ** n_qubits
    probs = np.abs(state) ** 2
    phases = np.angle(state)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    labels = [f'|{i:0{n_qubits}b}⟩' for i in range(dim)]
    ax.bar(range(dim), probs, color='steelblue')
    ax.set_xticks(range(dim))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Probability')
    ax.set_title('State Probabilities')
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.bar(range(dim), phases, color='coral')
    ax.set_xticks(range(dim))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Phase (radians)')
    ax.set_title('State Phases')
    ax.set_ylim(-np.pi, np.pi)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
