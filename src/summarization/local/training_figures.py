"""Plot the explicitly identified curated-v2 training run."""
import json
import math


def build_training_figures(root, save):
    import matplotlib.pyplot as plt

    folder = root / 'reports/training/qwen3_curated_v2'
    history_path = folder / 'training_history.json'
    best_path = folder / 'best_validation.json'
    if not history_path.is_file() or not best_path.is_file():
        return 'Curated-v2 training records are not installed.'
    history = json.loads(history_path.read_text(encoding='utf-8'))
    best = json.loads(best_path.read_text(encoding='utf-8'))
    from .settings import ADAPTER_SHA
    if best.get('adapter_sha256') != ADAPTER_SHA:
        raise ValueError('Training figures require the verified curated-v2 adapter identity.')
    train = [row for row in history if 'loss' in row and 'step' in row]
    validation = [row for row in history if 'eval_loss' in row and 'step' in row]
    if not train or not validation:
        raise ValueError('Training history has no usable train/validation observations.')
    selected = min(validation, key=lambda row: row['eval_loss'])
    if selected['step'] != best['global_step'] or not math.isclose(selected['eval_loss'], best['eval_loss']):
        raise ValueError('Best-checkpoint metadata does not match the training history.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].plot([r['step'] for r in train], [r['loss'] for r in train], color='#147d92', label='Logged training loss')
    axes[0].plot([r['step'] for r in validation], [r['eval_loss'] for r in validation], 'o-', color='#d87635', label='Validation loss')
    axes[0].axvline(best['global_step'], color='#667085', linestyle='--', linewidth=1)
    axes[0].set(title='Training and validation', xlabel='Optimizer step', ylabel='Cross-entropy loss')
    axes[0].legend(fontsize=8)
    axes[1].plot([r['epoch'] for r in validation], [r['eval_loss'] for r in validation], 'o-', color='#d87635')
    for row in validation:
        axes[1].annotate(f"{row['eval_loss']:.4f}", (row['epoch'], row['eval_loss']), xytext=(0, 10), textcoords='offset points', ha='center')
    axes[1].scatter([best['epoch']], [best['eval_loss']], s=110, facecolors='none', edgecolors='#147d92', linewidth=2)
    axes[1].set(title=f"Selected checkpoint: epoch {best['epoch']:g} / step {best['global_step']}", xlabel='Epoch', ylabel='Validation loss', xticks=[r['epoch'] for r in validation])
    axes[1].margins(x=.1, y=.3)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle('Qwen3-14B | Curated v2 | 924 train / 231 validation', fontsize=14)
    save(fig, '08_qwen_training_loss', 'Recorded training batches and three epoch-level validation observations. Loss does not measure factual accuracy; the two series use different aggregations.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, key, title, color in zip(axes, ['learning_rate', 'grad_norm'], ['Logged learning rate', 'Logged gradient norm'], ['#147d92', '#7866ab']):
        rows = [row for row in train if key in row]
        ax.plot([row['step'] for row in rows], [row[key] for row in rows], color=color)
        ax.set(title=title, xlabel='Optimizer step', ylabel=key.replace('_', ' ').capitalize())
        ax.grid(alpha=.18)
    fig.suptitle('Training diagnostics | Recorded observations', fontsize=14)
    save(fig, '09_qwen_training_diagnostics', 'Learning rate and gradient norm at logged steps, not reconstructed intermediate updates.')
    return str(history_path.relative_to(root))
