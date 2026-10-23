"""Analyze the frozen curated split without downloading data or models."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def describe(values):
    return {'count': len(values), 'min': min(values), 'max': max(values),
            'mean': round(mean(values), 2), 'median': median(values)}


def analyze(data):
    rows = {name: data[name] for name in ('train', 'validation')}
    seen_ids, seen_articles = set(), set()
    result = {}
    for name, examples in rows.items():
        if not examples:
            raise ValueError(f'Empty split: {name}')
        for row in examples:
            identity = str(row['id'])
            article = row['article']
            summary = row['reviewed_summary']
            if not isinstance(article, str) or not article.strip() or not isinstance(summary, str) or not summary.strip():
                raise ValueError(f'Invalid source/target: {identity}')
            normalized = ' '.join(article.casefold().split())
            if identity in seen_ids or normalized in seen_articles:
                raise ValueError(f'Duplicate ID or normalized article: {identity}')
            seen_ids.add(identity)
            seen_articles.add(normalized)
        source = [len(row['article'].split()) for row in examples]
        target = [len(row['reviewed_summary'].split()) for row in examples]
        result[name] = {'articles': describe(source), 'summaries': describe(target),
                        'mean_word_reduction_pct': round(mean(100 * (1-b/a) for a, b in zip(source, target)), 2),
                        'targets_over_40_words': sum(n > 40 for n in target)}
    return {'splits': result, 'duplicate_ids_or_articles': 0,
            'note': 'Whitespace word counts; no tokenization or filtering. Exact normalized overlap only; semantic duplication is not certified.'}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root/'data/curation/qwen3_curated_1200_v1/training_data.json')
    parser.add_argument('--output-dir', type=Path, default=root/'reports/curated_data')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    result = analyze(data)
    result.update(input_sha256=hashlib.sha256(raw).hexdigest(), manifest=data.get('manifest', {}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    all_rows = data['train'] + data['validation']
    source_bins = np.histogram_bin_edges([len(r['article'].split()) for r in all_rows], bins=25)
    target_bins = np.histogram_bin_edges([len(r['reviewed_summary'].split()) for r in all_rows], bins=20)
    reduction_bins = np.histogram_bin_edges([100*(1-len(r['reviewed_summary'].split())/len(r['article'].split())) for r in all_rows], bins=25)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {'train': '#147d92', 'validation': '#d87635'}
    for name in ('train', 'validation'):
        source = [len(row['article'].split()) for row in data[name]]
        target = [len(row['reviewed_summary'].split()) for row in data[name]]
        weights = [100 / len(source)] * len(source)
        axes[0, 0].hist(source, bins=source_bins, weights=weights, alpha=.55, label=name, color=colors[name])
        axes[0, 1].hist(target, bins=target_bins, weights=weights, alpha=.55, label=name, color=colors[name])
        axes[1, 0].scatter(source, target, s=12, alpha=.4, label=name, color=colors[name])
        axes[1, 1].hist([100*(1-b/a) for a,b in zip(source,target)], bins=reduction_bins, weights=weights, alpha=.55, label=name, color=colors[name])
    titles = ['Article word counts', 'Reviewed summary word counts', 'Article vs summary length', 'Word reduction']
    labels = [('Article words', 'Split examples (%)'), ('Summary words', 'Split examples (%)'), ('Article words', 'Summary words'), ('Reduction (%)', 'Split examples (%)')]
    for ax, title, (xlabel, ylabel) in zip(axes.flat, titles, labels):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=.15)
    fig.suptitle('Curated training data | Frozen train / validation split', fontsize=15)
    fig.tight_layout(rect=(0, .03, 1, .95))
    fig.text(.02, .01, 'Word reduction measures length, not factual accuracy. No records were removed.', fontsize=9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(args.output_dir/f'curated_data_dashboard.{extension}', dpi=160)
    plt.close(fig)
    (args.output_dir/'statistics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved curated data analysis:', args.output_dir)
    print('Train:', len(data['train']), '| Validation:', len(data['validation']))
    print('No dataset, split or model was changed.')


if __name__ == '__main__':
    main()
