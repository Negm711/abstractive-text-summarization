"""Create reproducible figures from existing reports; never downloads data/models."""
import argparse
import json
from pathlib import Path
import statistics
from .settings import PROJECT_ROOT


def build_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root = Path(root)
    reports = root / 'reports'
    destination = reports / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#334155',
                         'text.color': '#172033', 'savefig.facecolor': 'white'})
    teal, slate, amber = '#087f8c', '#53657e', '#d18a16'
    generated = []

    def save(fig, name, caption):
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.text(0.02, 0.015, caption, fontsize=8, color='#53657e')
        for extension in ('png', 'svg'):
            fig.savefig(destination / f'{name}.{extension}', dpi=160, bbox_inches='tight')
        plt.close(fig)
        generated.append({'file': name + '.png', 'caption': caption})

    audit_path = reports / 'data_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())['splits_metrics']
        splits = ['train', 'validation', 'test']
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(3)
        for offset, key, label, color in [(-0.2, 'exact_documents', 'Duplicate documents', slate),
                                         (0, 'exact_doc_sum_pairs', 'Duplicate article-summary pairs', teal)]:
            values = [audit[s]['duplicates'][key] for s in splits]
            bars = ax.bar(x + offset, values, width=0.2, label=label, color=color)
            ax.bar_label(bars, padding=3)
        values = [audit[s]['source_column']['empty_or_whitespace'] for s in splits]
        bars = ax.bar(x + 0.2, values, width=0.2, label='Empty articles', color=amber)
        ax.bar_label(bars, padding=3)
        ax.set(xticks=x, xticklabels=splits, ylabel='Rows / extra occurrences', title='Recorded XSum data-quality audit')
        ax.legend(fontsize=8)
        ax.margins(y=0.22)
        save(fig, '01_raw_data_quality', 'Historical raw-dataset audit; exact duplicates only. This is not the curated Qwen training sample.')

    cleaning = reports / 'cleaning_summary.json'
    if cleaning.exists():
        stats = json.loads(cleaning.read_text())['statistics']
        splits = ['train', 'validation', 'test']
        fig, ax = plt.subplots(figsize=(9, 4.8))
        values = [100 * stats[s]['processed_count'] / stats[s]['raw_count'] for s in splits]
        bars = ax.bar(splits, values, color=teal, width=0.55)
        ax.set(ylim=(0, 110), ylabel='Retained rows (%)', title='Recorded historical XSum cleaning')
        ax.bar_label(bars, labels=[f"{v:.3f}%\n{stats[s]['processed_count']:,} rows" for s, v in zip(splits, values)], padding=5)
        save(fig, '02_cleaning_retention', 'Recorded cleaning report; does not represent 1,200-example curation eligibility.')

    comparison = reports / 'local_quality/direct_vs_review_validation_20.json'
    if comparison.exists():
        data = json.loads(comparison.read_text())
        rows = [r for r in data['examples'] if r['direct']['status'] == r['reviewed']['status'] == 'ok']
        if not rows: raise ValueError('No valid local comparison pairs available.')
        article_words = [len(r['article'].split()) for r in rows]
        direct = [r['direct']['words'] for r in rows]
        reviewed = [r['reviewed']['words'] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        axes[0].hist(article_words, bins=8, color=slate, edgecolor='white')
        axes[0].set(title='Article lengths', xlabel='Words', ylabel='Articles')
        axes[1].hist(direct, bins=range(min(direct)-1, max(direct)+3, 2), color=teal, edgecolor='white')
        axes[1].axvline(40, linestyle='--', color=amber, label='40-word target')
        axes[1].set(title='Direct summary lengths', xlabel='Words', ylabel='Summaries')
        axes[1].legend()
        save(fig, '03_validation_lengths', f'Local development comparison, n={len(rows)} validation articles; whitespace-delimited word counts.')
        reduction = [100 * (1 - s / a) for a, s in zip(article_words, direct)]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.hist(reduction, bins=8, color=teal, edgecolor='white')
        ax.axvline(statistics.median(reduction), color=slate, linestyle='--', label=f'Median: {statistics.median(reduction):.1f}%')
        ax.set(title='How much text was removed?', xlabel='Word reduction (%)', ylabel='Articles')
        ax.legend()
        save(fig, '04_word_reduction', 'Compression measures length only; a shorter summary is not necessarily more accurate or useful.')
        direct_seconds = [r['direct']['seconds'] for r in rows]
        total_seconds = [r['direct']['seconds'] + r['reviewed']['seconds'] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(1, len(rows)+1)
        ax.plot(x, direct_seconds, 'o-', color=teal, label=f'Direct: mean {statistics.mean(direct_seconds):.1f}s')
        ax.plot(x, total_seconds, 's-', color=slate, label=f'Direct + review: mean {statistics.mean(total_seconds):.1f}s')
        ax.set(title='Local generation latency', xlabel='Validation example', ylabel='Generation time (seconds)', xticks=x[::2])
        ax.legend()
        save(fig, '05_local_latency', 'Quadro RTX 4000 + CPU; one sequential run. Total review time includes the draft and review stages, not startup.')
        fig, ax = plt.subplots(figsize=(8, 4.8))
        counts = [sum(v <= 40 for v in direct), sum(v <= 40 for v in reviewed)]
        ax.bar(['Direct', 'Reviewed'], counts, color=teal, label='At most 40 words')
        bars = ax.bar(['Direct', 'Reviewed'], [len(rows)-v for v in counts], bottom=counts, color=amber, label='Over 40 words')
        for i,c in enumerate(counts):ax.text(i,c/2,f'{c}/{len(rows)}',ha='center',color='white',weight='bold')
        ax.set(title='Summary length compliance', ylabel='Summaries', ylim=(0,len(rows)+3))
        ax.legend(loc='upper right')
        save(fig, '06_length_compliance', 'A formatting constraint, not a factual-accuracy measure. Baseline and review use the same articles.')
    qualitative = reports / 'local_quality/assistant_review_20.json'
    if qualitative.exists():
        items = json.loads(qualitative.read_text())['examples']
        categories = [('no_clear_factual_error', 'No clear factual error flagged', teal),
                      ('clear_factual_error', 'Clear factual error', '#bd473d'),
                      ('borderline_or_wording_issue', 'Uncertainty / wording concern', amber)]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        values = [sum(r['assessment'] == key for r in items) for key,_,_ in categories]
        bars = ax.barh([label for _,label,_ in categories], values, color=[c for _,_,c in categories])
        ax.bar_label(bars, padding=4)
        ax.set(title='Qualitative source-based review', xlabel='Examples', xlim=(0,len(items)+1))
        save(fig, '07_assistant_quality_review', 'Assistant assessment of 20 development examples; not independent human annotation or a general accuracy estimate.')
    from .training_figures import build_training_figures
    history_note = build_training_figures(root, save)
    index = {'figures': generated, 'training_history': history_note,
             'provenance': 'All figures use existing reports. No synthetic training curves or accuracy numbers.'}
    (destination/'index.json').write_text(json.dumps(index,indent=2))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT)
    args=parser.parse_args()
    index=build_figures(args.root)
    print('Saved',len(index['figures']),'figures (PNG and SVG) to',args.root/'reports/figures')
    print(index['training_history'])


if __name__ == '__main__':main()
