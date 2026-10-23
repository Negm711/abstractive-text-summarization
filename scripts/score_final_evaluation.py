"""Reproduce final-test metrics from saved outputs; no inference is performed."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import statistics


def summarize(data, review=None):
    from rouge_score import rouge_scorer
    rows = data['examples']
    if len(rows) != 100 or len({row['id'] for row in rows}) != 100:
        raise ValueError('Expected the complete frozen 100-example evaluation.')
    good = [row for row in rows if row['status'] == 'ok']
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = {key: [] for key in ['rouge1', 'rouge2', 'rougeL']}
    for row in rows:
        prediction = row.get('generated', {}).get('summary', '') if row['status'] == 'ok' else ''
        for key, value in scorer.score(row['reference'], prediction).items():
            rouge[key].append(value.fmeasure)
    lengths = [len(row['generated']['summary'].split()) for row in good]
    times = sorted(row['generated']['total_seconds'] for row in good)
    sources = [len(row['article'].split()) for row in good]
    def percentile(values, probability):
        index = (len(values)-1)*probability
        lower = int(index)
        upper = min(lower+1, len(values)-1)
        return values[lower] + (values[upper]-values[lower])*(index-lower)
    contract = dict(data['contract'])
    command = list(contract['runtime_command'])
    command[0] = Path(command[0]).name
    for flag in ('-m', '--lora'):
        command[command.index(flag)+1] = Path(command[command.index(flag)+1]).name
    contract['runtime_command'] = command
    result = {'sample_count': len(rows), 'successful_outputs': len(good), 'rejected_or_failed': len(rows)-len(good),
              'contract': contract,
              'rouge_f1': {key: statistics.mean(values) for key, values in rouge.items()},
              'rouge_configuration': {'package': 'rouge-score==0.1.2', 'use_stemmer': True, 'aggregation': 'Macro mean F1, including empty predictions for rejected outputs; original XSum references.'},
              'length': {'mean_article_words': statistics.mean(sources), 'mean_summary_words': statistics.mean(lengths),
                         'median_summary_words': statistics.median(lengths), 'min_summary_words': min(lengths), 'max_summary_words': max(lengths),
                         'within_40_words': sum(n <= 40 for n in lengths), 'over_40_words': sum(n > 40 for n in lengths),
                         'mean_word_reduction_percent': statistics.mean(100*(1-b/a) for a,b in zip(sources,lengths))},
              'latency_seconds': {'mean': statistics.mean(times), 'median': statistics.median(times), 'p95': percentile(times,.95), 'min': min(times), 'max': max(times), 'sum': sum(times)},
              'interpretation': 'ROUGE, brevity and successful generation do not measure factual accuracy. Latency excludes startup, uses no prompt cache and describes this one workstation run.'}
    if review:
        assessed = review['examples']
        if len(assessed) != len(rows) or {r['id'] for r in assessed} != {r['id'] for r in rows}:
            raise ValueError('Source review does not match evaluation IDs.')
        lookup = {r['id']: r for r in assessed}
        if any(lookup[row['id']]['generated']['summary'] != row['generated']['summary'] or lookup[row['id']]['article'] != row['article'] for row in rows):
            raise ValueError('Source review differs from actual source/output pairs.')
        labels = ['no_clear_issue', 'wording_or_qualification_concern', 'unsupported_secondary_detail', 'material_factual_issue']
        result['assistant_review'] = {'counts': {label: sum(row['source_review']['finding'] == label for row in assessed) for label in labels},
                                      'method': review['review_method'], 'rubric': review['rubric'],
                                      'material_issue_sample_fraction': sum(row['source_review']['finding']=='material_factual_issue' for row in assessed)/len(rows),
                                      'note': 'Assistant assessment, not independent human labels or a calibrated population accuracy percentage.'}
    exclusions = Path(__file__).resolve().parents[1]/'src/summarization/local/evaluation_exclusions.json'
    if exclusions.is_file():
        known = json.loads(exclusions.read_text(encoding='utf-8'))
        ids, hashes = set(known['ids']), set(known['article_hashes'])
        def digest_article(article):
            return hashlib.sha256(' '.join(re.findall(r'\w+', article.casefold())).encode()).hexdigest()
        result['known_development_overlap'] = {
            'id_overlap': sum(row['id'] in ids for row in rows),
            'normalized_article_overlap': sum(digest_article(row['article']) in hashes for row in rows),
            'checked_against_ids': len(ids), 'checked_against_article_hashes': len(hashes),
            'scope': 'Known recorded development and curation examples only.'}
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root/'reports/final_evaluation/results_100.json')
    parser.add_argument('--review', type=Path, default=root/'reports/final_evaluation/source_review_100.json')
    parser.add_argument('--output', type=Path, default=root/'reports/metrics/final_test_100.json')
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding='utf-8')) if args.review.is_file() else None
    if review and review['input_file_sha256'] != hashlib.sha256(args.input.read_bytes()).hexdigest():
        raise ValueError('Source review belongs to a different result file.')
    metrics = summarize(json.loads(args.input.read_text(encoding='utf-8')), review)
    metrics['input_file_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('Metrics saved:', args.output)


if __name__ == '__main__':main()
