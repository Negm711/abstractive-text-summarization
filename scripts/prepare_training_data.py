"""Build a frozen, source-grounded training/validation split using only reviewed targets.

Run from the project root: python prepare_training_data.py
This script uses the Python standard library; it does not load or train a model.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def choose_file(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'Missing reviewed file: {directory / pattern}')
    groups = {}
    for path in matches:
        contents = json.loads(path.read_text(encoding='utf-8-sig'))
        groups.setdefault(digest(contents), (path, contents))
    if len(groups) != 1:
        raise ValueError('Different versions found; keep the intended reviewed version in this folder: '
                         + ', '.join(str(p) for p in matches))
    return next(iter(groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    curation = root / 'data' / 'curation'
    sources = [choose_file(curation, 'training_candidates_200_adjudicated*.json')]
    for batch in range(1, 6):
        sources.append(choose_file(curation / 'expansion_1000_v1',
                                   f'curation_batch_{batch:02}_adjudicated*.json'))
    records, source_info = [], []
    seen_ids, seen_articles = set(), set()
    total = withheld = 0
    for path, contents in sources:
        rows = contents.get('examples')
        if not isinstance(rows, list) or len(rows) != 200:
            raise ValueError(f'{path.name}: expected a complete reviewed batch of 200 articles')
        eligible = 0
        for row in rows:
            total += 1
            if type(row.get('training_eligible')) is not bool:
                raise ValueError(f'{path.name}: missing explicit training eligibility')
            identity = str(row['id'])
            article = row.get('article')
            if not isinstance(article, str) or not article.strip():
                raise ValueError(f'{path.name}: empty article {identity}')
            normalized = ' '.join(re.findall(r'\w+', article.casefold()))
            if identity in seen_ids or normalized in seen_articles:
                raise ValueError(f'Duplicate article or ID across reviewed inputs: {identity}')
            seen_ids.add(identity)
            seen_articles.add(normalized)
            if not row['training_eligible']:
                withheld += 1
                continue
            summary = row.get('reviewed_summary')
            if not isinstance(summary, str) or not summary.strip() or len(summary.split()) > 40:
                raise ValueError(f'{path.name}: invalid reviewed target for {identity}')
            if row.get('review_status') not in ('accepted', 'revised'):
                raise ValueError(f'{path.name}: inconsistent review status for {identity}')
            records.append({'id': identity, 'article': article,
                            'reviewed_summary': summary, 'source_batch': path.name})
            eligible += 1
        source_info.append({'file': str(path.relative_to(root)), 'reviewed': len(rows),
                            'eligible': eligible, 'content_sha256': digest(contents)})
    # Source order cannot influence allocation; the seed and eligible set freeze the split.
    records.sort(key=lambda r: r['id'])
    random.Random(42).shuffle(records)
    validation_count = max(1, round(len(records) * 0.20))
    validation, training = records[:validation_count], records[validation_count:]
    assert not ({r['id'] for r in training} & {r['id'] for r in validation})
    if not training:
        raise ValueError('No training records after splitting')
    manifest = {
        'schema_version': 1, 'model': 'Qwen/Qwen3-14B',
        'base_revision': '40c069824f4251a91eefaf281ebe4c544efd3e18',
        'seed': 42, 'validation_fraction': 0.20,
        'reviewed': total, 'eligible': len(records), 'withheld': withheld,
        'train': len(training), 'validation': len(validation), 'sources': source_info,
        'target': 'reviewed_summary',
        'excluded_target_fields': ['original_summary', 'proposed_summary', 'pre_adjudication_summary',
                                   'source_evidence', 'review_note', 'adjudication'],
        'initialization_requirement': 'Start a new adapter from the pinned base model. Do not resume '
                                      'a pilot adapter; the new validation split may contain pilot training IDs.',
        'test_policy': 'No test data included. A separate, unused evaluation set must be frozen '
                       'before final evaluation; do not describe this train/validation split as a test set.',
        'provenance': 'AI-reviewed/source-checked targets; not guaranteed error-free or independently human reviewed.',
        'id_duplicates': 0, 'normalized_article_duplicates': 0,
        'semantic_overlap_not_certified': True,
        'eligible_records_sha256': digest(sorted(records, key=lambda r: r['id'])),
    }
    output = {'manifest': manifest, 'train': training, 'validation': validation}
    destination = curation / 'qwen3_curated_1200_v1' / 'training_data.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != output:
            raise FileExistsError(f'Existing split differs. It was not overwritten: {destination}')
        print('Existing frozen split verified.')
    else:
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(destination)
    print(f'Reviewed: {total} | Eligible: {len(records)} | Withheld: {withheld}')
    print(f'Train: {len(training)} | Validation: {len(validation)} | Seed: 42')
    print('Duplicate IDs/articles: 0 | Train-validation ID overlap: 0')
    print('Saved:', destination)
    print('No model download or training was started.')


if __name__ == '__main__':
    main()
