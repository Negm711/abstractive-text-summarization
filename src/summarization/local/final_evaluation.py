"""Freeze and evaluate a fresh local test sample without training or downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import urllib.error
from .settings import PROJECT_ROOT, ADAPTER_SHA, BASE_REVISION, SETTINGS
from .prompts import INSTRUCTION
from .service import LocalSummarizer, SummaryError
from .runtime import get_json

TRAINING_DATA_SHA = 'e9dd63b6893949a253f20102d48f5edab6e41096b7f266c0a18967dbe2e483b3'
DATASET_REVISION = '7d4d486c2f8ef850b1a11aead99b894ff3dd7da9'
SEED = 20261024
COUNT = 100


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def article_hash(text):
    normalized = ' '.join(re.findall(r'\w+', text.casefold()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def collect_seen(value, ids, articles):
    if isinstance(value, dict):
        for key in ('id', 'doc_id'):
            item = value.get(key)
            if isinstance(item, (str, int)):
                ids.add(str(item))
        for key in ('article', 'document'):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                articles.add(article_hash(text))
        if isinstance(value.get('ids'), list):
            ids.update(str(item) for item in value['ids'] if isinstance(item, (str, int)))
        for item in value.values():
            if isinstance(item, (dict, list)):
                collect_seen(item, ids, articles)
    elif isinstance(value, list):
        for item in value:
            collect_seen(item, ids, articles)


def choose(rows, denied_ids, denied_articles, count=COUNT):
    candidates, seen_ids, seen_articles = [], set(), set()
    excluded = {'previously_seen_or_curated': 0, 'empty': 0, 'duplicate': 0}
    for row in sorted(rows, key=lambda item: str(item['id'])):
        identity = str(row['id'])
        source, reference = row['document'], row['summary']
        if not isinstance(source, str) or not source.strip() or not isinstance(reference, str) or not reference.strip():
            excluded['empty'] += 1
            continue
        digest = article_hash(source)
        if identity in denied_ids or digest in denied_articles:
            excluded['previously_seen_or_curated'] += 1
            continue
        if identity in seen_ids or digest in seen_articles:
            excluded['duplicate'] += 1
            continue
        seen_ids.add(identity)
        seen_articles.add(digest)
        candidates.append({'id': identity, 'article': source, 'reference': reference, 'article_sha256': digest})
    candidates.sort(key=lambda row: hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest())
    if len(candidates) < count:
        raise ValueError('Not enough eligible unseen test records; no sample was written.')
    return candidates[:count], excluded


def paths(root):
    return root/'data/evaluation/qwen3_v2_final_100_v1/frozen_test_100.json', root/'reports/final_evaluation/results_100.json'


def prepare(root):
    frozen, _ = paths(root)
    if frozen.is_file():
        data = json.loads(frozen.read_text(encoding='utf-8'))
        if data['manifest']['training_data_sha256'] != TRAINING_DATA_SHA or len(data['examples']) != COUNT:
            raise ValueError('Existing frozen sample has an unexpected contract.')
        print('Frozen sample already exists; preserved:', frozen)
        print('Articles:', len(data['examples']), '| No model loaded.')
        return
    training = root/'data/curation/qwen3_curated_1200_v1/training_data.json'
    if not training.is_file() or file_hash(training) != TRAINING_DATA_SHA:
        raise ValueError('The verified frozen training split is missing or changed.')
    deny_path = Path(__file__).with_name('evaluation_exclusions.json')
    denied = json.loads(deny_path.read_text(encoding='utf-8'))
    ids, articles = set(denied['ids']), set(denied['article_hashes'])
    sources, unreadable = [], []
    # Read only local curation and report JSON. Do not traverse model weights or backups.
    for folder in (root/'data/curation', root/'reports'):
        for path in sorted(folder.rglob('*.json')):
            if path.stat().st_size > 50_000_000:
                unreadable.append(str(path.relative_to(root)))
                continue
            try:
                value = json.loads(path.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError) as error:
                unreadable.append(str(path.relative_to(root)))
                continue
            collect_seen(value, ids, articles)
            sources.append({'path': str(path.relative_to(root)), 'sha256': file_hash(path)})
    if unreadable:
        raise ValueError('Exclusion scan could not read these files: ' + ', '.join(unreadable))
    dataset_path = root/'data/processed/xsum'
    if not dataset_path.is_dir():
        raise FileNotFoundError('Local processed XSum is missing. No download requested: ' + str(dataset_path))
    cleaning = json.loads((root/'reports/cleaning_summary.json').read_text(encoding='utf-8'))
    if cleaning.get('metadata', {}).get('resolved_commit_sha') != DATASET_REVISION:
        raise ValueError('Processed dataset provenance does not match the pinned XSum revision.')
    from datasets import load_from_disk
    dataset = load_from_disk(str(dataset_path))
    examples, excluded = choose(dataset['test'], ids, articles)
    manifest = {'schema_version': 1, 'split': 'test', 'sample_count': COUNT, 'seed': SEED,
                'selection': 'Lowest SHA256(seed:id), after exact normalized-article and ID exclusions; no length or output-quality filtering.',
                'dataset_revision': DATASET_REVISION, 'processed_test_fingerprint': dataset['test']._fingerprint,
                'training_data_sha256': TRAINING_DATA_SHA, 'exclusion_snapshot_sha256': file_hash(deny_path),
                'exclusion_sources': sources, 'excluded': excluded, 'adapter_sha256': ADAPTER_SHA,
                'base_revision': BASE_REVISION, 'instruction': INSTRUCTION, 'policy': 'baseline',
                'scope': 'Held out from the recorded fine-tuning/development sets. Semantic duplicates and base-model pretraining exposure are not certified.'}
    write_json(frozen, {'manifest': manifest, 'examples': examples})
    print('Frozen final evaluation articles:', len(examples))
    print('Saved:', frozen)
    print('No training, generation or model download was started.')


def generate(root):
    frozen, output = paths(root)
    if not frozen.is_file():
        raise FileNotFoundError('Run prepare first.')
    data = json.loads(frozen.read_text(encoding='utf-8'))
    if len(data.get('examples', [])) != COUNT or len({row['id'] for row in data['examples']}) != COUNT:
        raise ValueError('Frozen sample must contain exactly 100 unique records.')
    if data['manifest']['instruction'] != INSTRUCTION or data['manifest']['adapter_sha256'] != ADAPTER_SHA:
        raise ValueError('Frozen inference policy differs from the current policy.')
    launch = json.loads((root/'reports/local_runtime/launch.json').read_text(encoding='utf-8'))
    if launch.get('policy') != 'baseline' or launch.get('instruction') != INSTRUCTION or launch.get('adapter', {}).get('adapter_sha256') != ADAPTER_SHA:
        raise ValueError('Start the verified baseline app before generating the final evaluation.')
    port = int(launch['command'][launch['command'].index('--port')+1])
    if get_json(f'http://127.0.0.1:{port}/health').get('status') != 'ok':
        raise RuntimeError('The local model is not ready.')
    contract = {'frozen_sample_sha256': file_hash(frozen), 'policy': 'baseline', 'instruction': INSTRUCTION,
                'adapter_sha256': ADAPTER_SHA, 'cache_prompt': False,
                'service_source_sha256': file_hash(Path(__file__).with_name('service.py')),
                'runtime_release': launch['release'], 'runtime_command': launch['command']}
    if output.is_file():
        result = json.loads(output.read_text(encoding='utf-8'))
        if result['contract'] != contract:
            raise ValueError('Saved results have a different evaluation contract; they were not overwritten.')
    else:
        result = {'contract': contract, 'examples': []}
    done = {row['id'] for row in result['examples']}
    client = LocalSummarizer(port=port, policy='baseline', cache_prompt=False)
    for index, row in enumerate(data['examples'], 1):
        if row['id'] in done:
            continue
        saved = dict(row)
        try:
            saved.update(status='ok', generated=client.summarize(row['article'], review=False))
        except SummaryError as error:
            saved.update(status='rejected', error=str(error))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            write_json(output, result)
            print('Connection interrupted; completed records saved. Rerun generate to resume.')
            raise
        saved['source_review'] = {'reviewer': None, 'factual_error': None, 'core_event_covered': None, 'clear_language': None, 'notes': None}
        result['examples'].append(saved)
        write_json(output, result)
        print(f"{index}/{COUNT}: {saved['status']} - saved", flush=True)
    print('Generation complete:', output)
    print('Source review and final metrics are next. Failed/rejected outputs remain in the denominator.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'generate'])
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    {'prepare': prepare, 'generate': generate}[args.stage](args.root.resolve())


if __name__ == '__main__':
    main()
