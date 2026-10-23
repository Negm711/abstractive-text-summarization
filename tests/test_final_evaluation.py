import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from summarization.local import final_evaluation as evaluation


class FinalEvaluationTests(unittest.TestCase):
    def test_selection_is_order_independent_and_excludes_seen_and_duplicates(self):
        rows = [{'id': str(i), 'document': f'News item number {i}.', 'summary': 'News.'} for i in range(20)]
        rows.append({'id': 'duplicate', 'document': ' NEWS item number 3! ', 'summary': 'News.'})
        excluded_ids = {'2'}
        excluded_articles = {evaluation.article_hash(rows[4]['document'])}
        first, excluded = evaluation.choose(rows, excluded_ids, excluded_articles, count=10)
        second, _ = evaluation.choose(rows[:-1][::-1], excluded_ids, excluded_articles, count=10)
        self.assertEqual([r['id'] for r in first], [r['id'] for r in second])
        self.assertNotIn('2', {r['id'] for r in first})
        self.assertNotIn('4', {r['id'] for r in first})
        self.assertEqual(excluded['previously_seen_or_curated'], 2)
        self.assertEqual(excluded['duplicate'], 1)

    def test_disconnection_preserves_completed_records_and_resume_skips_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen, output = evaluation.paths(root)
            examples = [{'id': str(i), 'article': f'Article {i}', 'reference': 'Reference'} for i in range(3)]
            evaluation.write_json(frozen, {'manifest': {'instruction': evaluation.INSTRUCTION, 'adapter_sha256': evaluation.ADAPTER_SHA}, 'examples': examples})
            evaluation.write_json(root/'reports/local_runtime/launch.json', {
                'policy': 'baseline', 'instruction': evaluation.INSTRUCTION,
                'adapter': {'adapter_sha256': evaluation.ADAPTER_SHA}, 'command': ['server', '--port', '8087'], 'release': 'test'})
            good = {'summary': 'A summary.', 'summary_words': 2}
            with patch.object(evaluation, 'COUNT', 3), patch.object(evaluation, 'get_json', return_value={'status': 'ok'}), patch.object(evaluation, 'LocalSummarizer') as factory:
                factory.return_value.summarize.side_effect = [good, TimeoutError('Disconnected')]
                with self.assertRaises(TimeoutError):evaluation.generate(root)
                self.assertEqual(len(json.loads(output.read_text())['examples']), 1)
                factory.return_value.summarize.side_effect = [good, good]
                evaluation.generate(root)
                result = json.loads(output.read_text())['examples']
                self.assertEqual([r['id'] for r in result], ['0', '1', '2'])
                self.assertEqual(factory.return_value.summarize.call_count, 4)
