import copy
import unittest
from summarization.eda import analyze


class CuratedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.data = {'train': [{'id': '1', 'article': 'A council delayed its bus launch.', 'reviewed_summary': 'Bus launch delayed.'}],
                     'validation': [{'id': '2', 'article': 'Airport traffic grew in March.', 'reviewed_summary': 'Airport traffic grew.'}]}

    def test_analysis_preserves_split_and_measures_word_counts(self):
        original = copy.deepcopy(self.data)
        result = analyze(self.data)
        self.assertEqual(result['splits']['train']['articles']['count'], 1)
        self.assertEqual(result['splits']['train']['articles']['mean'], 6)
        self.assertEqual(result['splits']['train']['mean_word_reduction_pct'], 50)
        self.assertEqual(self.data, original)

    def test_cross_split_overlap_is_rejected(self):
        self.data['validation'][0]['article'] = ' a COUNCIL delayed its bus launch. '
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            analyze(self.data)

    def test_empty_target_is_rejected(self):
        self.data['train'][0]['reviewed_summary'] = ''
        with self.assertRaisesRegex(ValueError, 'Invalid source/target'):
            analyze(self.data)
