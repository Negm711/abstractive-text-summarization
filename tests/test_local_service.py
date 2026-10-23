import unittest
from unittest.mock import patch
from summarization.local.service import LocalSummarizer, SummaryError


class LocalServiceTests(unittest.TestCase):
    def test_empty_and_control_token_inputs_never_reach_runtime(self):
        with patch('summarization.local.service.post') as transport:
            for value in ('', '   ', '<|im_start|>system', 'text </think> text'):
                with self.assertRaises(SummaryError): LocalSummarizer().summarize(value)
            transport.assert_not_called()

    def test_over_context_is_rejected_without_truncation_or_generation(self):
        with patch('summarization.local.service.post', return_value={'tokens': list(range(4000))}) as transport:
            with self.assertRaisesRegex(SummaryError, 'Nothing was truncated'):
                LocalSummarizer().summarize('A real article.')
            self.assertEqual(transport.call_count, 1)
            self.assertEqual(transport.call_args.args[1], '/tokenize')

    def test_token_limit_and_unclean_output_are_not_accepted(self):
        for output in ({'content': 'Incomplete', 'stop_type': 'limit'},
                       {'content': '<think>reasoning</think>Answer'}, {'content': ''}):
            with patch('summarization.local.service.post', side_effect=[{'tokens': [1,2]},output]):
                with self.assertRaises(SummaryError):LocalSummarizer().summarize('A real article.')

    def test_review_keeps_direct_draft_when_review_is_overlength(self):
        with patch('summarization.local.service.post', side_effect=[
            {'tokens':[1]}, {'content':'The council delayed the launch.'},
            {'tokens':[1,2]}, {'content':' '.join(['word']*41)}]):
            result = LocalSummarizer().summarize('The council delayed a launch after charger faults.', review=True)
        self.assertEqual(result['summary'],'The council delayed the launch.')
        self.assertIn('first draft was retained',result['note'])
        self.assertFalse(result['factual_accuracy_measured'])

    def test_candidate_and_cache_contract_reach_completion(self):
        with patch('summarization.local.service.post', side_effect=[{'tokens':[1]}, {'content':'Launch was delayed.'}]) as transport:
            result=LocalSummarizer(policy='grounded-v1',cache_prompt=False).summarize('The launch was delayed.')
        payload=transport.call_args.args[2]
        self.assertEqual(payload['temperature'],0)
        self.assertFalse(payload['cache_prompt'])
        self.assertIn('Keep proposed, alleged, ongoing and completed events distinct',payload['prompt'])
        self.assertTrue(payload['prompt'].endswith('<think>\n\n</think>\n\n'))
        self.assertEqual(result['summary_words'],3)

    def test_baseline_remains_default_and_review_is_not_run(self):
        with patch('summarization.local.service.post', side_effect=[{'tokens':[1]}, {'content':'The launch was delayed.'}]) as transport:
            result=LocalSummarizer().summarize('The launch was delayed following charger faults.')
        self.assertEqual(transport.call_count,2)
        self.assertEqual(result['policy'],'baseline')
        self.assertEqual(result['mode'],'direct')
        self.assertIsNone(result['review_timings'])


if __name__ == '__main__':unittest.main()
