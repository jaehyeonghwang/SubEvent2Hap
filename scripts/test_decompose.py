"""Contract checks for staged decomposition, independent of an LM server."""
import copy
import json
import unittest
from unittest.mock import patch
import decompose as dc


class DecomposeTests(unittest.TestCase):
    def setUp(self):
        self.outline = {'phases': ['silence', 'contact'],
                        'events': [{'source': 'door', 'action': 'door is knocked once'}]}
        self.values = {'start': 2., 'duration': .3, 'intensity': .7, 'transition': 'crossfade'}
        self.timing = {'event_1': self.values}
        self.captions = {'action_1': 'A single knock on a wooden door.'}

    def responses(self, outline=None, timing=None, review=None, captions=None):
        outputs = [outline or self.outline, review or {'outline': self.outline, 'changes': []},
                   timing if timing is not None else self.timing,
                   captions if captions is not None else self.captions]
        return patch.object(dc, '_post', side_effect=[
            {'choices': [{'message': {'content': json.dumps(o)}}]} for o in outputs])

    def test_stages_preserve_values_and_supply_original_input(self):
        with self.responses() as post:
            result = dc.decompose('Contact at 2 seconds')
        self.assertEqual(post.call_count, 4)
        self.assertEqual([s['stage'] for s in result['stages']], ['events', 'review', 'timing', 'captions'])
        self.assertEqual(result['sub_events'], [{'source': 'door', 'prompt': self.captions['action_1'], **self.values}])
        for call in post.call_args_list[:3]:
            messages = call.args[1]['messages']
            self.assertEqual([m['role'] for m in messages], ['system', 'user'])
            self.assertEqual(json.loads(messages[1]['content'])['original_event'], 'Contact at 2 seconds')
        caption_context = json.loads(post.call_args_list[-1].args[1]['messages'][1]['content'])
        self.assertEqual(set(caption_context), {'actions'})

    def test_missing_or_unknown_keys_fail_instead_of_silent_join(self):
        for timing in ({}, {'event_99': self.values}):
            with self.subTest(timing=timing):
                with self.responses(timing=timing) as post:
                    with self.assertRaisesRegex(ValueError, 'timing stage'):
                        dc.decompose('Contact')
                self.assertEqual(post.call_count, 4)

    def test_range_error_is_named_after_one_contract_correction(self):
        timing = copy.deepcopy(self.timing)
        timing['event_1']['intensity'] = 70
        with self.responses(timing=timing) as post:
            with self.assertRaisesRegex(ValueError, 'timing stage'):
                dc.decompose('Contact')
        self.assertEqual(post.call_count, 4)

    def test_review_corrections_flow_to_later_stages(self):
        outline = copy.deepcopy(self.outline)
        outline['events'][0]['source'] = 'wooden door'
        with self.responses(review={'outline': outline, 'changes': ['Clarified source']}) as post:
            result = dc.decompose('Contact')
        self.assertEqual(result['sub_events'][0]['source'], 'wooden door')
        self.assertEqual(result['review_changes'], ['Clarified source'])
        context = json.loads(post.call_args_list[-1].args[1]['messages'][1]['content'])
        self.assertEqual(context['actions']['action_1']['source'], 'wooden door')

    def test_repeated_events_share_one_lm_action_caption(self):
        outline = copy.deepcopy(self.outline)
        outline['events'].append(copy.deepcopy(outline['events'][0]))
        timing = {**self.timing, 'event_2': {**self.values, 'start': 3.}}
        with self.responses(outline=outline, timing=timing,
                            review={'outline': outline, 'changes': []}):
            result = dc.decompose('Two knocks')
        self.assertEqual([s['start'] for s in result['sub_events']], [2., 3.])
        self.assertEqual(result['sub_events'][0]['prompt'], result['sub_events'][1]['prompt'])

    def test_missing_source_fails(self):
        outline = copy.deepcopy(self.outline)
        del outline['events'][0]['source']
        with self.responses(outline=outline) as post:
            with self.assertRaisesRegex(ValueError, 'events stage'):
                dc.decompose('Contact')
        self.assertEqual(post.call_count, 2)

    def test_contract_correction_is_lm_authored_and_logged(self):
        bad = {**self.values, 'intensity': 70}
        outputs = [bad, self.values]
        with patch.object(dc, '_post', side_effect=[
                {'choices': [{'message': {'content': json.dumps(o)}}]} for o in outputs]) as post:
            trace = []
            result = dc._stage('timing', 'Schedule', {}, dc.TimingValues,
                               dc.DEFAULT_MODEL, dc.LMSTUDIO_URL, .3, trace)
        self.assertEqual(result.intensity, .7)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(len(trace[0]['contract_corrections']), 1)


if __name__ == '__main__':
    unittest.main()
