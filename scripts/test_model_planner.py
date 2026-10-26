"""LM owns selection; code validates contracts and links solver state."""
import unittest
from unittest.mock import patch
import model_planner as mp
from test_physical_models import setup


def apply(model):
    return mp.Decision(kind='apply', model=model, reason='LM selected', children=[])


def fallback():
    return mp.Decision(kind='fallback', model=None, reason='indivisible', children=[])


def child(description, entity='ball', link='none', **kw):
    return mp.Child(description=description, entity_key=entity, state_link=link,
                    connection=kw.get('connection','after_previous'), offset_s=kw.get('offset_s'))


def atom():
    return mp.AtomicEstimate(source='chair', action='sliding on the floor', start=0., duration=.5, intensity=.5)


def indivisible(value=True):
    return mp.Atomicity(contact_modes=['other'] * (1 if value else 2),
                        driving_modes=['other'] * (1 if value else 2),
                        regime_count=1 if value else 2,indivisible=value,reason='structure judgment')


def parts(route):
    return mp.StructureSplit(reason=route.reason,children=[mp.StructureChild(
        contact_mode='other',governing_conditions='test regime',
        description=c.description,entity_key=c.entity_key) for c in route.children])


class PlannerTests(unittest.TestCase):
    def test_atomicity_count_and_boolean_must_agree(self):
        with self.assertRaises(ValueError):
            mp.Atomicity(driving_modes=['gravity','passive','passive'],contact_modes=['free_flight','impact','rolling'],regime_count=3,indivisible=True,reason='inconsistent')
        with self.assertRaises(ValueError):
            mp.Atomicity(driving_modes=['passive'],contact_modes=['rolling'],regime_count=1,indivisible=False,reason='inconsistent')

    def test_lm_choice_is_executed_without_code_priority_override(self):
        with patch.object(mp.dc, '_stage', side_effect=[apply('single_impact'), setup('single_impact')]):
            r = mp.decompose('One contact')
        self.assertEqual(r['hierarchy']['tree']['model'], 'single_impact')
        self.assertEqual(r['hierarchy']['selection_owner'], 'LM')
        self.assertNotIn('priority', r['hierarchy']['catalog'][0])

    def test_model_covered_repetition_is_one_leaf(self):
        with patch.object(mp.dc, '_stage', side_effect=[apply('vertical_bounce'), setup('vertical_bounce',count=4)]):
            r = mp.decompose('Ball bounces four times')
        self.assertEqual(r['hierarchy']['tree']['children'], [])
        self.assertEqual(len(r['sub_events']), 4)

    def test_unknown_atomic_action_does_not_recurse(self):
        with patch.object(mp.dc, '_stage', side_effect=[fallback(),indivisible(),atom()]) as stage:
            r = mp.decompose('Unknown')
        self.assertEqual(stage.call_count, 3)
        self.assertEqual(len(r['sub_events']), 1)

    def test_flight_to_collision_transfers_speed_and_does_not_add_render_delay(self):
        route = mp.Decision(kind='split',model=None,reason='flight/contact',children=[child('Flight'),child('Contact',link='same_object')])
        with patch.object(mp.dc, '_stage', side_effect=[route,indivisible(False),parts(route),route,apply('free_fall'),setup('free_fall',height_m=1.,mass_kg=.2),
                mp.StateBinding(link='same_object',reason='continuous flight'),
                apply('single_impact'),setup('single_impact',restitution=.5)]):
            r = mp.decompose('Drop then hit')
        first, second = r['hierarchy']['tree']['children']
        self.assertEqual(first['sub_events'], [])
        self.assertAlmostEqual(r['sub_events'][0]['start'], (2/mp.pm.G)**.5)
        self.assertEqual(second['physics']['provenance']['normal_speed_m_s'],'inherited_state')
        self.assertAlmostEqual(second['physics']['parameters']['normal_speed_m_s'],(2*mp.pm.G)**.5)
        self.assertAlmostEqual(second['physics']['parameters']['mass_kg'],.2)

    def test_slide_to_normal_contact_uses_explicit_link_assumption(self):
        route=mp.Decision(kind='split',model=None,reason='motion/contact',children=[child('Slide'),child('Wall hit',link='motion_to_normal')])
        with patch.object(mp.dc,'_stage',side_effect=[route,indivisible(False),parts(route),route,apply('sliding'),setup('sliding',speed_m_s=1.,friction_coefficient=.1,duration_s=.2),
                mp.StateBinding(link='motion_to_normal',reason='perpendicular wall'),
                apply('single_impact'),setup('single_impact')]):
            r=mp.decompose('Slide then hit wall')
        leaf=r['hierarchy']['tree']['children'][1]
        self.assertAlmostEqual(leaf['physics']['parameters']['normal_speed_m_s'],1-.1*mp.pm.G*.2)
        self.assertIn('normal aligns',leaf['state_transfer']['assumption'])

    def test_wrong_entity_and_unmodeled_pause_cannot_inherit_state(self):
        with self.assertRaises(ValueError):
            mp.transfer_state({'entity_key':'ball','mass_kg':1.},'chair','same_object')
        with self.assertRaises(ValueError):
            child('after a pause',link='same_object',connection='after_pause',offset_s=1.)

    def test_split_review_can_remove_an_invented_pause(self):
        proposed=mp.Decision(kind='split',model=None,reason='two actions',children=[
            child('First',entity='a'),child('Second',entity='b',connection='after_pause',offset_s=.5)])
        corrected=mp.Decision(kind='split',model=None,reason='immediate sequence',children=[
            child('First',entity='a'),child('Second',entity='b')])
        with patch.object(mp.dc,'_stage',side_effect=[proposed,indivisible(False),parts(proposed),corrected,fallback(),indivisible(),atom(),fallback(),indivisible(),atom()]):
            r=mp.decompose('First action, immediately followed by second')
        self.assertEqual(r['sub_events'][1]['start'],.5)
        self.assertEqual(r['hierarchy']['tree']['children'][1]['placement']['connection'],'after_previous')

    def test_silent_root_is_valid_and_not_fabricated_vibration(self):
        with patch.object(mp.dc,'_stage',side_effect=[apply('free_fall'),setup('free_fall')]):
            r=mp.decompose('Falling')
        self.assertFalse(r['renderable'])
        self.assertEqual(r['sub_events'],[])
        self.assertGreater(r['hierarchy']['tree']['process_duration_s'],0)

    def test_rejected_model_is_reselected_by_lm_not_code(self):
        s=setup('single_impact'); s.supported=False
        with patch.object(mp.dc,'_stage',side_effect=[apply('single_impact'),s,indivisible(),fallback(),atom()]) as stage:
            r=mp.decompose('Unknown compound')
        self.assertEqual(r['hierarchy']['tree']['model'],'lm_atomic')
        self.assertEqual(stage.call_args_list[3].args[2]['excluded_models'],['single_impact'])

    def test_whole_model_rejection_checks_structure_before_trying_all_models(self):
        s=setup('vertical_bounce');s.supported=False
        split=mp.Decision(kind='split',model=None,reason='flight then unsupported rotation',children=[
            child('Flight',entity='a'),child('Unsupported rotation',entity='b')])
        with patch.object(mp.dc,'_stage',side_effect=[apply('vertical_bounce'),s,indivisible(False),parts(split),split,
                apply('free_fall'),setup('free_fall'),fallback(),indivisible(),atom()]) as stage:
            r=mp.decompose('Falls and rotates')
        tree=r['hierarchy']['tree']
        self.assertEqual(tree['model'],'composition')
        self.assertEqual(len(tree['model_rejections']),1)
        self.assertEqual([c['model'] for c in tree['children']],['free_fall','lm_atomic'])
        self.assertEqual(stage.call_args_list[2].args[0],'structure_review')

    def test_compound_fallback_is_reviewed_and_split(self):
        split=mp.Decision(kind='split',model=None,reason='two regimes',children=[
            child('First',entity='a'),child('Second',entity='b')])
        with patch.object(mp.dc,'_stage',side_effect=[fallback(),indivisible(False),parts(split),split,
                fallback(),indivisible(),atom(),fallback(),indivisible(),atom()]):
            r=mp.decompose('Compound unsupported action')
        self.assertEqual(r['hierarchy']['tree']['model'],'composition')
        self.assertEqual(len(r['sub_events']),2)

    def test_unavailable_requested_state_link_is_disclosed_without_fabricating_state(self):
        route=mp.Decision(kind='split',model=None,reason='two regimes',children=[
            child('First',entity='a',link='same_object'),child('Second',entity='a',link='same_object')])
        with patch.object(mp.dc,'_stage',side_effect=[route,indivisible(False),parts(route),route,
                fallback(),indivisible(),atom(),fallback(),indivisible(),atom()]):
            r=mp.decompose('Two unsupported regimes')
        for node in r['hierarchy']['tree']['children']:
            self.assertEqual(node['incoming_state'],{})
            self.assertEqual(node['state_transfer']['kind'],'none')
            self.assertEqual(node['state_transfer']['requested_link'],'same_object')
            self.assertIn('no computed state',node['state_transfer']['assumption'])

    def test_apply_cannot_also_split_and_depth_stop_is_labeled(self):
        with self.assertRaises(ValueError):
            mp.Decision(kind='apply',model='periodic',reason='x',children=[child('a')])
        with patch.object(mp.dc,'_stage',return_value=atom()):
            r=mp.decompose('Unknown',max_depth=0)
        self.assertEqual(r['hierarchy']['tree']['stop_reason'],'depth_limit')

    def test_invented_gap_is_contract_error(self):
        with self.assertRaises(ValueError):
            child('second action lasts 1s',offset_s=1.)

    def test_estimation_stage_only_when_enabled_and_values_carry_provenance(self):
        from test_physical_models import estimates
        s = setup('vertical_bounce', count=2)
        with patch.object(mp.dc, '_stage', side_effect=[apply('vertical_bounce'), s]) as stage:
            r = mp.decompose('Ball bounces twice')
        self.assertEqual(stage.call_count, 2)
        self.assertFalse(r['hierarchy']['parameter_estimation'])
        est = estimates('vertical_bounce', height_m=(.5, .3, .8), restitution=(.85, .75, .9), mass_kg=(.06, .04, .1))
        with patch.object(mp.dc, '_stage', side_effect=[apply('vertical_bounce'), s, est]) as stage:
            r = mp.decompose('Ball bounces twice', estimate_parameters=True)
        self.assertEqual(stage.call_args_list[2].args[0], 'estimate_vertical_bounce')
        physics = r['hierarchy']['tree']['physics']
        self.assertEqual(physics['parameters']['restitution'], .85)
        self.assertEqual(physics['provenance']['restitution'], 'estimated_by_lm')
        self.assertEqual(physics['provenance']['count'], 'text_extracted_by_lm')

    def test_omitted_contacts_surface_as_warnings(self):
        with patch.object(mp.dc, '_stage', side_effect=[apply('vertical_bounce'), setup('vertical_bounce', restitution=.8)]):
            r = mp.decompose('A ball is dropped')
        self.assertEqual(len(r['sub_events']), 8)
        self.assertEqual(r['warnings'][0]['count'], 3)
        with patch.object(mp.dc, '_stage', side_effect=[apply('vertical_bounce'), setup('vertical_bounce', count=2)]):
            self.assertEqual(mp.decompose('Ball bounces twice')['warnings'], [])

    def test_fully_stated_setup_skips_estimation_call(self):
        s = setup('single_impact', normal_speed_m_s=2., restitution=.5, mass_kg=.1)
        with patch.object(mp.dc, '_stage', side_effect=[apply('single_impact'), s]) as stage:
            mp.decompose('A 100 g puck hits a wall at 2 m/s, e=0.5', estimate_parameters=True)
        self.assertEqual(stage.call_count, 2)


if __name__ == '__main__':
    unittest.main()
