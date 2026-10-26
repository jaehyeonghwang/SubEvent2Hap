"""Analytical mechanics and provenance checks; no LM/GPU required."""
import math
import unittest

import bounce as bp


def setup(**values):
    return bp.BounceSetup(supported=True, reason="vertical drop", ball_material="rubber", surface_material="wooden",
                         **{**dict(drop_height_m=None, restitution=None, mass_kg=None,
                                   release_time_s=None, impact_count=None), **values})


class BounceTests(unittest.TestCase):
    def test_drop_time_and_geometric_intervals(self):
        result = bp.simulate(setup(drop_height_m=1., restitution=.6, impact_count=4))
        p = result['physics']
        first = math.sqrt(2 / bp.GRAVITY)
        self.assertAlmostEqual(p['impacts'][0]['time_s'], first)
        self.assertAlmostEqual(p['inter_impact_intervals_s'][0], 2 * .6 * first)
        for a, b in zip(p['inter_impact_intervals_s'], p['inter_impact_intervals_s'][1:]):
            self.assertAlmostEqual(b / a, .6)
        self.assertEqual(len(result['sub_events']), 4)
        self.assertEqual(len({s['prompt'] for s in result['sub_events']}), 1)

    def test_impulse_units_decay_and_mass_invariance_of_time(self):
        a = bp.simulate(setup(drop_height_m=1., restitution=.5, mass_kg=.1, impact_count=3))['physics']
        b = bp.simulate(setup(drop_height_m=1., restitution=.5, mass_kg=.2, impact_count=3))['physics']
        for k, (x, y) in enumerate(zip(a['impacts'], b['impacts'])):
            self.assertEqual(x['time_s'], y['time_s'])
            self.assertAlmostEqual(y['impulse_N_s'], 2 * x['impulse_N_s'])
            self.assertAlmostEqual(x['relative_impulse'], .5 ** k)
        self.assertAlmostEqual(a['impacts'][0]['impulse_N_s'], .1 * 1.5 * math.sqrt(2 * bp.GRAVITY))

    def test_missing_inputs_are_labeled_and_overrides_win(self):
        r = bp.simulate(setup(drop_height_m=2.), bp.BounceOverrides(drop_height_m=.5))['physics']
        self.assertEqual(r['parameters']['drop_height_m'], .5)
        self.assertEqual(r['provenance']['drop_height_m'], 'user_override')
        self.assertEqual(r['provenance']['restitution'], 'assumed_default')
        self.assertEqual(r['setup']['drop_height_m'], 2.)

    def test_release_time_is_not_shifted(self):
        r = bp.simulate(setup(release_time_s=2., impact_count=1))
        self.assertAlmostEqual(r['sub_events'][0]['start'], 2 + math.sqrt(2 / bp.GRAVITY))

    def test_higher_drop_scales_flight_times(self):
        a = bp.simulate(setup(drop_height_m=1.))['physics']
        b = bp.simulate(setup(drop_height_m=4.))['physics']
        for x, y in zip(a['impacts'], b['impacts']):
            self.assertAlmostEqual(y['time_s'], 2 * x['time_s'])

    def test_unsupported_and_impossible_conditions_are_not_silently_repaired(self):
        s = setup()
        s.supported = False
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            bp.simulate(s)
        for s in (setup(restitution=0., impact_count=2), setup(impact_count=9),
                  setup(release_time_s=10.)):
            with self.assertRaises(ValueError):
                bp.simulate(s)

    def test_overlap_preserves_physics_and_is_disclosed(self):
        r = bp.simulate(setup(restitution=.3, impact_count=4))
        self.assertTrue(any('overlap' in n for n in r['physics']['notes']))
        gaps = r['physics']['inter_impact_intervals_s']
        self.assertLess(gaps[-1], .15)
        self.assertAlmostEqual(gaps[-1] / gaps[-2], .3)


if __name__ == '__main__':
    unittest.main()
