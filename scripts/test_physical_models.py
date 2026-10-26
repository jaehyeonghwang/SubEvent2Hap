"""Analytical invariants for every model and signal control."""
import json
import math
import unittest
import numpy as np
import physical_models as pm
import signal_controls as sc


def setup(name, **values):
    fields = {k: None for k in pm.REGISTRY[name].parameters}
    if name in ('sliding','rolling'):
        fields['motion_mode'] = 'coasting'
    fields.update(values)
    return pm.setup_type(name)(supported=True, reason='within scope', object_name='test object',
                               surface_name='test surface', **fields)


class ModelPhysicsTests(unittest.TestCase):
    def test_all_model_families_are_executable(self):
        for spec in pm.MODELS:
            with self.subTest(model=spec.name):
                r = pm.solve(spec.name, setup(spec.name))
                self.assertLessEqual(r['process_duration_s'], 10)
                self.assertIn('provenance', r['physics'])

    def test_free_fall_is_silent_and_conserves_energy(self):
        r = pm.solve('free_fall', setup('free_fall', height_m=2., mass_kg=.3))
        self.assertEqual(r['sub_events'], [])
        self.assertAlmostEqual(r['process_duration_s'], math.sqrt(4 / pm.G))
        self.assertAlmostEqual(r['terminal_state']['energy_J'], .3 * pm.G * 2)

    def test_free_fall_inherits_upward_rebound(self):
        r = pm.solve('free_fall', setup('free_fall'), {'height_m':0., 'vertical_velocity_m_s':2.})
        self.assertAlmostEqual(r['process_duration_s'], 4 / pm.G)
        self.assertAlmostEqual(r['terminal_state']['vertical_velocity_m_s'], -2.)

    def test_bounce_interval_ratio_and_contact_count(self):
        r = pm.solve('vertical_bounce', setup('vertical_bounce', restitution=.6, count=4))
        times = [s['start'] for s in r['sub_events']]
        self.assertEqual(len(times), 4)
        self.assertAlmostEqual((times[2]-times[1])/(times[1]-times[0]), .6)
        self.assertLess(r['process_duration_s'], max(s['start']+s['duration'] for s in r['sub_events']))

    def test_unspecified_count_bounces_until_impulse_threshold(self):
        r = pm.solve('vertical_bounce', setup('vertical_bounce', restitution=.7))
        ints = [s['intensity'] for s in r['sub_events']]
        self.assertEqual(len(ints), 7)  # .7^6 = .118 >= .1 > .7^7
        self.assertGreaterEqual(min(ints), pm.BOUNCE_STOP_RATIO)
        self.assertEqual(r['physics']['provenance']['count'], 'derived_until_threshold')
        self.assertIsNone(r['physics']['omitted'])

    def test_contacts_beyond_render_limit_are_reported_not_output(self):
        r = pm.solve('vertical_bounce', setup('vertical_bounce', restitution=.8))
        self.assertEqual(len(r['sub_events']), 8)
        self.assertEqual(r['physics']['parameters']['physical_contact_count'], 11)
        self.assertEqual(r['physics']['omitted']['count'], 3)
        self.assertIn('render limit', r['physics']['omitted']['reason'])

    def test_contacts_beyond_ten_seconds_are_reported_not_output(self):
        r = pm.solve('vertical_bounce', setup('vertical_bounce', height_m=100., restitution=.5))
        self.assertEqual(len(r['sub_events']), 2)
        self.assertEqual(r['physics']['omitted']['count'], 2)
        self.assertIn('10-second', r['physics']['omitted']['reason'])

    def test_restitution_edge_cases(self):
        self.assertEqual(len(pm.solve('vertical_bounce', setup('vertical_bounce', restitution=0.))['sub_events']), 1)
        r = pm.solve('vertical_bounce', setup('vertical_bounce', restitution=1.))
        self.assertEqual(len(r['sub_events']), 8)
        self.assertIsNone(r['physics']['omitted']['count'])

    def test_stated_count_is_not_overridden_by_threshold(self):
        r = pm.solve('vertical_bounce', setup('vertical_bounce', restitution=.9, count=2))
        self.assertEqual(len(r['sub_events']), 2)
        self.assertEqual(r['physics']['provenance']['count'], 'text_extracted_by_lm')
        with self.assertRaises(ValueError):
            pm.solve('vertical_bounce', setup('vertical_bounce', count=12))

    def test_impact_inherits_mass_and_normal_speed(self):
        r = pm.solve('single_impact', setup('single_impact', restitution=.5),
                     {'mass_kg':2., 'normal_speed_m_s':3., 'vertical_velocity_m_s':-3.})
        self.assertAlmostEqual(r['physics']['samples'][0]['impulse_N_s'], 9.)
        self.assertEqual(r['physics']['provenance']['normal_speed_m_s'], 'inherited_state')
        self.assertAlmostEqual(r['terminal_state']['vertical_velocity_m_s'], 1.5)

    def test_explicit_input_overrides_inherited_state(self):
        r = pm.solve('single_impact', setup('single_impact', normal_speed_m_s=2.), {'normal_speed_m_s':5.})
        self.assertEqual(r['physics']['parameters']['normal_speed_m_s'], 2.)
        self.assertEqual(r['physics']['provenance']['normal_speed_m_s'], 'text_extracted_by_lm')

    def test_zero_speed_contact_does_not_invent_vibration(self):
        r=pm.solve('single_impact',setup('single_impact',normal_speed_m_s=0.))
        self.assertEqual(r['sub_events'],[])
        self.assertEqual(r['terminal_state']['impulse_N_s'],0.)

    def test_sliding_stop_distance_and_duration(self):
        r = pm.solve('sliding', setup('sliding', speed_m_s=2., friction_coefficient=.2))
        self.assertAlmostEqual(r['process_duration_s'], 2/(.2*pm.G))
        self.assertAlmostEqual(r['terminal_state']['distance_m'], 4/(2*.2*pm.G))
        self.assertEqual(r['terminal_state']['horizontal_speed_m_s'], 0.)
        self.assertEqual(r['sub_events'][0]['envelope']['end_ratio'], 0.)

    def test_friction_motion_stops_but_explicit_duration_remains(self):
        r = pm.solve('sliding', setup('sliding', speed_m_s=.2, duration_s=1.))
        self.assertEqual(r['process_duration_s'], 1.)
        self.assertEqual(r['terminal_state']['horizontal_speed_m_s'], 0.)

    def test_rolling_uses_rolling_resistance(self):
        r = pm.solve('rolling', setup('rolling', speed_m_s=1., rolling_resistance=.05))
        self.assertAlmostEqual(r['process_duration_s'], 1/(.05*pm.G))
        self.assertIn('rolling_resistance', r['physics']['parameters'])

    def test_maintained_drag_does_not_spontaneously_stop(self):
        r=pm.solve('sliding',setup('sliding',motion_mode='constant_speed',speed_m_s=1.,duration_s=2.,friction_coefficient=.2,mass_kg=.5))
        self.assertEqual(r['process_duration_s'],2.)
        self.assertEqual(r['terminal_state']['horizontal_speed_m_s'],1.)
        self.assertEqual(r['terminal_state']['distance_m'],2.)
        self.assertAlmostEqual(r['physics']['parameters']['required_drive_force_N'],.2*.5*pm.G)

    def test_damping_time_constant_and_energy_decay(self):
        r = pm.solve('damped_oscillation', setup('damped_oscillation', frequency_hz=20., damping_ratio=.1, duration_s=.2), {'vibration_energy_J':2.})
        tau = 1/(.1*2*math.pi*20)
        self.assertAlmostEqual(r['sub_events'][0]['envelope']['tau_s'], tau)
        self.assertAlmostEqual(r['terminal_state']['vibration_energy_J'], 2*math.exp(-2*.2/tau))

    def test_translation_is_not_silently_converted_to_ring_energy(self):
        r = pm.solve('damped_oscillation', setup('damped_oscillation'), {'energy_J':2.})
        self.assertNotIn('vibration_energy_J',r['terminal_state'])

    def test_periodic_rpm_mapping_and_phase_transfer(self):
        r = pm.solve('periodic', setup('periodic', rpm=600., duration_s=.25), {'phase_rad':.3})
        self.assertEqual(r['sub_events'][0]['synthesis']['frequency_hz'], 10.)
        self.assertAlmostEqual(r['terminal_state']['phase_rad'], (.3+2*math.pi*10*.25)%(2*math.pi))

    def test_poisson_seed_is_reproducible_and_time_bounded(self):
        s = setup('stochastic', seed=42, rate_hz=2., duration_s=2.)
        a, b = pm.solve('stochastic', s), pm.solve('stochastic', s)
        self.assertEqual(a['sub_events'], b['sub_events'])
        self.assertTrue(all(0 <= x['start'] < 2 for x in a['sub_events']))
        fixed = pm.solve('stochastic', setup('stochastic', seed=42, count=3))
        self.assertEqual(len(fixed['sub_events']), 3)

    def test_intentional_rhythm_uses_tempo_not_restitution(self):
        r = pm.solve('rhythmic_action', setup('rhythmic_action', count=3, tempo_bpm=120.))
        self.assertEqual([s['start'] for s in r['sub_events']], [0., .5, 1.])
        self.assertEqual([s['intensity'] for s in r['sub_events']], [1., 1., 1.])

    def test_renderer_overflow_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            pm.solve('rhythmic_action', setup('rhythmic_action', count=9))
        with self.assertRaises(ValueError):
            pm.solve('periodic', setup('periodic', start_s=9., duration_s=2.))

    def test_unsupported_scope_is_not_silently_solved(self):
        s = setup('sliding')
        s.supported = False
        with self.assertRaises(pm.UnsupportedModel):
            pm.solve('sliding', s)


def estimates(name, **values):
    """values: key -> (value, low, high) or None for 'no basis'."""
    keys = list(values)
    data = {k: (dict(value=v[0], low=v[1], high=v[2], basis='test basis') if v else
                dict(value=None, low=None, high=None, basis='no basis in description'))
            for k, v in values.items()}
    return pm.estimate_type(name, keys).model_validate_json(json.dumps(data))  # same path as LM output


class EstimationTests(unittest.TestCase):
    def test_only_unstated_uninherited_properties_are_requested(self):
        s = setup('vertical_bounce', restitution=.6, count=3)
        self.assertEqual(pm.missing_estimable('vertical_bounce', s), ['height_m', 'mass_kg'])
        self.assertEqual(pm.missing_estimable('vertical_bounce', s, {'mass_kg': .2}), ['height_m'])
        self.assertNotIn('count', pm.missing_estimable('rhythmic_action', setup('rhythmic_action')))
        self.assertNotIn('start_s', pm.missing_estimable('single_impact', setup('single_impact')))

    def test_precedence_stated_inherited_estimated_default(self):
        est = estimates('single_impact', normal_speed_m_s=(3., 2., 4.), restitution=(.5, .4, .6), mass_kg=(1., .5, 2.))
        r = pm.solve('single_impact', setup('single_impact', restitution=.9), {'mass_kg': .3}, est)
        p, o = r['physics']['parameters'], r['physics']['provenance']
        self.assertEqual((p['restitution'], o['restitution']), (.9, 'text_extracted_by_lm'))
        self.assertEqual((p['mass_kg'], o['mass_kg']), (.3, 'inherited_state'))
        self.assertEqual((p['normal_speed_m_s'], o['normal_speed_m_s']), (3., 'estimated_by_lm'))
        self.assertEqual(r['physics']['estimates']['normal_speed_m_s']['low'], 2.)

    def test_null_estimate_falls_back_to_labeled_default(self):
        est = estimates('vertical_bounce', height_m=None, restitution=(.8, .7, .9), mass_kg=None)
        r = pm.solve('vertical_bounce', setup('vertical_bounce', count=3), None, est)
        o = r['physics']['provenance']
        self.assertEqual(o['height_m'], 'assumed_default')
        self.assertEqual(o['restitution'], 'estimated_by_lm')
        ints = [s['intensity'] for s in r['sub_events']]
        self.assertAlmostEqual(ints[1] / ints[0], .8)

    def test_incoherent_or_out_of_bounds_estimates_are_contract_errors(self):
        t = pm.estimate_type('vertical_bounce', ['restitution'])
        cls = t.model_fields['restitution'].annotation
        with self.assertRaises(ValueError):
            cls(value=.9, low=.1, high=.5, basis='value outside its own range')
        with self.assertRaises(ValueError):
            cls(value=.5, low=None, high=None, basis='partial range')
        with self.assertRaises(ValueError):
            cls(value=1.4, low=1.2, high=1.6, basis='restitution above 1')


class SignalControlTests(unittest.TestCase):
    def test_sine_frequency_and_continuous_phase(self):
        a = sc.carrier({'kind':'sine','frequency_hz':100.,'phase_rad':.3}, 1000, 8000)
        phase = (.3 + 2*math.pi*100*1000/8000) % (2*math.pi)
        b = sc.carrier({'kind':'sine','frequency_hz':100.,'phase_rad':phase}, 1000, 8000)
        full = sc.carrier({'kind':'sine','frequency_hz':100.,'phase_rad':.3}, 2000, 8000)
        np.testing.assert_allclose(np.concatenate([a,b]), full, atol=1e-6)

    def test_envelopes_are_actual_gain_controls(self):
        x = sc.envelope({'kind':'exponential','tau_s':.1}, 1600, 8000)
        self.assertAlmostEqual(float(x[800]), math.exp(-1), places=6)
        y = sc.envelope({'kind':'linear','end_ratio':0.}, 10, 8000)
        self.assertEqual(float(y[-1]), 0.)

    def test_invalid_control_is_rejected(self):
        with self.assertRaises(ValueError):
            sc.carrier({'kind':'sine','frequency_hz':4000.}, 10, 8000)
        with self.assertRaises(ValueError):
            sc.envelope({'kind':'exponential','tau_s':0.}, 10, 8000)


if __name__ == '__main__':
    unittest.main()
