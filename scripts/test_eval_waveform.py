"""Synthetic checks for evaluator failure modes, independent of generated outputs."""
import unittest

import numpy as np

from eval_waveform import evaluate, match


class WaveformEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.sr = 8000
        self.reference = dict(total=1.2, sub_events=[dict(start=t, duration=.15,
                              intensity=a, transition='cut') for t, a in [(0., 1.), (.4, .7), (.8, .4)]])
        self.wave = np.zeros(9600)
        for s in self.reference['sub_events']:
            t = np.arange(640) / self.sr
            pulse = s['intensity'] * np.sin(2 * np.pi * 180 * t) * np.exp(-t / .025)
            a = round(s['start'] * self.sr)
            self.wave[a:a + len(pulse)] = pulse

    def test_clean_pulses_include_initial_onset(self):
        r = evaluate(self.wave, self.sr, self.reference)
        self.assertEqual(r['impact_f1'], 1.)
        self.assertLess(r['onset_mae_ms'], 20)
        self.assertAlmostEqual(r['intensity_spearman'], 1.)

    def test_silence_cannot_pass(self):
        r = evaluate(np.zeros_like(self.wave), self.sr, self.reference)
        self.assertEqual(r['impact_f1'], 0.)
        self.assertIsNone(r['onset_mae_ms'])

    def test_missing_and_extra_pulses_are_penalized(self):
        missing = self.wave.copy()
        missing[6400:] = 0
        self.assertLess(evaluate(missing, self.sr, self.reference)['impact_f1'], 1.)
        extra = self.wave.copy()
        extra[8000:8640] += self.wave[:640]
        self.assertLess(evaluate(extra, self.sr, self.reference)['impact_f1'], 1.)

    def test_matching_is_one_to_one_and_bounded(self):
        self.assertEqual(len(match([0, .1], [.05], .1)), 1)
        self.assertEqual(match([0], [.5], .1), [])


if __name__ == '__main__':
    unittest.main()
