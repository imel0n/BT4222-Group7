"""Run: python -m unittest discover -s tests -p 'test_step8*.py' -v"""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

# The notebook is standalone in Colab; test its actual helper cell rather than a copy.
here = Path(__file__).resolve().parent
notebook = here / 'step8_feature_validation.ipynb'
if not notebook.exists():
    notebook = here.parent / 'step8_feature_validation.ipynb'
nb = json.loads(notebook.read_text())
source = next(''.join(c['source']) for c in nb['cells']
              if c['cell_type'] == 'code' and 'def click_context(' in ''.join(c['source']))
ns = {}
exec(compile(source, str(notebook), 'exec'), ns)

class FeatureValidationTests(unittest.TestCase):
    def test_click_window_context_alignment_and_no_outcomes(self):
        events = pd.DataFrame({'user':[0,0,0,0,0,1], 'wid':[5,5,5,5,5,5],
            'type':[0,0,0,0,2,0], 'age_days':[.5,1.,6.9,7.,.1,.2], 'qry':[0,-1,-1,0,0,0]})
        c = pd.DataFrame({'user':[1,0,0], 'wid':[5,6,5]}, index=[7,8,9])
        out = ns['click_context'](events, c)
        self.assertEqual(out.index.tolist(), [7,8,9])
        self.assertEqual(out.iloc[2].tolist(), [1,0,1,2])
        self.assertEqual(out.iloc[0].tolist(), [1,0,1,0])
        self.assertTrue((out.iloc[1] == 0).all())
        c['label'] = [3,2,1]
        pd.testing.assert_frame_equal(out, ns['click_context'](events, c))

    def test_metrics_purchase_denominator_and_all_zero_slate(self):
        d = pd.DataFrame({'user':[0,0,1,1,2], 'wid':[1,2,1,2,1], 'label':[3,0,1,0,0]})
        m = ns['slate_metrics'](d, [0,1,1,0,1], k=1)
        self.assertEqual(m.loc[0,'ndcg10'], 0)
        self.assertEqual(m.loc[1,'ndcg10'], 1)
        self.assertEqual(m.loc[0,'purchase_hr10'], 0)
        self.assertTrue(np.isnan(m.loc[1,'purchase_hr10']))
        self.assertTrue(m.loc[2].isna().all())

    def test_tie_results_invariant_to_candidate_order(self):
        d = pd.DataFrame({'user':[0]*20, 'wid':np.arange(20), 'label':[3]+[0]*19})
        original = ns['slate_metrics'](d, np.zeros(20))
        shuffled = ns['slate_metrics'](d.sample(frac=1, random_state=9), np.zeros(20))
        pd.testing.assert_frame_equal(original, shuffled)

    def test_ndcg_matches_hand_calculation(self):
        d = pd.DataFrame({'user':[0]*3, 'wid':[1,2,3], 'label':[3,2,0]})
        m = ns['slate_metrics'](d, [2,3,1], k=2)
        expected = (3 + 7 / np.log2(3)) / (7 + 3 / np.log2(3))
        self.assertAlmostEqual(m.loc[0,'ndcg10'], expected)

    def test_contract_rejects_test_data_and_cross_split_slates(self):
        d = pd.DataFrame({'user':[0,0], 'wid':[1,2], 'label':[1,0],
                          'split':['train','train'], 'u_click_7d':[1,1]})
        ns['validate_table'](d)
        for splits in [['train','val'], ['test','test']]:
            with self.assertRaises(AssertionError):
                ns['validate_table'](d.assign(split=splits))
        with self.assertRaises(AssertionError):
            ns['validate_table'](d.assign(wid=[1,1]))

    def test_popularity_ablation_removes_derived_rank(self):
        groups = ns['feature_groups'](['i_users_ord_365d','slate_pct_pop','uc_click_7d',
                                      'u_queryless_share_7d','q_name_tfidf_sim'])
        self.assertEqual(groups['popularity'], ['i_users_ord_365d','slate_pct_pop'])

    def test_paired_interval_zero_and_empty(self):
        ci = ns['paired_interval'](np.zeros(8), draws=50)
        self.assertEqual((ci['delta'],ci['lower95'],ci['upper95']), (0,0,0))
        self.assertEqual(ns['paired_interval']([], draws=50)['slates'], 0)

if __name__ == '__main__':
    unittest.main()
