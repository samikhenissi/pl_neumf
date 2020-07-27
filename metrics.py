import math
import pandas as pd

"""
 from : https://github.com/yihong-chen/neural-collaborative-filtering
"""


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @property
    def subjects_explicit(self):
        return self._subjects_explicit


    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_pred, test_true = subjects[0], subjects[1], subjects[2], subjects[3]
        # the golden set
        full = pd.DataFrame({'user': test_users,
                             'item': test_items,
                             'test_score': test_true,
                             'test_pred': test_pred})
        full['test_interaction'] = (full['test_score']>0)*1
        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['test_pred'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_interaction'] == 1]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()


    def cal_ndcg_implicit(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_interaction'] == 1].copy()
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()



