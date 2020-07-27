import random
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

random.seed(42)

class UserItemRatingDataset(Dataset):
    """
    from: https://github.com/yihong-chen/neural-collaborative-filtering
    """

    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class UserItemRatingDataset_bpr(Dataset):
    """Wrapper, convert <user, positive item, negative item> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_positive_tensor, target_negative_tensor):
        self.user_tensor = user_tensor
        self.item_positive_tensor = item_positive_tensor
        self.target_negative_tensor = target_negative_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_positive_tensor[index], self.target_negative_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)




class SampleGenerator(object):

    def __init__(self, ratings,config):
        """
        Modified  from: https://github.com/yihong-chen/neural-collaborative-filtering
        Added batching for validation
        Optimized the negative sampling for both pointwise and pairwise loss
        """


        bpr = config['BPR_loss']
        self.batch_size = config['batch_size']
        self.ratings = ratings

        self.n_users = len(ratings['userId'].unique())
        self.n_items = len(ratings['itemId'].unique())

        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        self.ratings = self._binarize(self.ratings)

        self.train_ratings, self.test_ratings = self._split_loo(self.ratings)
        self.negatives = self._create_negative(self.train_ratings)

        if bpr == 1:
            self.train_ratings = self._augment_negative_train_bpr(self.negatives,config['augment_neg_number']*int(len(self.ratings)/self.n_users))
        else:
            augment_neg_number = config['augment_neg_number']
            self.train_ratings = self._augment_negative_train(self.train_ratings, augment_neg_number)

        self.test_ratings = self._augment_negative_test(self.test_ratings)


    @staticmethod
    def _split_loo(ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        print('train length is: ', len(train))
        print('test length is: ', len(test))
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    @staticmethod
    def _binarize(ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings['rating']= 1.0
        return ratings


    def _create_negative(self, ratings):
        """return all negative items interacted items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: list(self.item_pool - x))
        interact_status['interacted_items'] = interact_status['interacted_items'].apply(lambda x: list(x))

        return interact_status[['userId', 'negative_items','interacted_items']]

    def _augment_negative_train_bpr(self,interact_status,bootstrap_factor):
        """augmenting the bpr dataset by sampling (u,i,j) from user x positive_item x negative_item pool using bottstraping with repalce = True """

        print(' ----augmenting train data with negative items -------')
        interact_status['negative_sample'] = interact_status['negative_items'].apply(lambda x: np.random.choice(x,bootstrap_factor,replace = True))
        interact_status['positive_sample'] = interact_status['interacted_items'].apply(lambda x: np.random.choice(x,bootstrap_factor,replace = True))
        rating_init = interact_status.copy()
        rating = pd.DataFrame(columns = ['userId','sampled_positive_item','sampled_negative_item'])

        for k in range(bootstrap_factor):
            rating_init['sampled_negative_item'] = rating_init['negative_sample'].apply(lambda x: x[k])
            rating_init['sampled_positive_item'] = rating_init['positive_sample'].apply(lambda x: x[k])
            rating_negative = rating_init[['userId','sampled_negative_item','sampled_positive_item']].copy()
            rating = pd.concat([rating,rating_negative])
            del rating_negative
        print(' ----train augmentation finished, new train length is',len(rating))
        return rating.reset_index()


    def _augment_negative_train(self,rating,num_negatives):
        """augmenting  dataset with negative sampling """

        print(' ----augmenting train data with negative items -------')
        rating = pd.merge(rating, self.negatives[['userId', 'negative_items']], on='userId')
        rating['augment_negative'] = rating['negative_items'].apply(lambda x: random.sample(x,num_negatives))
        rating_init = rating.copy()
        for k in range(num_negatives):
            rating_init['sampled_negative_item'] = rating_init['augment_negative'].apply(lambda x: x[k])
            rating_negative = rating_init[['userId','sampled_negative_item']].copy()
            rating_negative['rating'] = 0
            rating_negative = rating_negative.rename(columns= {'sampled_negative_item':'itemId'})
            rating = pd.concat([rating,rating_negative])
            del rating_negative
        print(' ----train augmentation finished, new train length is',len(rating))
        return rating.drop(['negative_items','augment_negative'],axis = 1)



    def _augment_negative_test(self,rating):
        "Creating a 99 negative item for each user for testing"
        print(' ----augmenting test data with negative items -------')
        self.negatives['negative_samples'] = self.negatives['negative_items'].apply(lambda x: random.sample(x, 99))

        rating = pd.merge(rating, self.negatives[['userId', 'negative_samples']], on='userId')
        rating_init = rating.copy()

        for i in range(99):
            rating_init['sampled_negative_item'] = rating_init['negative_samples'].apply(lambda x: x[i])
            rating_negative = rating_init[['userId','sampled_negative_item']].copy()
            rating_negative['rating'] = 0
            rating_negative = rating_negative.rename(columns= {'sampled_negative_item':'itemId'})
            rating = pd.concat([rating,rating_negative])
            del rating_negative
        print(' ----test augmentation finished, new test length is',len(rating))
        return rating.drop('negative_samples',axis = 1)

    def instance_a_train_loader_bpr(self, batch_size):
        """instance train loader for one training epoch"""
        users, items_pos, items_neg = self.train_ratings['userId'].tolist(), self.train_ratings['sampled_positive_item'].tolist(), self.train_ratings['sampled_negative_item'].tolist()
        dataset = UserItemRatingDataset_bpr(user_tensor=torch.LongTensor(users),
                                        item_positive_tensor=torch.LongTensor(items_pos),
                                        target_negative_tensor=torch.LongTensor(items_neg))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = self.train_ratings['userId'].tolist(), self.train_ratings['itemId'].tolist(), self.train_ratings['rating'].tolist()
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


    @property
    def instance_val_loader(self):
        """create evaluate data"""
        test_users, test_items, test_output,  = self.test_ratings['userId'].tolist(), self.test_ratings['itemId'].tolist(),self.test_ratings['rating'].tolist()

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(test_users),
                                        item_tensor=torch.LongTensor(test_items),
                                        target_tensor=torch.FloatTensor(test_output))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


def preprocess_data(rating,config):
    """
    args:
        ratings: pd.DataFrame, which contains 4 columns = ['uid', 'mid', 'rating', 'timestamp']
        config: dict // name of the config dict.
    """
    assert 'uid' in rating.columns
    assert 'mid' in rating.columns
    assert 'rating' in rating.columns
    assert 'timestamp' in rating.columns

    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    item_id = rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    rating = pd.merge(rating, item_id, on=['mid'], how='left')
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))
    sample_generator = SampleGenerator(ratings=rating,config = config)

    print("created a generator object! ")
    return sample_generator

