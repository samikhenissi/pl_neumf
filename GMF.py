import torch
##from https://github.com/yihong-chen/neural-collaborative-filtering/blob/master/src/gmf.py
import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import MetronAtK


class GMF(pl.LightningModule):
    """
    Architecture from: https://github.com/yihong-chen/neural-collaborative-filtering
    Input: User tensor and Item tensor for Cross-entropy loss //
           user tensor, positive item tensor and negative item tensor for BPR loss
    Output: predicted interactions in logits (toget probability use a sigmoid function on the output)
    """

    def __init__(self, config):
        super(GMF, self).__init__()
        self.config = config
        self._metron = MetronAtK(top_k= config['topk'])
        self.crit = torch.nn.BCEWithLogitsLoss()

        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.bpr = config['BPR_loss']
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        return logits

    def use_optimizer(self, config):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=config['adam_lr'],
                                     weight_decay=config['l2_regularization'])

        return optimizer


    def training_step(self,batch,batch_idx):

        if self.bpr ==1:
            user,items_pos,items_neg = batch[0],batch[1], batch[2]
            ratings_pos = self(user,items_pos)
            rating_neg = self(user,items_neg)
            xij = ratings_pos- rating_neg
            loss = self.logsigmoid(xij)
            loss = -loss.sum()
        else:
            user,items,ratings = batch[0],batch[1], batch[2]
            ratings_pred = self(user,items)
            loss = self.crit(ratings_pred.view(-1), ratings)

        return {'loss':loss}



    def validation_step(self,batch,batch_idx):
        test_users, test_items , test_true = batch[0],batch[1], batch[2]
        test_pred = self(test_users, test_items)
        loss = self.crit(test_pred.view(-1), test_true)
        return {'val_loss':loss,'test_users': test_users.detach(),'test_items':test_items.detach(),
                'test_pred':test_pred.sigmoid().detach(),'test_true':test_true.detach() }

    def configure_optimizers(self):
        optimizer = self.use_optimizer(self.config)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.2, min_lr=1e-8)
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        test_users = torch.cat([x['test_users'] for x in outputs])
        test_items = torch.cat([x['test_items'] for x in outputs])
        test_pred = torch.cat([x['test_pred'] for x in outputs])
        test_true = torch.cat([x['test_true'] for x in outputs])

        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_pred.data.view(-1).tolist(),
                                 test_true.data.view(-1).tolist(),]


        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg_implicit()
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(self.current_epoch, hit_ratio, ndcg))
        log = {'val_loss':avg_loss,'HR':hit_ratio,'NDCG':ndcg}

        return {'log':log,'val_loss':ndcg,'HR':hit_ratio}


    def init_weight(self):
        pass


