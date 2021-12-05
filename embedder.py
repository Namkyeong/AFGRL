import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
from models import LogisticRegression
from utils import printConfig
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)

import os

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise

class embedder:
    def __init__(self, args):
        self.args = args
        self.hidden_layers = eval(args.layers)
        printConfig(args)

    def infer_embeddings(self, epoch):
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._dev_mask = self._test_mask = None
        for bc, batch_data in enumerate(self._loader):
            batch_data.to(self._device)
            emb, _, _, _ = self._model(x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                                                           neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                                                           edge_weight=batch_data.edge_attr, epoch=epoch)
            emb = emb.detach()
            y = batch_data.y.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])

    def evaluate(self, task, epoch):
        if task == "node":
            self.evaluate_node(epoch)
        elif task == "clustering":
            self.evaluate_clustering(epoch)
        elif task == "similarity":
            self.run_similarity_search(epoch)
        

    def evaluate_node(self, epoch):

        # print()
        # print("Evaluating ...")
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]

        dev_accs, test_accs = [], []

        for i in range(20):

            self._train_mask = self._dataset[0].train_mask[i]
            self._dev_mask = self._dataset[0].val_mask[i]
            if self._args.dataset == "wikics":
                self._test_mask = self._dataset[0].test_mask
            else:
                self._test_mask = self._dataset[0].test_mask[i]

            classifier = LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

            for _ in range(100):
                classifier.train()
                logits, loss = classifier(self._embeddings[self._train_mask], self._labels[self._train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            dev_logits, _ = classifier(self._embeddings[self._dev_mask], self._labels[self._dev_mask])
            test_logits, _ = classifier(self._embeddings[self._test_mask], self._labels[self._test_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() /
                       self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() /
                        self._labels[self._test_mask].shape[0]).detach().cpu().numpy()

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        print('** [{}] [Epoch: {}] Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(self.args.embedder, epoch, dev_acc, dev_std, test_acc, test_std))

        if dev_acc > self.best_dev_acc:
            self.best_dev_acc = dev_acc
            self.best_test_acc = test_acc
            self.best_dev_std = dev_std
            self.best_test_std = test_std
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best val | Best test: {:.4f} ({:.4f}) / {:.4f} ({:.4f})**\n'.format(
            self.best_epoch, self.best_dev_acc, self.best_dev_std, self.best_test_acc, self.best_test_std)
        print(self.st_best)


    def evaluate_clustering(self, epoch):
        
        embeddings = F.normalize(self._embeddings, dim = -1, p = 2).detach().cpu().numpy()
        nb_class = len(self._dataset[0].y.unique())
        true_y = self._dataset[0].y.detach().cpu().numpy()

        estimator = KMeans(n_clusters = nb_class)

        NMI_list = []

        for i in range(10):
            estimator.fit(embeddings)
            y_pred = estimator.predict(embeddings)

            s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
            NMI_list.append(s1)

        s1 = sum(NMI_list) / len(NMI_list)
        print('** [{}] [Current Epoch {}] Clustering NMI: {:.4f} **'.format(self.args.embedder, epoch, s1))

        if s1 > self.best_dev_acc:
            self.best_epoch = epoch
            self.best_dev_acc = s1
            if self._args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                torch.save(self._embeddings.detach().cpu(), os.path.join(self._args.checkpoint_dir, 'embeddings_{}_{}.pt'.format(self._args.dataset, self._args.task)))

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best NMI: {:.4f} **\n'.format(self.best_epoch, self.best_dev_acc)
        print(self.st_best)


    def run_similarity_search(self, epoch):

        test_embs = self._embeddings.detach().cpu().numpy()
        test_lbls = self._dataset[0].y.detach().cpu().numpy()
        numRows = test_embs.shape[0]

        cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
        st = []
        for N in [5, 10]:
            indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
            tmp = np.tile(test_lbls, (numRows, 1))
            selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
            original_label = np.repeat(test_lbls, N).reshape(numRows,N)
            st.append(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4))

        print("** [{}] [Current Epoch {}] sim@5 : {} | sim@10 : {} **".format(self.args.embedder, epoch, st[0], st[1]))

        if st[0] > self.best_dev_acc:
            self.best_dev_acc = st[0]
            self.best_test_acc = st[1]
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best @5 : {} | Best @10: {} **\n'.format(self.best_epoch, self.best_dev_acc, self.best_test_acc)
        print(self.st_best)

        return st


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        self.stacked_gnn = nn.ModuleList([GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList([nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight=edge_weight)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)

        return x
