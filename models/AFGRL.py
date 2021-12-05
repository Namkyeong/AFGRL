from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
import sys
from torch import optim
from tensorboardX import SummaryWriter

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
from utils import EMA, set_requires_grad, init_weights, update_moving_average, loss_fn, repeat_1d_tensor, currentTime
import copy

from data import Dataset
from embedder import embedder
from utils import config2string
from embedder import Encoder
import faiss


class AFGRL_ModelTrainer(embedder):
    
    def __init__(self, args):
        embedder.__init__(self, args)
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))

    def _init(self):
        args = self._args
        self._task = args.task
        print("Downstream Task : {}".format(self._task))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        self._dataset = Dataset(root=args.root, dataset=args.dataset)
        self._loader = DataLoader(dataset=self._dataset)
        layers = [self._dataset.data.x.shape[1]] + self.hidden_layers
        self._model = AFGRL(layers, args).to(self._device)
        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay= 1e-5)

    def train(self):

        self.best_test_acc, self.best_dev_acc, self.best_test_std, self.best_dev_std, self.best_epoch = 0, 0, 0, 0, 0 
        self.best_dev_accs = []
        
        # get Random Initial accuracy
        self.infer_embeddings(0)
        print("initial accuracy ")
        self.evaluate(self._task, 0)

        f_final = open("results/{}.txt".format(self._args.embedder), "a")

        # Start Model Training
        print("Training Start!")
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):
                batch_data.to(self._device)
                _, loss, ind, k = self._model(x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                           neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                           edge_weight=batch_data.edge_attr, epoch=epoch)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()

                st = '[{}][Epoch {}/{}] Loss: {:.4f}'.format(currentTime(), epoch, self._args.epochs, loss.item())
                print(st)

            if (epoch) % 5 == 0:
                self.infer_embeddings(epoch)
                self.evaluate(self._task, epoch)

        print("\nTraining Done!")
        print("[Final] {}".format(self.st_best))

        f_final.write("{} -> {}\n".format(self.config_str, self.st_best))


class AFGRL(nn.Module):
    def __init__(self, layer_config, args, **kwargs):
        super().__init__()
        self.student_encoder = Encoder(layer_config=layer_config, dropout=args.dropout, **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.epochs)
        self.neighbor = Neighbor(args)

        rep_dim = layer_config[-1]

        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid), nn.PReLU(), nn.Linear(args.pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)

        self.topk = args.topk

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x, y, edge_index, neighbor, edge_weight=None, epoch=None):
        student = self.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pred = self.student_predictor(student)

        with torch.no_grad():
            teacher = self.teacher_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)

        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]), [x.shape[0], x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [x.shape[0], x.shape[0]])

        ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk, epoch)

        loss1 = loss_fn(pred[ind[0]], teacher[ind[1]].detach())
        loss2 = loss_fn(pred[ind[1]], teacher[ind[0]].detach())
        loss = loss1 + loss2

        return student, loss.mean(), ind, k


class Neighbor(nn.Module):
    def __init__(self, args):
        super(Neighbor, self).__init__()
        self.device = args.device
        self.num_centroids = args.num_centroids
        self.num_kmeans = args.num_kmeans
        self.clus_num_iters = args.clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def forward(self, adj, student, teacher, top_k, epoch):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []

        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)
        
            clust_labels = I_kmeans[:,0]

            pred_labels.append(clust_labels)

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long()

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn, I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality

        return pos_.coalesce()._indices(), I_knn.shape[1]

    def create_sparse(self, I):
        
        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])
        
        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result
