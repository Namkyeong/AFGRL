import numpy as np
import torch
import torch.nn.functional as F

from models import LogisticRegression
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
from sklearn import metrics


def evaluate_node(embeddings, dataset, name):

    labels = dataset.y
    emb_dim, num_class = embeddings.shape[1], dataset.y.unique().shape[0]

    dev_accs, test_accs = [], []

    for i in range(20):

        train_mask = dataset.train_mask[i]
        dev_mask = dataset.val_mask[i]
        if name == "wikics":
            test_mask = dataset.test_mask
        else:
            test_mask = dataset.test_mask[i]

        classifier = LogisticRegression(emb_dim, num_class)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

        for _ in range(100):
            classifier.train()
            logits, loss = classifier(embeddings[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_logits, _ = classifier(embeddings[dev_mask], labels[dev_mask])
        test_logits, _ = classifier(embeddings[test_mask], labels[test_mask])
        dev_preds = torch.argmax(dev_logits, dim=1)
        test_preds = torch.argmax(test_logits, dim=1)

        dev_acc = (torch.sum(dev_preds == labels[dev_mask]).float() /
                       labels[dev_mask].shape[0]).detach().cpu().numpy()
        test_acc = (torch.sum(test_preds == labels[test_mask]).float() /
                        labels[test_mask].shape[0]).detach().cpu().numpy()

        dev_accs.append(dev_acc * 100)
        test_accs.append(test_acc * 100)

    dev_accs = np.stack(dev_accs)
    test_accs = np.stack(test_accs)

    dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
    test_acc, test_std = test_accs.mean(), test_accs.std()

    print('Evaluate node classification results')
    print('** Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(dev_acc, dev_std, test_acc, test_std))

    
def evaluate_clustering(embeddings, dataset):
        
    embeddings = F.normalize(embeddings, dim = -1, p = 2).detach().cpu().numpy()
    nb_class = len(dataset.y.unique())
    true_y = dataset.y.detach().cpu().numpy()

    estimator = KMeans(n_clusters = nb_class)

    NMI_list = []
    h_list = []

    for i in range(10):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)
        
        h_score = metrics.homogeneity_score(true_y, y_pred)
        s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)
        h_list.append(h_score)

    s1 = sum(NMI_list) / len(NMI_list)
    h_score = sum(h_list) / len(h_list)
    print('Evaluate clustering results')
    print('** Clustering NMI: {:.4f} | homogeneity score: {:.4f} **'.format(s1, h_score))
    

def run_similarity_search(embeddings, dataset):

    test_embs = embeddings.detach().cpu().numpy()
    test_lbls = dataset.y.detach().cpu().numpy()
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4))

    print('Evaluate similarity search results')
    print("** sim@5 : {} | sim@10 : {} **".format(st[0], st[1]))