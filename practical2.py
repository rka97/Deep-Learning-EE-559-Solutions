import torch.tensor as Tensor
import torch as torch


# 1 - Nearest Neighbor
# train_input: 2d tensor of dim nxd
# train_target: 1d tensor of dim n
# x: 1d tensor of dim d
def nearest_classification(train_input, train_target, x):
    error = train_input - x
    error_norms = error.norm(2, 1)
    n = torch.argmin(error_norms)
    return train_target[n]


# 2 - Error estimation
# train_input: nxd
# train_target: n
# test_input: mxd
# test_target: m
def compute_nb_errors(train_input, train_target, test_input, test_target, mean=None, proj=None):
    if mean is not None:
        train_input -= mean
        test_input -= mean
    if proj is not None:
        train_input = torch.mm(train_input, proj.trn())
        test_input = torch.mm(test_input, proj.trn())
    num_errors = 0
    for i in range(test_input.shape[0]):
        x = test_input.narrow(0, i, 1)
        if nearest_classification(train_input, train_target, x) != test_target[i]:
            num_errors += 1
    return num_errors


# 3 - PCA
# x - n x d
def pca(x):
    mean = x.mean(0)
    y = x - mean
    e = torch.eig(torch.mm(y.t(), y), True)
    t = e[0].narrow(1,0,1).reshape((e[0].shape[0]))
    _, indices = torch.sort(t, descending=True)
    return mean, e[1][indices]


def embed_target(mean, basis, data, dim):
    useful_basis = basis.narrow(0, 0, dim)
    embedding = torch.mm(data - mean, useful_basis.t())
    reconstruction = torch.mm(embedding, useful_basis) + mean
    return embedding, reconstruction

def generate_dataset(size, dim, num_classes):
    T = torch.zeros(size, dim)
    T.normal_(0, 1)
    target = torch.zeros(size)
    for i in range(0, size, num_classes):
        T.narrow(0, i, num_classes).add_(i/num_classes)
        target.narrow(0, i, num_classes).fill_(i/num_classes)
    test_size = int(size)
    x = torch.zeros(test_size, dim).normal_(0, 1.25)
    x_target = torch.zeros(test_size)
    for i in range(0, size, num_classes):
        x.narrow(0, i, num_classes).add_(i/num_classes)
        x_target.narrow(0, i, num_classes).fill_(i/num_classes)
    return T, target, x, x_target


def test_nearest_classification(T, target, x, x_target, embedding_dim):
    non_pca_error = compute_nb_errors(T, target, x, x_target)
    train_mean, train_basis = pca(T)
    train_embedding, train_recon = embed_target(train_mean, train_basis, T, embedding_dim)
    test_embedding, test_recon = embed_target(train_mean, train_basis, x, embedding_dim)
    with_pca_error = compute_nb_errors(train_embedding, target, test_embedding, x_target)
    return non_pca_error, with_pca_error


T, target, x, x_target = generate_dataset(1000, 100, 5)
base_error, error_100 = test_nearest_classification(T, target, x, x_target, 100)
_, error_50 = test_nearest_classification(T, target, x, x_target, 50)
_, error_10 = test_nearest_classification(T, target, x, x_target, 10)
_, error_3 = test_nearest_classification(T, target, x, x_target, 3)
print("Test set size: %i samples" % 1000)
print("Number of Errors without PCA: \t \t %i" % base_error)
print("Number of Errors with 100d PCA: \t %i" % error_100)
print("Number of Errors with 50d PCA: \t \t %i" % error_50)
print("Number of Errors with 10d PCA: \t \t %i" % error_10)
print("Number of Errors with 3d PCA: \t \t %i" % error_3)
