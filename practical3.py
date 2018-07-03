import torch.tensor as Tensor
import torch as torch
import dlc_practical_prologue as prologue
import progressbar

# 1 - Activation Function
def sigma(x):
    return torch.tanh(x)


def dsigma(x):
    y = sigma(x)
    return 1 - torch.mul(y, y)


# 2 - Loss Function
def loss(v, t):
    y = v-t
    return torch.sum(torch.mul(y, y))


def dloss(v, t):
    return 2 * (v-t)


# 3 - Forward and Backward Passes
def forward_pass(w1, b1, w2, b2, x):
    x0 = x
    s1 = torch.mm(w1, x0) + b1
    x1 = sigma(s1)
    s2 = torch.mm(w2, x1) + b2
    x2 = sigma(s2)
    return x0, s1, x1, s2, x2


def backward_pass(w1, b1, w2, b2, t, x, s1, x1, s2, x2, dl_dw1, dl_db1, dl_dw2, dl_db2):
    dl_dx2 = dloss(x2, t)
    dl_ds2 = torch.mul(dl_dx2, dsigma(s2))
    dl_dw2 += torch.mm(dl_ds2, torch.t(x1))
    dl_db2 += dl_ds2
    dl_dx1 = torch.mm(torch.t(w2), dl_ds2)
    dl_ds1 = torch.mul(dl_dx1, dsigma(s1))
    dl_dw1 += torch.mm(dl_ds1, torch.t(x))
    dl_db1 += dl_ds1


def compute_error(test_set, test_target_set, w1, b1, w2, b2):
    n = test_set.shape[0]
    num_errors = 0
    for i in range(n):
        x = test_set[i].resize(test_set[i].shape[0], 1)
        t = torch.argmax(test_target_set[i])
        _, _, _, _, predicted = forward_pass(w1, b1, w2, b2, x)
        predicted_class = torch.argmax(predicted, 0)
        if t != predicted_class:
            num_errors += 1
    return num_errors/n

# 4 - Training the network
def train():
    mnist_train, mnist_train_target, mnist_test, mnist_test_target = \
        prologue.load_data(cifar=False, one_hot_labels=True, normalize=True)
    mnist_test_target *= 0.9
    mnist_train_target *= 0.9
    eps = 0.000001
    print(eps)
    w1 = torch.zeros(50, 784).normal_(0, eps)
    b1 = torch.zeros(50, 1).normal_(0, eps)
    w2 = torch.zeros(10, 50).normal_(0, eps)
    b2 = torch.zeros(10, 1).normal_(0, eps)
    eta = 0.1/mnist_train.shape[0]
    for i in progressbar.progressbar(range(1000)):
        dl_dw1 = torch.zeros(50, 784)
        dl_db1 = torch.zeros(50, 1)
        dl_dw2 = torch.zeros(10, 50)
        dl_db2 = torch.zeros(10, 1)
        for j in range(1000):
            x = mnist_train[j].resize_(mnist_train[j].shape[0], 1)
            t = mnist_train_target[j].resize_(mnist_train_target[j].shape[0], 1)
            x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, x)
            backward_pass(w1, b1, w2, b2, t, x, s1, x1, s2, x2, dl_dw1, dl_db1, dl_dw2, dl_db2)
        w1 = w1 - eta * dl_dw1
        w2 = w2 - eta * dl_dw2
        b1 = b1 - eta * dl_db1
        b2 = b2 - eta * dl_db2
    print("Training error: ", compute_error(mnist_train, mnist_train_target, w1, b1, w2, b2))
    print("Test error: ", compute_error(mnist_test, mnist_test_target, w1, b1, w2, b2))


train()