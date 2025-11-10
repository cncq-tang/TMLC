import copy
from utils.data import *
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from model import MultiViewDataset, RCML
from loss_function import get_loss
from utils.pseudo import uncertainty_SMOTE


def imblance(dataset_name, ratio, args):
    X, Y, num_sample, num_views, num_classes, dims = read_mymat('./datasets/', dataset_name, ['X', 'Y'])
    print("dataset_name:", dataset_name)
    print("num samples: ", num_sample)
    print("num views: ", num_views)
    print("dims:", dims)

    partition = build_imbalanced_dataset(Y, ratio, num_classes, seed=999)
    train_index = partition['train']
    test_index = partition['test']
    print("the number of train data:", len(train_index))
    print("the number of test data:", len(test_index))
    X_train = [X[v][train_index] for v in range(num_views)]
    Y_train = Y[train_index]
    X_test = [X[v][test_index] for v in range(num_views)]
    Y_test = Y[test_index]

    for v in range(num_views):
        X[v] = normalize(X[v])
        X_train[v] = X[v][train_index]
        X_test[v] = X[v][test_index]

    train_loader = DataLoader(MultiViewDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(MultiViewDataset(X_test, Y_test), batch_size=args.batch_size, shuffle=False)

    device = args.device
    model = RCML(num_views, dims, num_classes)
    model.to(device)

    print('--------------Pre-Train------------------------')
    optimizer_pre = optim.Adam(model.parameters(), lr=args.lr[0], weight_decay=1e-5)
    best_valid_acc = 0.
    best_model_wts = model.state_dict()
    for epoch in range(1, args.epochs[0] + 1):
        model.train()
        train_loss, num_correct, num_samples = 0, 0, 0
        for X, Y in train_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            evidences, evidence_a = model(X)
            loss = get_loss(evidences, evidence_a, Y, epoch, num_classes, args.annealing_step, 1, device)
            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()

            train_loss += loss
            num_correct += torch.sum(evidence_a.argmax(dim=-1).eq(Y)).item()
            num_samples += len(Y)

        train_acc = num_correct / num_samples
        valid_acc = test(model, test_loader, num_views, device)
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Pre-Train Epoch {epoch:2d}; train loss {train_loss:.4f}, '
              f'train acc {train_acc:.4f}; val acc: {valid_acc:.4f}')
    before_acc = best_valid_acc
    model.load_state_dict(best_model_wts)

    X_train, Y_train, alphas, alpha_a = test(model, train_loader, num_views, device, isTraindata=True)
    X_pseudo_set, Y_pseudo_set = uncertainty_SMOTE(X_train, Y_train, alphas, alpha_a, R=args.R, device=device)

    for v in range(num_views):
        X_train[v] = torch.cat((X_train[v], X_pseudo_set[v]), dim=0)
    Y_train = torch.cat((Y_train, Y_pseudo_set), dim=0).to(torch.int64)
    val_loader = DataLoader(MultiViewDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True)

    print('--------------Fine-Tuning------------------------')
    optimizer = optim.Adam(model.parameters(), lr=args.lr[1], weight_decay=1e-5)
    best_test_acc = 0.
    for epoch in range(1, args.epochs[1] + 1):
        model.train()
        train_loss, num_correct, num_samples = 0, 0, 0
        for X, Y in val_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            evidences, evidence_a = model(X)
            loss = get_loss(evidences, evidence_a, Y, epoch, num_classes, args.annealing_step, 1, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            num_correct += torch.sum(evidence_a.argmax(dim=-1).eq(Y)).item()
            num_samples += len(Y)

        train_acc = num_correct / num_samples
        test_acc = test(model, test_loader, num_views, device)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
        print(f'Fine-Tuning Epoch {epoch:2d}; train loss {train_loss:.4f}, '
              f'train acc {train_acc:.4f}; test acc: {test_acc:.4f}')
    after_acc = best_test_acc

    return before_acc, after_acc


def test(model, loader, num_views, device, isTraindata=False):
    model.eval()
    num_correct, num_samples = 0, 0
    if isTraindata:
        batch = [torch.tensor([]).to(device) for _ in range(num_views)]
        target = torch.tensor([], dtype=torch.int64).to(device)
        alphas = [torch.tensor([]).to(device) for _ in range(num_views)]
        alpha_a = torch.tensor([]).to(device)
    for X, Y in loader:
        for v in range(num_views):
            X[v] = X[v].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a = model(X)
            num_correct += torch.sum(evidence_a.argmax(dim=-1).eq(Y)).item()
            num_samples += Y.shape[0]
            if isTraindata:
                for v in range(num_views):
                    batch[v] = torch.cat((batch[v], X[v]), dim=0)
                    alphas[v] = torch.cat((alphas[v], evidences[v] + 1), dim=0)
                target = torch.cat((target, Y), dim=0)
                alpha_a = torch.cat((alpha_a, evidence_a + 1), dim=0)
    acc = num_correct / num_samples
    if isTraindata:
        return batch, target, alphas, alpha_a
    else:
        return acc
