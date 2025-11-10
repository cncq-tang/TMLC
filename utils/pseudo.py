import torch
import random

epsilon = 1e-8


def uncertainty_SMOTE(X_train, Y_train, alphas, alpha_a, R, device):
    num_views = len(X_train)
    num_classes = alpha_a.shape[1]

    num_classes_train = torch.unique(Y_train).tolist()
    class_samples_index = dict()
    for c in num_classes_train:
        index_tensor = torch.nonzero(Y_train == c).squeeze()
        if index_tensor.ndimension() == 0:
            class_samples_index[c] = [index_tensor.item()]
        else:
            class_samples_index[c] = index_tensor.tolist()

    E_views = [alpha - 1 for alpha in alphas]
    E = alpha_a - 1

    distance = torch.cdist(E, E, p=2)

    # e_ij=(e_i+e_j)/2
    E_views_couple = [(evidence_view.unsqueeze(1) + evidence_view.unsqueeze(0)) / 2 for evidence_view in E_views]

    # S_ij=\sum(e_ij+1)
    S_views_couple_sum = [torch.sum(E_couple + 1, dim=2, keepdim=True) for E_couple in E_views_couple]

    # p_ij=e_ij/S_ij
    P_views_couple_sum = [(E_views_couple[v] + 1) / S_views_couple_sum[v] for v in range(num_views)]

    # u_ij=num_classes/S_ij
    U_views_couple_sum = [num_classes / S_couple_sum for S_couple_sum in S_views_couple_sum]

    X_pseudo_set = [torch.tensor([]).to(device) for _ in range(num_views)]
    Y_pseudo_set = torch.tensor([], dtype=torch.int64).to(device)

    for c in num_classes_train:
        number = len(class_samples_index[0]) - len(class_samples_index[c])
        indices = torch.tensor(class_samples_index[c]).to(device)
        n = 0

        while n < number:
            random_index = random.choice(indices.tolist())
            distances = distance[random_index, :]

            nearest_indices = torch.topk(-distances, R + 1, largest=True).indices

            X_nearest = [X[nearest_indices] for X in X_train]
            P_nearest_couple = [P_couple_sum[random_index][nearest_indices] for P_couple_sum in P_views_couple_sum]
            U_nearest_couple = [U_couple_sum[random_index][nearest_indices] for U_couple_sum in U_views_couple_sum]

            uncertainty_entropy = [torch.exp(U_nearest_couple[v]) * -torch.log(P_nearest_couple[v][:, c:c + 1])
                                   for v in range(num_views)]
            uncertainty_entropy_inverse = [1 / (uncertainty_entropy[v] + epsilon) for v in range(num_views)]

            uncertainty_entropy_inverse_sum = [torch.sum(uncertainty_entropy_inverse[v], dim=0, keepdim=True)
                                               for v in range(num_views)]
            X_nearest_weights = [uncertainty_entropy_inverse[v] / uncertainty_entropy_inverse_sum[v]
                                 for v in range(num_views)]

            X_pseudo = [torch.sum(X_nearest[v] * X_nearest_weights[v], dim=0, keepdim=True) for v in range(num_views)]

            for v in range(num_views):
                X_pseudo_set[v] = torch.cat((X_pseudo_set[v], X_pseudo[v]), dim=0)
            Y_pseudo_set = torch.cat((Y_pseudo_set, torch.tensor([c], device=device)), dim=0)
            n += 1

    return X_pseudo_set, Y_pseudo_set
