# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
def eval_ukb(net, test_transform, outputs=['inst_out'], num_workers=5, batch_size=256, **dataset_kwargs):
    datas = UKBench(transform=test_transform, **dataset_kwargs)
    test_loader = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    net.eval()
    features = [[] for o in outputs]

    pbar = tqdm(total=len(datas))
    with torch.no_grad():
        for i, (data, label) in enumerate(iter(test_loader)):
            B = data.size(0)
            out = net({'input': data})
            for o, flist in zip(outputs, features):
                flist.append(out[o].cpu())
            pbar.update(B)
    pbar.close()

    features = [torch.cat(f, dim=0) for f in features]
    d_mats = [get_distance_matrix(f) for f in features]
    preds = [torch.sort(d.cpu(), dim=1)[1] for d in d_mats]

    GT = np.array([k[1] for k in datas.imgs])

    scores = [(GT.reshape(-1, 1) == GT[p[:, :4].cpu().numpy()]).mean().item() * 4 for p in preds]
    return scores


def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = torch.sum(x ** 2.0, dim=1, keepdim=True)
    distance_square = square + square.t() - (2.0 * torch.matmul(x, x.t()))
    return F.relu(distance_square) ** 0.5

def eval_holidays(net, test_transform=None, output='inst_out', pool=None, num_workers=5, batch_size=1, test_loader=None,
                  N=None, fix_rotate=False):
    net.eval()
    if test_loader is None:
        datas = Holidays(transform=test_transform, rotated=True, fix_rotate=fix_rotate)
        test_loader = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=True)
    outputs = []
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(test_loader, total=N if N is not None else len(test_loader))):

            out = net({'input': data.cuda()})
            feat = out[output]
            if pool is not None:
                feat = pool(feat)
            outputs.append(feat.detach().cpu())
            if N is not None and i > N:
                break

    outputs = torch.cat(outputs, dim=0)
    outputs = outputs.squeeze()

    d_mat = get_distance_matrix(outputs)
    # d_mat_ic = get_distance_matrix(outputs_ic)
    # solve problem of duplicates by removing 1 on diagonal
    d_mat -= torch.eye(d_mat.size(0))
    # d_mat_ic -= torch.eye(d_mat.size(0))
    _, pred = torch.sort(d_mat, dim=1)
    # _, pred_ic = torch.sort(d_mat_ic, dim=1)

    idx_to_query = -torch.ones(pred.size(0), dtype=torch.long)
    queries = []
    for g in test_loader.dataset.class_groups.values():
        if N is not None and any([i > N for i in g]):
            continue
        idx_to_query[g[1:]] = g[0]
        queries.append(g[0])
    queries = torch.tensor(queries)

    pred_q = pred[queries]

    assert (pred_q[:, 0] != queries).sum() == 0
    results = pred_q[:, 1:]

    results_class = idx_to_query[results]
    correct = results_class == queries.view(-1, 1)

    map = 0.
    for corr in correct:
        ranks = corr.nonzero().numpy().reshape(-1)
        ap = score_ap(ranks, len(ranks))
        map += ap
        # print(ranks, ap)
    score = map / len(correct)
    return score


def eval_copydays(net, test_transform, output='inst_out', pool=None, num_workers=5, test_loader=None, batch_size=1, **kwargs):
    net.eval()
    if test_loader is None:
        datas = CopyDays(transform=test_transform, subset=['strong'], **kwargs)
        test_loader = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=True)
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):

            out = net({'input': data['input'].cuda()})
            feat = out[output]
            if pool is not None:
                feat = pool(feat)
            outputs.append(feat.detach().cpu())

    outputs = torch.cat(outputs, dim=0)
    outputs = outputs.squeeze()

    d_mat = get_distance_matrix(outputs)
    # d_mat_ic = get_distance_matrix(outputs_ic)
    # solve problem of duplicates by removing 1 on diagonal
    d_mat -= torch.eye(d_mat.size(0))
    # d_mat_ic -= torch.eye(d_mat.size(0))
    _, pred = torch.sort(d_mat, dim=1)
    # _, pred_ic = torch.sort(d_mat_ic, dim=1)

    # idx_to_query = -torch.ones(pred.size(0), dtype=torch.long)

    map = 0.
    seen = 0

    queries = set()
    for g in test_loader.dataset.class_groups:
        for query in g[1:]:
            queries.add(query)

    # ranks = []
    # all_neighbours = {}
    for g in test_loader.dataset.class_groups:
        for query in g[1:]:
            neighbours = pred[query]
            # all_neighbours[query] = neighbours
            rank = 0
            for n in neighbours:
                n = n.item()
                if n == g[0]:
                    break
                elif n not in queries:
                    rank += 1
            map += score_ap([rank], 1)
            # ranks.append(rank)
            seen += 1
    score = map / seen
    return score # , ranks, all_neighbours, queries

