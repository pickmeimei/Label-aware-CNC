import random
import time

import torch

from parse_args import *
from utils1 import *
from utils import *
from dgi import *
import torch.nn.functional as F
import loss_func
import torch.optim as optim
import torch.nn as nn
import math


def main(args):
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的, 方便下次复现
    np.random.seed(args.seed)  # 函数用于生成指定随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    adj_s, feature_s, label_s, idx_tot_s, c, c1, cn, cn1, adj_s_dense, A_s_num = search_s_data1(
        dataset=args.source_dataset + '.mat', device=device,
        seed=args.seed, is_blog=args.is_blog)

    adj_t, feature_t, label_t, idx_tot_t, b, adj_t_dense, ct, ctn, A_t_num ,A_head ,A_tail = search_t_data_tail(
        dataset=args.target_dataset + '.mat', device=device,
        seed=args.seed, is_blog=args.is_blog)


    # 目标的二跳节点对返回在字典b中
    dict = b
    # print(dict)

    # 找出尾的正样本节点对放入列表postail中，负样本节点对放入negtail中
    postail = []
    postaildeg = []
    for i in c:
        # a = []
        for j in c[i]:
            temp = []
            temp.append(i)
            temp.append(j)
            # a.append(temp)
            x = A_s_num[j]
            postaildeg.append(x)
            postail.append(temp)
    negtail = []
    negtaildeg = []
    for m in cn:
        for n in cn[m]:
            temp1 = []
            temp1.append(m)
            temp1.append(n)
            x = A_s_num[n]
            negtaildeg.append(x)
            negtail.append(temp1)

    # 找出头的正样本节点对放入列表poshead中，负样本节点对放入neghead中
    poshead = []
    for i in c1:
        # a = []
        for j in c1[i]:
            temp2 = []
            temp2.append(i)
            temp2.append(j)
            # a.append(temp)
            poshead.append(temp2)
    neghead = []
    for m in cn1:
        for n in cn1[m]:
            temp3 = []
            temp3.append(m)
            temp3.append(n)
            neghead.append(temp3)

    # 找出目标的每个节点的二跳节点对放进targethop中，再取出其表示，看其是否正相关。
    targethop = []
    for k in b:
        for i in b[k]:
            temp4 = []
            temp4.append(k)
            temp4.append(i)
            targethop.append(temp4)

    postail_t = []
    for i in ct:
        # a = []
        for j in ct[i]:
            temp5 = []
            temp5.append(i)
            temp5.append(j)
            # a.append(temp)
            postail_t.append(temp5)

    negtail_t = []
    for i in ctn:
        # a = []
        for j in ctn[i]:
            temp6 = []
            temp6.append(i)
            temp6.append(j)
            # a.append(temp)
            negtail_t.append(temp6)

    len1 = len(postail_t)
    targethop1 = postail_t;
    temp7 = 0
    for i in range(len(negtail_t)):
        if temp7 >= len1:
            break
        targethop1.append(random.choice(negtail_t))
        # ep.append(random.sample(e,1))
        temp7 += 1

    length = len(A_s_num)
    number = sum(A_s_num, length)
    number = (number / length)
    print(number)

    length1 = len(A_t_num)
    number1 = sum(A_t_num, length1)
    number1 = (number1 / length1)
    print(number1)

    n_samples = args.n_samples.split(',')
    output_dims = args.output_dims.split(',')
    emb_model1 = GraphSAGE1(**{
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        "input_dim": feature_s.shape[1],
        "layer_specs": [
            {
                "n_sample": int(n_samples[0]),
                "output_dim": int(output_dims[0]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[1]),
                "output_dim": int(output_dims[1]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[-1]),
                "output_dim": int(output_dims[-1]),
                "activation": F.relu,
            }
        ],
        "device": device
    }).to(device)
    # cly_model = Cly_net(2 * int(output_dims[-1]), label_s.shape[1], args.arch_cly).to(device)   #分类器
    # disc_model = Disc(2 * int(output_dims[-1]) * label_s.shape[1], args.arch_disc, 1).to(device)  #鉴别器
    # define the optimizers
    # total_params = list(emb_model1.parameters()) + list(cly_model.parameters()) + list(disc_model.parameters())
    total_params = list(emb_model1.parameters())
    # dgi_model = DGI(2 * int(output_dims[-1])).to(device)       #最大化局部全局互信息
    # total_params += list(dgi_model.parameters())
    # cly_optim = torch.optim.Adam(total_params, lr=args.lr_cly, weight_decay=args.weight_decay)
    lr_lambda = lambda epoch: (1 + 10 * float(epoch) / args.epochs1) ** (-0.75)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(cly_optim, lr_lambda=lr_lambda)
    best_micro_f1, best_macro_f1, best_epoch = 0, 0, 0
    num_batch = round(max(feature_s.shape[0] / (args.batch_size / 2), feature_t.shape[0] / (args.batch_size / 2)))
    for epoch in range(args.epochs1):
        if epoch >= 0:
            t0 = time.time()
        s_batches = batch_generator1(idx_tot_s, int(args.batch_size / 2))
        t_batches = batch_generator1(idx_tot_t, int(args.batch_size / 2))
        emb_model1.train()
        # cly_model.train()
        # disc_model.train()
        # dgi_model.train()
        p = float(epoch) / args.epochs1
        # grl_lambda = min(2. / (1. + np.exp(-10. * p)) - 1, 0.2)
        for iter in range(num_batch):
            b_nodes_s = next(s_batches)
            b_nodes_t = next(t_batches)
            source_features = do_iter1(emb_model1, adj_s, feature_s, label_s, idx=b_nodes_s,
                                       is_social_net=args.is_social_net)
            target_features = do_iter1(emb_model1, adj_t, feature_t, label_t, idx=b_nodes_t,
                                       is_social_net=args.is_social_net)
            # dgi扰动进行了增强，可能能缓解稀疏性
            # shuf_idx_s = np.arange(label_s.shape[0])
            # np.random.shuffle(shuf_idx_s)
            # shuf_feat_s = feature_s[shuf_idx_s, :]
            # shuf_idx_t = np.arange(label_t.shape[0])
            # np.random.shuffle(shuf_idx_t)
            # shuf_feat_t = feature_t[shuf_idx_t, :]
            # neg_source_feats = emb_model1(b_nodes_s, adj_s, shuf_feat_s)
            # logits_s = dgi_model(neg_source_feats, source_features)
            # neg_target_feats = emb_model1(b_nodes_t, adj_t, shuf_feat_t)
            # logits_t = dgi_model(neg_target_feats, target_features)
            # labels_dgi = torch.cat(
            #     [torch.zeros(int(args.batch_size / 2)), torch.ones(int(args.batch_size / 2))]).unsqueeze(0).to(device)
            # dgi_loss = args.dgi_param * (
            #             F.binary_cross_entropy_with_logits(logits_s, labels_dgi) + F.binary_cross_entropy_with_logits(
            #         logits_t, labels_dgi))
            features = torch.cat((source_features, target_features), 0)
            # outputs = cly_model(features)
            # softmax_output = nn.Softmax(dim=1)(outputs)
            # domain_loss = args.cdan_param * loss_func.CDAN([features, softmax_output], disc_model, None, grl_lambda,
            #                                                None, device=device)
            # loss = cly_loss_s + dgi_loss + domain_loss
            # loss = cly_loss_s + domain_loss
            # cly_optim.zero_grad()
            # loss.backward()
            # cly_optim.step()

        emb_model1.eval()
        # print(source_features)
        # cly_model.eval()

        embs_whole_s, targets_whole_s = evaluate1(emb_model1, adj_s, feature_s, label_s, idx_tot_s, args.batch_size,
                                                  mode='test', is_social_net=args.is_social_net)

        embs_whole_t, targets_whole_t = evaluate1(emb_model1, adj_t, feature_t, label_t, idx_tot_t, args.batch_size,
                                                  mode='test', is_social_net=args.is_social_net)

    embs_whole_s = (embs_whole_s - embs_whole_s.mean(axis=0)) / (embs_whole_s.std(axis=0))
    embs_whole_s = torch.tensor(embs_whole_s)
    embs_whole_s = torch.where(torch.isnan(embs_whole_s), torch.full_like(embs_whole_s, 0), embs_whole_s)
    embs_whole_s = torch.where(embs_whole_s == -float('inf'), torch.zeros_like(embs_whole_s), embs_whole_s)
    embs_whole_s = embs_whole_s.detach().numpy()

    embs_whole_t = (embs_whole_t - embs_whole_t.mean(axis=0)) / (embs_whole_t.std(axis=0))
    embs_whole_t = torch.tensor(embs_whole_t)
    embs_whole_t = torch.where(torch.isnan(embs_whole_t), torch.full_like(embs_whole_t, 0), embs_whole_t)
    embs_whole_t = torch.where(embs_whole_t == -float('inf'), torch.zeros_like(embs_whole_t), embs_whole_t)
    embs_whole_t = embs_whole_t.detach().numpy()

    e = []  # 头和尾的正样本的表示
    length = len(poshead)
    # for k in range(length):
    #     el = embs_whole_s[poshead[k][0]]
    #     er = embs_whole_s[poshead[k][1]]
    #     e1 = abs(el-er)
    #     e2 = el+er
    #     e3 = np.multiply(el,er)
    #     #x=np.hstack([el,er])
    #     x=np.hstack([e1,e2,e3])
    #     e.append(x)
    for k in range(len(postail)):
        el = embs_whole_s[postail[k][0]]
        er = embs_whole_s[postail[k][1]]
        e1 = abs(el - er)
        e2 = el + er
        e3 = np.multiply(el, er)
        # x = np.hstack([el, er])
        x = np.hstack([e1, e2, e3])
        e.append(x)

    en = []  # 头和尾的负样本的表示
    # for k in range(len(neghead)):
    #     enl = embs_whole_s[neghead[k][0]]
    #     enr = embs_whole_s[neghead[k][1]]
    #     en1 = abs(enl - enr)
    #     en2 = enl + enr
    #     en3 = np.multiply(enl, enr)
    #     #x=np.hstack([enl,enr])
    #     x = np.hstack([en1, en2, en3])
    #     en.append(x)
    for k in range(len(negtail)):
        enl = embs_whole_s[negtail[k][0]]
        enr = embs_whole_s[negtail[k][1]]
        en1 = abs(enl - enr)
        en2 = enl + enr
        en3 = np.multiply(enl, enr)
        # x=np.hstack([enl,enr])
        x = np.hstack([en1, en2, en3])
        en.append(x)

    ##训练一个MLP，看看其鉴别能力怎么样
    ###################################################
    a = int(len(e))
    eneg = [];
    temp4 = 0
    for i in range(len(en)):
        if temp4 >= a:
            break
        eneg.append(random.choice(en))
        # ep.append(random.sample(e,1))
        temp4 += 1
    en = torch.tensor(eneg)
    lenneg = len(en)
    e = torch.tensor(e)
    lenpos = len(e)
    # net = Net(384,128, 64);
    # net = Net(768, 384, 128);
    net = Net12(768, 384, 256, 128, 32)
    # net = Net(192, 256, 128, 64, 32);

    ##########
    loss_function = nn.BCELoss();
    optimizer = optim.Adam(net.parameters(), lr=1e-3);
    y = torch.ones((lenpos, 1));
    y1 = torch.zeros((lenneg, 1));

    for i in range(800):
        net.train()
        pred = net(e);
        # pred = pred.detach().numpy()

        pred1 = net(en);
        # pred1 = pred1.detach().numpy()

        loss = loss_function(pred, y);
        loss1 = loss_function(pred1, y1);
        loss_all = loss + loss1
        optimizer.zero_grad();
        loss_all.backward();
        optimizer.step()

    lamta1 = 0
    for i in range(len(pred)):
        lamta1 += pred[i][0]
    lamta1 = lamta1 / len(pred)
    lamta1 = lamta1.detach().numpy();

    pred = pred.detach().numpy()
    pred1 = pred1.detach().numpy()

    lamta = 0.5
    net.eval()
    print(loss_all)
    print("本轮阈值（平均值）：" + str(lamta1))
    print("手动设定本轮阈值：" + str(lamta))

    # 对源域和目标域都进行修改
    # 源域添边
    source_edge = 0
    for i in c:
        for j in c[i]:
            if A_s_num[i] < 15 and A_s_num[j] < 15:
                adj_s_dense[i][j] = 1
                adj_s_dense[j][i] = 1
                source_edge += 1

    print("源域添边" + str(source_edge))

    # source_edge = 0
    # for i in c:
    #     for j in c[i]:
    #         adj_s_dense[i][j] = 1
    #         adj_s_dense[j][i] = 1
    #         source_edge += 1
    #
    # print("源域添边" + str(source_edge))

    # 对目标表示进行处理，测试
    # 目标域的二跳节点的表示,找的时候就判断
    dict = {}
    rightnumber = 0
    falsenumber = 0
    for i in range(len(targethop1)):
        targe = []
        m = targethop1[i][0]
        n = targethop1[i][1]
        el = embs_whole_t[targethop1[i][0]]
        er = embs_whole_t[targethop1[i][1]]
        em1 = abs(el - er)
        em2 = el + er
        em3 = np.multiply(el, er)
        emb = np.hstack([em1, em2, em3])
        emb = torch.tensor(emb)
        # net = Net13(384,1024,512,256,128,128,256,512,1024)
        # net = Net(384, 128, 64)
        # net = Net(768, 384, 128)
        net = Net12(768, 384, 256, 128, 32)
        # net = Net12(192, 256, 128, 64, 32);
        tarpred = net(emb)
        tarpred = tarpred.detach().numpy()
        if (label_t[m] == label_t[n]).all():
            flag = 1
        else:
            flag = 0
        x1 = [tarpred, flag]
        x2 = tuple([m, n])
        targe.append(x1)
        dict[x2] = targe

        # print(dict)
        # targe =targe.detach().numpy()

        # if (label_t[m] == label_t[n]).all():
        #     adj_t_dense[m][n] = 1
        #     adj_t_dense[n][m] = 1
        #     rightnumber = rightnumber + 1

        if tarpred >= lamta and A_t_num[m] < 15 and A_t_num[n] < 15:
            ######是否要加一个度不超过多少的条件

            adj_t_dense[m][n] = 1
            adj_t_dense[n][m] = 1

            if (label_t[m] == label_t[n]).all():
                rightnumber = rightnumber + 1
            else:
                falsenumber = falsenumber + 1
    # for key in dict:
    #     if
    dict_sort = {}
    dict_tuple_sorted = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (dict_key, dict_val) in enumerate(dict_tuple_sorted, 1):
        dict_sort[dict_key] = (rank, dict_val)

    print('加入的正确的节点对有' + str(rightnumber) + '个')
    print('加入的错误的节点对有' + str(falsenumber) + '个')

    # 加载预训练后的数据
    adj_s, feature_s, label_s, idx_tot_s = load_data_s1(dataset=args.source_dataset + '.mat', device=device,
                                                        seed=args.seed, is_blog=args.is_blog, adj_new_dense=adj_s_dense)
    adj_t, feature_t, label_t, idx_tot_t = load_data_t1(dataset=args.target_dataset + '.mat', device=device,
                                                        seed=args.seed, is_blog=args.is_blog, adj_new_dense=adj_t_dense)
    # 重新训练

    del emb_model1

    emb_model = GraphSAGE(**{
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        "input_dim": feature_s.shape[1],
        "layer_specs": [
            {
                "n_sample": int(n_samples[0]),
                "output_dim": int(output_dims[0]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[1]),
                "output_dim": int(output_dims[1]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[-1]),
                "output_dim": int(output_dims[-1]),
                "activation": F.relu,
            }
        ],
        "device": device
    }).to(device)
    pse_model = Pseudo_Loss(label_s.shape[1]).to(device)

    cly_model = Cly_net(2 * int(output_dims[-1]), label_s.shape[1], args.arch_cly).to(device)  # 分类器
    disc_model = Disc(2 * int(output_dims[-1]) * label_s.shape[1], args.arch_disc, 1).to(device)  # 鉴别器
    # define the optimizers
    total_params = list(emb_model.parameters()) + list(cly_model.parameters()) + list(disc_model.parameters())
    dgi_model = DGI(2 * int(output_dims[-1])).to(device)  # 最大化局部全局互信息
    total_params += list(dgi_model.parameters())

    total_params += list(pse_model.parameters())

    cly_optim = torch.optim.Adam(total_params, lr=args.lr_cly, weight_decay=args.weight_decay)
    lr_lambda = lambda epoch: (1 + 10 * float(epoch) / args.epochs) ** (-0.75)
    scheduler = torch.optim.lr_scheduler.LambdaLR(cly_optim, lr_lambda=lr_lambda)
    best_micro_f1, best_macro_f1, best_epoch = 0, 0, 0
    best_micro_f1_head, best_macro_f1_head, best_epoch_head = 0, 0, 0
    best_micro_f1_tail, best_macro_f1_tail, best_epoch_tail = 0, 0, 0
    num_batch = round(max(feature_s.shape[0] / (args.batch_size / 2), feature_t.shape[0] / (args.batch_size / 2)))
    for epoch in range(args.epochs):
        if epoch >= 0:
            t0 = time.time()
        s_batches = batch_generator(idx_tot_s, int(args.batch_size / 2))
        t_batches = batch_generator(idx_tot_t, int(args.batch_size / 2))
        emb_model.train()
        cly_model.train()
        disc_model.train()
        dgi_model.train()

        pse_model.train()

        p = float(epoch) / args.epochs
        grl_lambda = min(2. / (1. + np.exp(-10. * p)) - 1, 0.2)
        for iter in range(num_batch):
            b_nodes_s = next(s_batches)
            b_nodes_t = next(t_batches)
            source_features, cly_loss_s = do_iter(emb_model, cly_model, adj_s, feature_s, label_s, idx=b_nodes_s,
                                                  is_social_net=args.is_social_net)
            target_features, _ = do_iter(emb_model, cly_model, adj_t, feature_t, label_t, idx=b_nodes_t,
                                         is_social_net=args.is_social_net)
            shuf_idx_s = np.arange(label_s.shape[0])
            np.random.shuffle(shuf_idx_s)
            shuf_feat_s = feature_s[shuf_idx_s, :]
            shuf_idx_t = np.arange(label_t.shape[0])
            np.random.shuffle(shuf_idx_t)
            shuf_feat_t = feature_t[shuf_idx_t, :]
            neg_source_feats = emb_model(b_nodes_s, adj_s, shuf_feat_s)
            logits_s = dgi_model(neg_source_feats, source_features)
            neg_target_feats = emb_model(b_nodes_t, adj_t, shuf_feat_t)
            logits_t = dgi_model(neg_target_feats, target_features)
            labels_dgi = torch.cat(
                [torch.zeros(int(args.batch_size / 2)), torch.ones(int(args.batch_size / 2))]).unsqueeze(0).to(device)
            dgi_loss = args.dgi_param * (
                    F.binary_cross_entropy_with_logits(logits_s, labels_dgi) + F.binary_cross_entropy_with_logits(
                logits_t, labels_dgi))
            features = torch.cat((source_features, target_features), 0)

            outputs = cly_model(features)
            softmax_output = nn.Softmax(dim=1)(outputs)
            domain_loss = args.cdan_param * loss_func.CDAN([features, softmax_output], disc_model, None, grl_lambda,
                                                           None, device=device)

            pseudo_loss_t = pse_model(target_features)

            loss = cly_loss_s + dgi_loss + domain_loss + 0.001 * pseudo_loss_t

            # loss = cly_loss_s + dgi_loss + domain_loss
            # loss = cly_loss_s + domain_loss
            cly_optim.zero_grad()
            loss.backward()
            cly_optim.step()

        emb_model.eval()
        cly_model.eval()
        cly_loss_bat_s, micro_f1_s, macro_f1_s, embs_whole_s, targets_whole_s = evaluate(emb_model, cly_model, adj_s,
                                                                                         feature_s, label_s,
                                                                                         idx_tot_s, args.batch_size,
                                                                                         mode='test',
                                                                                         is_social_net=args.is_social_net)

        print("epoch {:03d} | source loss {:.4f} | source micro-F1 {:.4f} | source macro-F1 {:.4f}".
              format(epoch, cly_loss_bat_s, micro_f1_s, macro_f1_s))
        print("pseudo_loss  {:.8f} ".format(pseudo_loss_t))
        cly_loss_bat_t, micro_f1_t, macro_f1_t, embs_whole_t, targets_whole_t = evaluate(emb_model, cly_model, adj_t,
                                                                                         feature_t, label_t,
                                                                                         idx_tot_t, args.batch_size,
                                                                                         mode='test',
                                                                                         is_social_net=args.is_social_net)

        print("target loss {:.4f} | target micro-F1 {:.4f} | target macro-F1 {:.4f}".format(cly_loss_bat_t, micro_f1_t,
                                                                                            macro_f1_t))

        cly_loss_bat_t, tail_micro_f1_t, tail_macro_f1_t, head_micro_f1_t, head_macro_f1_t, embs_whole_t, targets_whole_t = evaluate_tail(emb_model, cly_model,
                                                                                              adj_t,
                                                                                              feature_t, label_t,
                                                                                              idx_tot_t,
                                                                                              args.batch_size,
                                                                                              mode='test',
                                                                                              is_social_net=args.is_social_net,
                                                                                              A_head_list=A_head,
                                                                                              A_tail_list=A_tail)

        print("tail_target micro-F1 {:.4f} | tail_target macro-F1 {:.4f}".format(tail_micro_f1_t, tail_macro_f1_t))

        print("head_target micro-F1 {:.4f} | head_target macro-F1 {:.4f}".format(head_micro_f1_t, head_macro_f1_t))


        if (micro_f1_t + macro_f1_t) > (best_micro_f1 + best_macro_f1):
            best_micro_f1 = micro_f1_t
            best_macro_f1 = macro_f1_t
            best_micro_f1_head = head_micro_f1_t
            best_macro_f1_head = head_macro_f1_t
            best_micro_f1_tail = tail_micro_f1_t
            best_macro_f1_tail = tail_macro_f1_t
            best_epoch = epoch
            print('saving model...')
        scheduler.step()




        # if (tail_micro_f1_t + tail_macro_f1_t) > (best_micro_f1_tail + best_macro_f1_tail):
        #     best_micro_f1_tail = tail_micro_f1_t
        #     best_macro_f1_tail = tail_macro_f1_t
        #     best_epoch_tail = epoch
        #     print('saving model...')
        # scheduler.step()
        #
        #
        # if (head_micro_f1_t + head_macro_f1_t) > (best_micro_f1_head + best_macro_f1_head):
        #     best_micro_f1_head = head_micro_f1_t
        #     best_macro_f1_head = head_macro_f1_t
        #     best_epoch_head = epoch
        #     print('saving model...')
        # scheduler.step()

    print("Time(s) {:.4f} ".
          format(time.time() - t0))
    print("test metrics on target graph:")
    print('---------- random seed: {:03d} ----------'.format(args.seed))
    print("micro-F1 {:.4f} | macro-F1 {:.4f}".format(best_micro_f1, best_macro_f1))
    print("head_micro-F1 {:.4f} | head_macro-F1 {:.4f}".format(best_micro_f1_head, best_macro_f1_head))
    print("tail_micro-F1 {:.4f} | tail_macro-F1 {:.4f}".format(best_micro_f1_tail, best_macro_f1_tail))

    f.write(
        'best epoch: %d \t best micro_f1: %.6f \t best macro_f1: %.6f \n' % (best_epoch, best_micro_f1, best_macro_f1))
    f.flush()

    # f.write(
    #     'best epoch: %d \t best micro_f1_tail: %.6f \t best macro_f1_tail: %.6f \n' % (best_epoch, best_micro_f1_tail, best_macro_f1_tail))
    # f.flush()
    #
    # f.write(
    #     'best epoch: %d \t best micro_f1_head: %.6f \t best macro_f1_head: %.6f \n' % (
    #     best_epoch, best_micro_f1_head, best_macro_f1_head))
    # f.flush()


if __name__ == '__main__':
    args = parse_args()

    f = open('output/' + args.source_dataset + '_' + args.target_dataset + '.txt', 'a+')
    f.write('\n\n\n{}\n'.format(args))
    f.flush()

    device = torch.device('cuda:' + str(args.device))
    print(device)
    main(args)

pass


