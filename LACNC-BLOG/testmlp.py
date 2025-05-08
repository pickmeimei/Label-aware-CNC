if num % 20 == 19 and turn < 5 and train_pair_noisy_1 != []:
    net = Net2(1536, 512, 256);
    loss_function = nn.BCELoss();
    optimizer = optim.Adam(net.parameters(), lr=1e-3);
    classify += 1
    print("第" + str(turn) + "轮迭代的第" + str(classify) + "次鉴别开始。。。")
    Lvecpos = np.array([vec[e] for e in train_pair_pos1]);    #是个矩阵还是什么，列表里的矩阵怎么取出来
    Rvecpos = np.array([vec[e] for e in train_pair_pos2]);
    lenclassify0 = Lvecpos.shape[0];
    martix = np.hstack([Lvecpos, Rvecpos]);
    martix = torch.tensor(martix)
    y = torch.ones((lenclassify0, 1));
    y1 = torch.zeros((lenclassify0, 1));
    martix_ne = negativasample(train_pair_pos1);
    martix_ne2 = negativasample(train_pair_pos1);
    martix_ne3 = negativasample(train_pair_pos1);
    martix_ne4 = negativasample(train_pair_pos1);
    martix_ne5 = negativasample(train_pair_pos1)

    martix_net = negativasample(train_pair_pos2);
    martix_net2 = negativasample(train_pair_pos2);
    martix_net3 = negativasample(train_pair_pos2);
    martix_net4 = negativasample(train_pair_pos2);
    martix_net5 = negativasample(train_pair_pos2);

    if negative != []:
        triples_ne1 = np.array([vec[e] for e in negative_1]);
        triples_ne2 = np.array([vec[e] for e in negative_2]);
        martix_neg = np.hstack([triples_ne1, triples_ne2]);
        martix_neg = torch.tensor(martix_neg)
        lenne = triples_ne1.shape[0];

    for i in range(400):
        net.train()
        pred = net(martix);
        pred1 = net(martix_ne);
        predt1 = net(martix_net);
        pred2 = net(martix_ne2);
        predt2 = net(martix_net2);
        pred3 = net(martix_ne3);
        predt3 = net(martix_net3);
        pred4 = net(martix_ne4);
        predt4 = net(martix_net4);
        pred5 = net(martix_ne5);
        predt5 = net(martix_net5);
        loss = loss_function(pred, y);
        loss1 = loss_function(pred1, y1);
        losst1 = loss_function(predt1, y1);
        loss2 = loss_function(pred2, y1);
        losst2 = loss_function(predt2, y1);
        loss3 = loss_function(pred3, y1);
        losst3 = loss_function(predt3, y1);
        loss4 = loss_function(pred4, y1);
        losst4 = loss_function(predt4, y1);
        loss5 = loss_function(pred5, y1);
        losst5 = loss_function(predt5, y1);
        if negative != []:
            yn = torch.zeros((lenne, 1));
            predn = net(martix_neg);
            lossn = loss_function(predn, yn)
            loss_all = loss + loss1 + losst1 + loss2 + losst2 + loss3 + losst3 + lossn + loss4 + losst4 + loss5 + losst5
        else:
            loss_all = loss + loss1 + losst1 + loss2 + losst2 + loss3 + losst3 + losst4 + losst4 + loss5 + losst5
        optimizer.zero_grad();
        loss_all.backward();
        optimizer.step()
    net.eval()
    print(loss_all)
    print("本轮阈值：" + str(lamta))