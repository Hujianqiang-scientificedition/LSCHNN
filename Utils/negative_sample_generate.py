import random
import numpy as np
neg_train_num = 1
neg_test_num = 29
def neg_data_generate(adj_data_all,train_data_fix,val_data_fix,seed):
    random.seed(seed)
    train_neg_ls_all = []
    val_neg_1_ls = []
    arr_true = np.zeros((190,219,163))
    for line in adj_data_all:
        arr_true[int(line[0]), int(line[1]), int(line[2])] = 1
    arr_false_train = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])),
                                len(set(adj_data_all[:,2]))))


    L1 = 0
    for i in train_data_fix:
        L1 += 1
        k1 = 0
        tr_diet_ls = [j for j in range(0, arr_true.shape[0])]
        tr_mic_ls = [j for j in range(0, arr_true.shape[1])]
        tr_dis_ls = [j for j in range(0, arr_true.shape[2])]
        while k1 < neg_train_num:
            a = int(i[0])
            b = random.randint(0, arr_true.shape[1] - 1)
            c = int(i[2])
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                k1 += 1
                train_neg_ls_all.append((a, b, c, 0))
            else:
                distance_t2 = neg_train_num - k1
                # print('triplet:', i, 'tr_4:', distance_t4)
                last_ind = len(train_neg_ls_all) - 1
                for k in range(distance_t2):
                    train_neg_ls_all.append(train_neg_ls_all[last_ind])
                break


    train_neg_all = np.array(train_neg_ls_all)
    train_data_all = np.vstack((np.array(train_neg_ls_all), train_data_fix))
    np.random.shuffle(train_neg_all)
    np.random.shuffle(train_data_all)
    L2 = 0
    for i in val_data_fix:
        t1 = 0
        neg_1_i = []
        # Because it is too easy to repeat, it is only guaranteed that multiple negative samples generated for a
        # certain positive sample are not repeated, and negative samples of different positive samples may be repeated.
        arr_false_val_1 = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])), len(set(adj_data_all[:,2]))))
        neg_1_i.append(i)
        L2 += 1
        arr_false_val_2 = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])), len(set(adj_data_all[:,2]))))
        diet_ls = [j for j in range(0, arr_true.shape[0])]
        mic_ls = [j for j in range(0, arr_true.shape[1])]
        dis_ls = [j for j in range(0, arr_true.shape[2])]
        while t1 < neg_test_num:
            a_3 = int(i[0])
            c_3 = int(i[2])
            if mic_ls != []:
                b_3 = random.choice(mic_ls)
                mic_ls.remove(b_3)
                if arr_true[a_3, b_3, c_3] != 1 and arr_false_train[a_3, b_3, c_3] != 1 and arr_false_val_1[
                    a_3, b_3, c_3] != 1:
                    arr_false_val_1[a_3, b_3, c_3] = 1
                    t1 += 1
                    neg_1_i.append((a_3, b_3, c_3, 0))
            else:
                distance_3 = neg_test_num - t1
                # print('triplet:', i, 'val_3:', distance_3)
                last_ind = len(neg_1_i) - 1
                for k in range(distance_3):
                    neg_1_i.append(neg_1_i[last_ind])
                break
        np.random.shuffle(neg_1_i)
        val_neg_1_ls.extend(neg_1_i)
        # print('fold_num:', fold_num, 'neg_2:', neg_2, 'neg_3:', neg_3, 'neg_4:', neg_4)
    return train_data_all, train_neg_all, val_neg_1_ls
