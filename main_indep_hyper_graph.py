import torch.backends.cudnn
from model_hyper_graph import *
from Utils.Tools2 import *
from Utils.negative_sample_generate import *
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import confusion_matrix, classification_report

resultFileName = 'Data/indep_results.txt'
sorted_pred_dict = {}
predictFileName = 'Data/predict_results.txt'
pred_val = torch.tensor([])
def series_num(data):
    for line in data:
        line[1] = line[1] + 190
        line[2] = line[2] + 190 + 219
    return data

def random_seed(sd):
    random.seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

def train(diet_fea, mic_fea, dis_fea, hg_pos, hg_neg_ls, train_data):
    model.train()
    print('--- Start training ---')
    optimizer.zero_grad()
    _, pred, train_embed_pos, graph_embed_neg_ls, train_summary, outcome = model(diet_fea, mic_fea, dis_fea, hg_pos,
                                                                     train_data[:, 0], train_data[:, 1],
                                                                     train_data[:, 2], hg_neg_ls)
    loss_1 = my_loss(pred.view(-1, 1), torch.from_numpy(train_data[:, 3]).view(-1, 1).float())
    loss_2 = model.DGI_loss(train_embed_pos, train_summary, graph_embed_neg_ls)
    loss = args.hypergraph_loss_ratio * loss_1 + (1 - args.hypergraph_loss_ratio) * loss_2
    loss.backward()
    optimizer.step()
    print('epoch:{:02d},'.format(e + 1), 'loss_train:{:.6f},'.format(loss.item()))
    return loss.item()

def test(diet_fea, mic_fea, dis_fea, hg_pos, hg_neg_ls, val_data_all, e):
    with (torch.no_grad()):
        model.eval()
        print('--- Start valuating ---')
        val = 0
        acc = 0
        auc_value = 0
        aupr_value = 0
        recall = 0
        precision = 0
        hit = 0
        ndcg = 0
        for val_data in [val_data_all]:
            val += 1
            global pred_val
            _, pred_val, te_embed_pos, graph_embed_neg_ls, te_summary, pred_list = model(diet_fea, mic_fea,
                dis_fea, hg_pos,val_data[:, 0], val_data[:, 1], val_data[:, 2], hg_neg_ls)
            hits_1, ndcg_1 = hit_ndcg_value(pred_val, val_data, args.top_1)
            hits_3, ndcg_3 = hit_ndcg_value(pred_val, val_data, args.top_3)
            hits_5, ndcg_5 = hit_ndcg_value(pred_val, val_data, args.top_5)
            global sorted_pred_list
            sorted_pred_list = sorted(pred_list, key=lambda x: x[0], reverse=True)
            ealy_stop(val, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5)
            real_scores = []
            for item in val_data:
                real_scores.append(item[-1])
            real_scores = np.array(real_scores)
            predictions_np = pred_val.numpy()
            fpr, tpr, thresholds = roc_curve(real_scores, predictions_np)
            auc_value += auc(fpr, tpr)
            precision1, recall1, _ = precision_recall_curve(real_scores, predictions_np)
            aupr_value += auc(recall1, precision1)
            binary_predictions = (predictions_np >= 0.5).astype(int)
            binary_predictions_tensor = torch.tensor(binary_predictions)
            TP = np.sum(np.logical_and(binary_predictions == 1, real_scores == 1))
            FP = np.sum(np.logical_and(binary_predictions == 1, real_scores == 0))
            FN = np.sum(np.logical_and(binary_predictions == 0, real_scores == 1))
            TN = np.sum(np.logical_and(binary_predictions == 0, real_scores == 0))
            acc += (TN+TP)/(TP+FP+FN+TN)
            precision += TP / (TP + FP)
            recall += TP / (TP + FN)
            hit += hits_3
            ndcg += ndcg_3
        hit = hit
        ndcg = ndcg
        auc_value = auc_value
        aupr_value = aupr_value
        acc = acc
        precision = precision
        recall = recall
        f1_val = (2 * precision * recall) / (precision + recall)
        print('hits:', hit)
        print('ndcg:', ndcg)
        print("AUC:", auc_value)
        print("AUPR:", aupr_value)
        print("Accuracy:", acc)
        print("F1 Score:", f1_val)
        print("Precision:", precision)
        print("Recall:", recall)
        return sorted_pred_list, hit, ndcg, auc_value, aupr_value, acc, f1_val, precision, recall, real_scores, pred_val
def ealy_stop(val, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5):
    if hits_1 >= hits_max_matrix[val-1][0]:
        hits_max_matrix[val-1][0] = hits_1
        ndcg_max_matrix[val-1][0] = ndcg_1
        hits_max_matrix[val-1][1] = hits_3
        ndcg_max_matrix[val-1][1] = ndcg_3
        hits_max_matrix[val-1][2] = hits_5
        ndcg_max_matrix[val-1][2] = ndcg_5
        epoch_max_matrix[0][val-1] = e + 1
        patience_num_matrix[0][val-1] = 0
    else:
        patience_num_matrix[0][val-1] += 1
if __name__ == '__main__':
    args = parameters_set()
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    patience_num_matrix = np.zeros((1, 4))
    epoch_max_matrix = np.zeros((1, 4))
    hits_max_matrix = np.zeros((4, 3))
    ndcg_max_matrix = np.zeros((4, 3))

    patience = 300

    dis_sim = np.loadtxt('Data/diseases_features.txt', delimiter=' ')
    mic_sim = np.loadtxt('Data/microbiome_features.txt', delimiter=' ')
    diet_sim = np.loadtxt('Data/dietarys_features.txt', delimiter=' ')
    dis_input = torch.from_numpy(dis_sim).type(torch.FloatTensor)
    mic_input = torch.from_numpy(mic_sim).type(torch.FloatTensor)
    diet_input = torch.from_numpy(diet_sim).type(torch.FloatTensor)

    adj_data = np.loadtxt('Data/adj.txt')
    np.random.shuffle(adj_data)
    test_data = adj_data[:int(0.1 * len(adj_data)), :]
    train_cv_data = adj_data[int(0.1 * len(adj_data)):, :]

    train_data_all, train_neg_all, test_data_all = \
        neg_data_generate(adj_data, train_cv_data, test_data,args.seed)

    train_data_pos = series_num(train_cv_data.copy().astype(int))
    train_data_all = series_num(np.array(train_data_all).copy().astype(int))

    hypergraph_pos = build_hypergraph(train_data_pos)
    hypergraph_neg_1 = build_hypergraph(train_neg_all)
    hypergraph_neg_ls = [hypergraph_neg_1]

    val_data_1 = series_num(np.array(test_data_all).copy().astype(int))

    model = HGNN(BioEncoder(diet_sim.shape[0], mic_sim.shape[0], dis_sim.shape[0], args.bio_out_dim),
                 HgnnEncoder(args.bio_out_dim, args.hgnn_dim_1),
                 Decoder(((args.hgnn_dim_1) // 4) * 3), (args.hgnn_dim_1 // 4))
    my_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    epochs = []
    hits_list = []
    ndcg_list = []
    auc_list = []
    aupr_list = []
    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    epoch_prediction = {}
    epoch_loss_list = []
    with open(predictFileName, 'a') as result_file:
        epoch_prediction = {}
        for e in range(args.epochs):
            epoch_loss = train(diet_input, mic_input, dis_input, hypergraph_pos, hypergraph_neg_ls, train_data_all)
            sorted_pred_list, hit, ndcg, auc_value, aupr_value, acc_val, f1_val, precision, recall, real_scores, pred_val = test(diet_input, mic_input, dis_input, hypergraph_pos, hypergraph_neg_ls, val_data_1, e)
            epoch_prediction[e+1] =sorted_pred_list
            epochs.append(e)
            epoch_loss_list.append(epoch_loss)
            hits_list.append(hit)
            ndcg_list.append(ndcg)
            auc_list.append(auc_value)
            aupr_list.append(aupr_value)
            acc_list.append(acc_val)

            f1_list.append(f1_val)
            recall_list.append(recall)
            precision_list.append(precision)
            if e == 1999:
                real_scores_np = np.array(real_scores)
                # 将预测值转换为NumPy数组
                predictions_np = pred_val.numpy()
                fpr, tpr, thresholds = roc_curve(real_scores_np, predictions_np)
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic Example')
                plt.legend(loc="lower right")
                plt.show()
            if patience_num_matrix[0][0] >= patience and patience_num_matrix[0][1] >= patience and \
                    patience_num_matrix[0][2] >= patience and patience_num_matrix[0][3] >= patience:
                break
        print('success')
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_loss_list, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, hits_list, label='Hits@3')
        plt.plot(epochs, ndcg_list, label='NDCG@3')
        plt.plot(epochs, auc_list, label='AUC')
        plt.plot(epochs, aupr_list, label='AUPR')
        plt.plot(epochs, acc_list, label='Acc')
        plt.plot(epochs, f1_list, label='F1 Score')
        plt.plot(epochs, recall_list, label='Recall')
        plt.plot(epochs, precision_list, label='Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics Value')
        plt.title('indep_results')
        plt.legend()
        plt.grid(True)
        plt.show()
        for keys,values in epoch_prediction.items():
            if keys in [epoch_max_matrix[0][0], epoch_max_matrix[0][1],
                        epoch_max_matrix[0][2], epoch_max_matrix[0][3]]:
                result_file.write("Epoch {} Predictions:\n".format(keys))
                for item in values:
                    result_file.write("{}: {}\n".format(item[0], item[1]))
                result_file.write("\n")
        write_type_1234(resultFileName, 'indep', hits_max_matrix[0][0], ndcg_max_matrix[0][0], hits_max_matrix[1][0],
                        ndcg_max_matrix[1][0], hits_max_matrix[2][0], ndcg_max_matrix[2][0], hits_max_matrix[3][0],
                        ndcg_max_matrix[3][0], epoch_max_matrix[0][0], epoch_max_matrix[0][1],
                        epoch_max_matrix[0][2], epoch_max_matrix[0][3], top='top1')
        write_type_1234(resultFileName, 'indep', hits_max_matrix[0][1], ndcg_max_matrix[0][1], hits_max_matrix[1][1],
                        ndcg_max_matrix[1][1], hits_max_matrix[2][1], ndcg_max_matrix[2][1], hits_max_matrix[3][1],
                        ndcg_max_matrix[3][1], epoch_max_matrix[0][0], epoch_max_matrix[0][1],
                        epoch_max_matrix[0][2], epoch_max_matrix[0][3], top='top3')
        write_type_1234(resultFileName, 'indep', hits_max_matrix[0][2], ndcg_max_matrix[0][2], hits_max_matrix[1][2],
                        ndcg_max_matrix[1][2], hits_max_matrix[2][2], ndcg_max_matrix[2][2], hits_max_matrix[3][2],
                        ndcg_max_matrix[3][2], epoch_max_matrix[0][0], epoch_max_matrix[0][1],
                        epoch_max_matrix[0][2], epoch_max_matrix[0][3], top='top5')
        with open('Data/Metrics_results.txt', "w") as file:
            file.write("Hits:{}\n".format(hits_list[-1]))
            file.write("NDCG:{}\n".format(ndcg_list[-1]))
            file.write("AUC: {}\n".format((auc_list[-1])))
            file.write("AUPR: {}\n".format((aupr_list[-1])))
            file.write("Acc: {}\n".format((acc_list[-1])))
            file.write("F1 Score: {}\n".format((f1_list[-1])))
            file.write("Precision: {}\n".format((precision_list[-1])))
            file.write("Recall: {}\n".format((recall_list[-1])))
        print("Results written to", 'Data/Metrics_results.txt')
        pred_list = []
        for i in range(len(pred_val)):
            if pred_val[i] >= 0.5:
                pred_list.append(1)
            else:
                pred_list.append(0)
        cm = confusion_matrix(real_scores, pred_list)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        report = classification_report(real_scores, pred_list)
        print(report)
