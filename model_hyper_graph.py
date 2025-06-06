import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from Utils.utils_ import *

EPS = torch.tensor(1e-15)


class HgnnEncoder(nn.Module):
    def __init__(self, in_channels, dim_1):
        super(HgnnEncoder, self).__init__()
        self.dropout = nn.Dropout(0.15)
        self.conv1 = HypergraphConv(in_channels, dim_1)
        self.conv2 = HypergraphConv(dim_1, dim_1 // 2)
        self.conv3 = HypergraphConv(dim_1 // 2, dim_1 // 4)
    def forward(self, x, edge):
        x = self.dropout(x)
        x = torch.relu(self.conv1(x, edge))
        x = torch.relu(self.conv2(x, edge))
        x = torch.relu(self.conv3(x, edge))
        return x


class BioEncoder(nn.Module):
    def __init__(self, dim_diet, dim_mic, dim_dis, output):
        super(BioEncoder, self).__init__()
        self.dis_layer1 = nn.Linear(dim_dis, output)
        self.batch_dis1 = nn.BatchNorm1d(output)
        self.mic_layer1 = nn.Linear(dim_mic, output)
        self.batch_mic1 = nn.BatchNorm1d(output)
        self.diet_layer1 = nn.Linear(dim_diet, output)
        self.batch_diet1 = nn.BatchNorm1d(output)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.15)
        # self.conv1 = GINConv(nn.Sequential(nn.Linear(78, output),
        #                                    nn.ReLU(),
        #                                    nn.Linear(output, output)))
        # self.bn1 = torch.nn.BatchNorm1d(output)
        #
        # self.conv2 = GINConv(nn.Sequential(nn.Linear(output, output),
        #                                    nn.ReLU(),
        #                                    nn.Linear(output, output)))
        # self.bn2 = torch.nn.BatchNorm1d(output)
        #
        # self.conv3 = GINConv(nn.Sequential(nn.Linear(output, output),
        #                                    nn.ReLU(),
        #                                    nn.Linear(output, output)))
        self.bn3 = torch.nn.BatchNorm1d(output)
        self.fc1_xd = nn.Linear(output, output)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, diet_feature, mic_feature, dis_feature):
        # -----drug_train
        x_diet = self.diet_layer1(diet_feature)
        x_diet = self.batch_diet1(F.relu(x_diet))
        x_diet = self.drop_out(x_diet)
        x_dis = self.dis_layer1(dis_feature)
        x_dis = self.batch_dis1(F.relu(x_dis))
        x_dis = self.drop_out(x_dis)
        x_mic = self.mic_layer1(mic_feature)
        x_mic = self.batch_mic1(F.relu(x_mic))
        x_mic = self.drop_out(x_mic)
        return x_diet, x_mic, x_dis


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc23 = nn.Linear(in_channels // 4, in_channels // 8)
        self.batch23 = nn.BatchNorm1d(in_channels // 8)
        self.fc3 = nn.Linear(in_channels // 8, 1)
        self.dropout = nn.Dropout(0.15)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, diet_id, mic_id, dis_id):
        h_0 = torch.cat((graph_embed[diet_id, :], graph_embed[mic_id, :], graph_embed[dis_id, :]), 1)
        h_1 = torch.tanh(self.fc1(h_0))
        h_1 = self.batch1(h_1)
        h_1 = self.dropout(h_1)
        h_2 = torch.tanh(self.fc2(h_1))
        h_2 = self.batch2(h_2)
        h_2 = self.dropout(h_2)
        h_23 = torch.tanh(self.fc23(h_2))
        h_23 = self.batch23(h_23)
        h_23 = self.dropout(h_23)
        h_3 = self.fc3(h_23)
        return h_23, torch.sigmoid(h_3.squeeze(dim=1))


class HGNN(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder, out_channels):  #
        super(HGNN, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.weight = nn.Parameter(torch.Tensor(out_channels, out_channels))  # hidden_channels
        nn.init.xavier_uniform_(self.weight)
        # self.reset_parameters()

    def reset_parameters(self):
        # reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.decoder)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def DGI_loss(self, pos_z, summary, *args):
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(self.discriminate(pos_z, summary) + EPS).mean()
        neg_loss = []
        for i in args[0]:
            neg_loss_1 = -torch.log(1 - self.discriminate(i, summary) + EPS).mean()
            neg_loss.append(neg_loss_1)
        loss = pos_loss
        for i in neg_loss:
            loss += i
        return loss

    def forward(self, diet_feature, mic_feature, dis_feature, edge_pos, diet_id, mic_id, dis_id, *args):
        x_diet, x_mic, x_dis = self.bio_encoder(diet_feature, mic_feature, dis_feature)
        embed = torch.cat((x_diet, x_mic, x_dis), 0)
        graph_embed_pos = self.graph_encoder(embed, edge_pos)
        graph_embed_neg_ls = []
        for i in args[0]:
            graph_embed_neg = self.graph_encoder(embed, i)
            graph_embed_neg_ls.append(graph_embed_neg)
        summary = torch.sigmoid(graph_embed_pos.mean(dim=0))
        emb, res = self.decoder(graph_embed_pos, diet_id, mic_id, dis_id)
        graph_embed_pos = torch.sigmoid(graph_embed_pos)
        for i in range(len(graph_embed_neg_ls)):
            graph_embed_neg_ls[i] = torch.sigmoid(graph_embed_neg_ls[i])
        pred_list = []
        for pred, diet, mic, dis in zip(res, diet_id, mic_id, dis_id):
            pred_list.append((pred.item(), (diet, mic, dis)))
        return emb, res, graph_embed_pos, graph_embed_neg_ls, summary, pred_list
