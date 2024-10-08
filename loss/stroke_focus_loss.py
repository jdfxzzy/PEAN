import torch
import numpy as np
import torch.nn as nn
from loss.transformer_english_decomposition import Transformer

def to_gray_tensor(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 1:2, :, :]
    B = tensor[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


class StrokeFocusLoss(nn.Module):
    def __init__(self, args):
        super(StrokeFocusLoss, self).__init__()
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.english_stroke_alphabet = '0123456789'
        self.english_stroke_dict = {}
        for index in range(len(self.english_stroke_alphabet)):
            self.english_stroke_dict[self.english_stroke_alphabet[index]] = index

        stroke_decompose_lines = open('./english_decomposition.txt', 'r').readlines()
        self.dic = {}
        for line in stroke_decompose_lines:
            line = line.strip()
            character, sequence = line.split()
            self.dic[character] = sequence

        self.build_up_transformer()

    def build_up_transformer(self):
        transformer = Transformer().cuda()
        transformer = nn.DataParallel(transformer, device_ids=[0])
        transformer.load_state_dict(torch.load('./pretrain_transformer_stroke_decomposition.pth'))
        transformer.eval()
        for p in transformer.parameters():
            p.requires_grad = False
        self.transformer = transformer

    def label_stroke_encoder(self, label):
        new_label_list = []
        for one_label in label:
            stroke_sequence = ''
            for character in one_label:
                if character not in self.dic:
                    continue
                else:
                    stroke_sequence += self.dic[character]
            stroke_sequence += '0'
            new_label_list.append(stroke_sequence)
        label = new_label_list

        batch = len(label)

        length = [len(i) for i in label]
        length_tensor = torch.Tensor(length).long().cuda()

        max_length = max(length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                input_tensor[i][j + 1] = self.english_stroke_dict[label[i][j]]

        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_stroke_dict[j])
        text_gt = torch.Tensor(text_gt).long().cuda()

        input_tensor = torch.from_numpy(input_tensor).long().cuda()
        return length_tensor, input_tensor, text_gt

    def forward(self,sr_img, hr_img, label):
        mse_loss = self.mse_loss(sr_img, hr_img)
        length_tensor, input_tensor, _ = self.label_stroke_encoder(label)
        _, word_attention_map_gt, hr_correct_list = self.transformer(to_gray_tensor(hr_img), length_tensor, input_tensor, test=False)
        _, word_attention_map_pred, sr_correct_list = self.transformer(to_gray_tensor(sr_img), length_tensor, input_tensor, test=False)

        correct_flag = False
        correct_list = []
        for i in range(len(hr_correct_list)):
            if hr_correct_list[i] and sr_correct_list[i]:
                correct_list.append(True)
            else:
                correct_list.append(False)
            
        if len(correct_list) == 0 and correct_flag:
            attention_loss = 0
        else:
            if correct_flag:
                correct_list = torch.Tensor(correct_list)
                word_attention_map_gt = word_attention_map_gt[correct_list==True]
                word_attention_map_pred = word_attention_map_pred[correct_list==True]
            attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)

        loss = 0.8 * mse_loss + attention_loss * 75
        return loss
        