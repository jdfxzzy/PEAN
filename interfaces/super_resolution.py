import torch
import sys
import time
import os
import csv
from datetime import datetime
import copy
import torch.nn as nn
sys.path.append('../')
sys.path.append('./')
from interfaces import base
from utils.metrics import get_str_list
from utils.util import str_filt


class TextSR(base.TextBase):

    def train(self):
        cfg = self.config.TRAIN                   
        _, train_loader = self.get_train_data()
        _, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']

        parseq = self.PARSeq_init()
        parseq.eval()
        for p in parseq.parameters():
            p.requires_grad = False
        self.diffusion = self.init_diffusion_model()

        optimizer_G = self.optimizer_init([model, self.diffusion.netG])  

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
            
        best_history_acc = dict(easy_aster=0, medium_aster=0, hard_aster=0,
                                easy_crnn=0, medium_crnn=0, hard_crnn=0,
                                easy_moran=0, medium_moran=0, hard_moran=0)        
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc_aster = 0
        best_acc_crnn = 0
        best_acc_moran = 0
        converge_list = []
        log_path = os.path.join(cfg.ckpt_dir, "log.csv")

        print('='*110)
        display = True
        ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
        for epoch in range(cfg.epochs):
            for j, data in (enumerate(train_loader)):
                if display:
                    start = time.time()
                    display = False
                model.train()

                for p in model.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j + 1 
                images_hr, images_lr, labels, label_vecs, weighted_mask, weighted_tics = data
                
                text_label = label_vecs
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                prob_str_hr = []
                batch_size = images_hr.shape[0]
                for i in range(batch_size):
                    parseq_input = self.parse_parseq_data(images_hr[i, :3, :, :])
                    parseq_output = parseq(parseq_input)
                    pred = parseq_output.softmax(-1)
                    prob_str_hr.append(pred)
                prob_str_hr = torch.cat(prob_str_hr, dim=0)

                if not self.args.pre_training:
                    prob_str_lr = []
                    batch_size = images_lr.shape[0]
                    for i in range(batch_size):
                        parseq_input = self.parse_parseq_data(images_lr[i, :3, :, :])
                        parseq_output = parseq(parseq_input)
                        pred = parseq_output.softmax(-1)
                        prob_str_lr.append(pred)
                    prob_str_lr = torch.cat(prob_str_lr, dim=0)

                    text_sum = text_label.sum(1).squeeze(1)
                    text_pos = (text_sum > 0).float().sum(1)
                    text_len = text_pos.reshape(-1)
                    predicted_length = torch.ones(prob_str_lr.shape[0]) * prob_str_lr.shape[1]

                    data_diff = {"HR":prob_str_hr, "SR":prob_str_lr, "weighted_mask": weighted_mask, "predicted_length": predicted_length, "text_len": text_len}
                    self.diffusion.feed_data(data_diff)
                    loss_diff, label_vecs_final = self.diffusion.process()
                    label_vecs_final = label_vecs_final.to(self.device)
                else:
                    label_vecs_final = prob_str_hr
                    loss_diff = 0

                images_sr, logits = model(images_lr, label_vecs_final)

                loss_im = 0
                loss_im = loss_im + image_crit(images_sr, images_hr, labels) * 100
                aux_rec_loss = 0
                loss_aux_rec_module = 0
                aux_rec_loss += ctc_loss(logits.log_softmax(2),
                                         weighted_mask.long().to(self.device),
                                         predicted_length.long().to(self.device),
                                         text_len.long())
                aux_rec_loss_each = (aux_rec_loss * weighted_tics.float().to(self.device))
                aux_rec_loss_each = aux_rec_loss_each.mean()
                loss_aux_rec_module += aux_rec_loss_each
                loss_im = loss_im + loss_aux_rec_module
                loss_im = loss_im + loss_diff

                optimizer_G.zero_grad()
                loss_im.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()

                if iters % cfg.displayInterval == 0:
                    end = time.time()
                    duration = end - start
                    display = True
                    print('[{}] | '
                          'Epoch: [{}][{} / {}] | '
                          'Loss: {} | Duration: {}s'
                            .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  float(loss_im.data), duration))
                    print('-'*110)

                if iters % cfg.VAL.valInterval == 0:
                    print('='*110)
                    current_acc_dict = {}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('\\')[-1]
                        print('evaling %s' % data_name)
                        metrics_dict = self.eval(model, val_loader, epoch, parseq)
                        converge_list.append({'iterator': iters,
                                              'acc_aster': metrics_dict['accuracy_aster'],
                                              'acc_crnn': metrics_dict['accuracy_crnn'],
                                              'acc_moran': metrics_dict['accuracy_moran'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc_aster = metrics_dict['accuracy_aster']
                        acc_crnn = metrics_dict['accuracy_crnn']
                        acc_moran = metrics_dict['accuracy_moran']
                        current_acc_dict[data_name+'_aster'] = float(acc_aster)
                        current_acc_dict[data_name+'_crnn'] = float(acc_crnn)
                        current_acc_dict[data_name+'_moran'] = float(acc_moran)
                        
                        if acc_aster > best_history_acc[data_name+'_aster']:
                            best_history_acc[data_name+'_aster'] = float(acc_aster)
                            best_history_acc['epoch'] = epoch
                            best_model_info = {'accuracy_aster': float(acc_aster), 'psnr': metrics_dict['psnr_avg'], 'ssim': metrics_dict['ssim_avg']}
                            print('best_%s = %.2f%% (A New Record)' % (data_name+'_aster', best_history_acc[data_name+'_aster'] * 100))
                            self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, data_name+'_aster')
                            self.diffusion.save_network(epoch, j, "aster")
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_aster', metrics_dict['accuracy_aster'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg'], "best_{}".format(data_name+'_aster')])
                        else:
                            print('best_%s_aster = %.2f%%' % (data_name, best_history_acc[data_name+'_aster'] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_aster', metrics_dict['accuracy_aster'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg']])

                        if acc_crnn > best_history_acc[data_name+'_crnn']:
                            best_history_acc[data_name+'_crnn'] = float(acc_crnn)
                            best_history_acc['epoch'] = epoch
                            best_model_info = {'accuracy_crnn': float(acc_crnn), 'psnr': metrics_dict['psnr_avg'], 'ssim': metrics_dict['ssim_avg']}
                            print('best_%s = %.2f%% (A New Record)' % (data_name+'_crnn', best_history_acc[data_name+'_crnn'] * 100))
                            self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, data_name+'_crnn')
                            self.diffusion.save_network(epoch, j, "crnn")
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_crnn', metrics_dict['accuracy_crnn'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg'], "best_{}".format(data_name+'_crnn')])
                        else:
                            print('best_%s_crnn = %.2f%%' % (data_name, best_history_acc[data_name+'_crnn'] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_crnn', metrics_dict['accuracy_crnn'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg']])
                        
                        if acc_moran > best_history_acc[data_name+'_moran']:
                            best_history_acc[data_name+'_moran'] = float(acc_moran)
                            best_history_acc['epoch'] = epoch
                            best_model_info = {'accuracy_moran': float(acc_moran), 'psnr': metrics_dict['psnr_avg'], 'ssim': metrics_dict['ssim_avg']}
                            print('best_%s = %.2f%% (A New Record)' % (data_name+'_moran', best_history_acc[data_name+'_moran'] * 100))
                            self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, data_name+'_moran')
                            self.diffusion.save_network(epoch, j, "moran")
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_moran', metrics_dict['accuracy_moran'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg'], "best_{}".format(data_name+'_moran')])
                        else:
                            print('best_%s_moran = %.2f%%' % (data_name, best_history_acc[data_name+'_moran'] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name+'_moran', metrics_dict['accuracy_moran'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg']])
                        print('-'*110)
                    if (current_acc_dict['easy_aster'] * 1619 + current_acc_dict['medium_aster'] * 1411 + current_acc_dict['hard_aster'] * 1343) / (1619 + 1411 + 1343) > best_acc_aster:       # 三个测试集识别准确率的和大于历史最好时
                        best_acc_aster = (current_acc_dict['easy_aster'] * 1619 + current_acc_dict['medium_aster'] * 1411 + current_acc_dict['hard_aster'] * 1343) / (1619 + 1411 + 1343)
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model for aster')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, 'sum_aster')
                        self.diffusion.save_network(epoch, j, "sum_aster")
                        with open(log_path, "a+", newline="") as out:
                            writer = csv.writer(out)
                            writer.writerow([epoch, "", "", "", "", "", "best_sum_aster"])
                    if (current_acc_dict['easy_crnn'] * 1619 + current_acc_dict['medium_crnn'] * 1411 + current_acc_dict['hard_crnn'] * 1343) / (1619 + 1411 + 1343) > best_acc_crnn:       # 三个测试集识别准确率的和大于历史最好时
                        best_acc_crnn = (current_acc_dict['easy_crnn'] * 1619 + current_acc_dict['medium_crnn'] * 1411 + current_acc_dict['hard_crnn'] * 1343) / (1619 + 1411 + 1343)
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model for crnn')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, 'sum_crnn')
                        self.diffusion.save_network(epoch, j, "sum_crnn")
                        with open(log_path, "a+", newline="") as out:
                            writer = csv.writer(out)
                            writer.writerow([epoch, "", "", "", "", "", "best_sum_crnn"])
                    if (current_acc_dict['easy_moran'] * 1619 + current_acc_dict['medium_moran'] * 1411 + current_acc_dict['hard_moran'] * 1343) / (1619 + 1411 + 1343) > best_acc_moran:       # 三个测试集识别准确率的和大于历史最好时
                        best_acc_moran = (current_acc_dict['easy_moran'] * 1619 + current_acc_dict['medium_moran'] * 1411 + current_acc_dict['hard_moran'] * 1343) / (1619 + 1411 + 1343)
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model for moran')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, 'sum_moran')
                        self.diffusion.save_network(epoch, j, "sum_moran")
                        with open(log_path, "a+", newline="") as out:
                            writer = csv.writer(out)
                            writer.writerow([epoch, "", "", "", "", "", "best_sum_moran"])
                    print('='*110)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list)
                    self.diffusion.save_network("ckpt", "ckpt", "ckpt")

    def eval(self, model, val_loader, index, parseq):
        for p in model.parameters():
            p.requires_grad = False

        aster, aster_info = self.Aster_init()
        aster.eval()
        for p in aster.parameters():
            p.requires_grad = False
        crnn = self.CRNN_init()
        crnn.eval()
        for p in crnn.parameters():
            p.requires_grad = False
        moran = self.MORAN_init()
        moran.eval()
        for p in moran.parameters():
            p.requires_grad = False

        model.eval()
        n_correct_aster = 0
        n_correct_crnn = 0
        n_correct_moran = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        for _, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs, _ = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            prob_str_lr = []
            for i in range(images_lr.shape[0]):
                parseq_input = self.parse_parseq_data(images_lr[i, :3, :, :])
                parseq_output = parseq(parseq_input)
                pred = parseq_output.softmax(-1)
                prob_str_lr.append(pred)
            prob_str_lr = torch.cat(prob_str_lr, dim=0)

            if not self.args.pre_training:
                label_vecs_final_new = None
                for j in range(val_batch_size):
                    data_diff = {"SR": prob_str_lr[j, :, :].unsqueeze(0)}
                    self.diffusion.feed_data(data_diff)
                    self.diffusion.test()
                    visuals = self.diffusion.get_current_visuals()
                    prior = visuals['SR']
                    if label_vecs_final_new is None:
                        label_vecs_final_new = prior
                    else:
                        label_vecs_final_new = torch.concat([label_vecs_final_new, prior], dim=0)
                label_vecs_final_new = label_vecs_final_new.to(self.device)
            else:
                prob_str_hr = []
                for j in range(images_hr.shape[0]):
                    parseq_input = self.parse_parseq_data(images_hr[j, :3, :, :])
                    parseq_output = parseq(parseq_input)
                    pred = parseq_output.softmax(-1)
                    prob_str_hr.append(pred)
                label_vecs_final_new = torch.cat(prob_str_hr, dim=0)

            images_sr, _ = model(images_lr, label_vecs_final_new)

            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
            aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
            aster_output_lr = aster(aster_dict_lr)
            aster_output_sr = aster(aster_dict_sr)
            pred_rec_lr = aster_output_lr['output']['pred_rec']
            pred_rec_sr = aster_output_sr['output']['pred_rec']
            pred_str_lr_aster, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            pred_str_sr_aster, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

            crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
            crnn_output, _ = crnn(crnn_input)
            _, preds = crnn_output.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
            pred_str_sr_crnn = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            
            moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
            moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True, debug=True)
            preds, _ = moran_output[0]
            _, preds = preds.max(1)
            sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
            pred_str_sr_moran = [pred.split('$')[0] for pred in sim_preds]
            for pred, target in zip(pred_str_sr_aster, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct_aster += 1
            for pred, target in zip(pred_str_sr_crnn, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct_crnn += 1
            for pred, target in zip(pred_str_sr_moran, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct_moran += 1

            sum_images += val_batch_size
            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        print('[{}] | '
              'PSNR {:.2f} | SSIM {:.4f}'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg), float(ssim_avg)))
        print('save display images')

        self.tripple_display(images_lr, images_sr[0], images_hr, pred_str_lr_aster, pred_str_sr_aster, label_strs, index)
        accuracy_aster = round(n_correct_aster / sum_images, 4)
        accuracy_crnn = round(n_correct_crnn / sum_images, 4)
        accuracy_moran = round(n_correct_moran / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        print('aster_accuray: %.2f%% | crnn_accuray: %.2f%% | moran_accuray: %.2f%% ' % (accuracy_aster * 100, accuracy_crnn * 100, accuracy_moran * 100))
        metric_dict['accuracy_aster'] = accuracy_aster
        metric_dict['accuracy_crnn'] = accuracy_crnn
        metric_dict['accuracy_moran'] = accuracy_moran
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg
        return metric_dict

    def test(self):
        model_dict = self.generator_init()
        model, _ = model_dict['model'], model_dict['crit']
        if not self.args.pre_training:
            self.diffusion = self.init_diffusion_model()
            self.diffusion.netG.eval()
            for p in self.diffusion.netG.parameters():
                p.requires_grad = False
        _, test_loader = self.get_test_data(self.test_data_dir)
        data_name = self.args.test_data_dir.split('\\')[-1]
        print('evaling %s' % data_name)
        parseq = self.PARSeq_init()
        for p in parseq.parameters():
            p.requires_grad = False
        parseq.eval()
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        print('='*110)
        
        for i, data in (enumerate(test_loader)):
            images_hr, images_lr, label_strs, _ = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            prob_str_lr = []
            for j in range(images_lr.shape[0]):
                parseq_input = self.parse_parseq_data(images_lr[j, :3, :, :])
                parseq_output = parseq(parseq_input)
                pred = parseq_output.softmax(-1)
                prob_str_lr.append(pred)
            prob_str_lr = torch.cat(prob_str_lr, dim=0)

            if not self.args.pre_training:
                label_vecs_final_new = None
                for j in range(val_batch_size):
                    data_diff = {"SR": prob_str_lr[j, :, :].unsqueeze(0)}
                    self.diffusion.feed_data(data_diff)
                    self.diffusion.test()
                    visuals = self.diffusion.get_current_visuals()
                    prior = visuals['SR']
                    if label_vecs_final_new is None:
                        label_vecs_final_new = prior
                    else:
                        label_vecs_final_new = torch.concat([label_vecs_final_new, prior], dim=0)
                label_vecs_final_new = label_vecs_final_new.to(self.device)
            else:
                prob_str_hr = []
                for j in range(images_hr.shape[0]):
                    parseq_input = self.parse_parseq_data(images_hr[j, :3, :, :])
                    parseq_output = parseq(parseq_input)
                    pred = parseq_output.softmax(-1)
                    prob_str_hr.append(pred)
                label_vecs_final_new = torch.cat(prob_str_hr, dim=0)
            
            images_sr, _ = model(images_lr, label_vecs_final_new)
            
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True, debug=True)
                preds, _ = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output, _ = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            for j in range(val_batch_size):
                if str_filt(pred_str_sr[j], 'lower') == str_filt(label_strs[j], 'lower'):
                    n_correct += 1
                    
            sum_images += val_batch_size
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{} / {}]'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader)))
            
        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        fps = sum_images/(time_end - time_begin)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
        print(result)
        print('='*110)