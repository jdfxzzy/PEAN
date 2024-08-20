import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
import string
from PIL import Image
import torchvision
from torchvision import transforms
from collections import OrderedDict

from model import pean
from model import recognizer
from model import moran
from model import crnn
from dataset import lmdbDataset, alignCollate_real, lmdbDataset_real, alignCollate_syn, lmdbDataset_mix, alignCollate_realWTLAMask, alignCollate_realWTL
from loss import stroke_focus_loss
from model.parseq.strhub.data.module import SceneTextDataModule
from utils.labelmaps import get_vocabulary

sys.path.append('../')
from utils import ssim_psnr, utils_moran, utils_crnn
import dataset.dataset as dataset
import model.tpem.core.logger as Logger
import model.tpem.model as Model


class TextBase(object):

    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        if self.args.syn:                               
            self.align_collate = alignCollate_syn 
            self.load_dataset = lmdbDataset
        elif self.args.mixed:                        
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_mix
        else:                                        
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real
            
            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real

        self.resume = args.demo_dir+args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {                                 
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type] 
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.mask = self.args.mask
        self.max_gen_perms = self.args.max_gen_perms
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')  
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

    def get_train_data(self):

        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(self.load_dataset(root=data_dir_, voc_type=cfg.voc_type, max_len=cfg.max_len))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers), pin_memory=True,
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=True)  
        
        return train_dataset, train_loader

    def get_val_data(self):

        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, _ = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            test_dataset = dataset.ConcatDataset(dataset_list)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers), pin_memory=True,
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=False)
        return dataset_list, test_loader

    def get_test_data(self, dir_):

        cfg = self.config.TRAIN
        self.args.test_data_dir
        test_dataset = self.load_dataset(root=dir_, voc_type=cfg.voc_type, max_len=cfg.max_len, test=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers), pin_memory=True,
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self, resume_this=''):
        cfg = self.config.TRAIN
        model = pean.PEAN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                            STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u, testing=self.args.test)
        image_crit = stroke_focus_loss.StrokeFocusLoss(self.args)
        
        model = model.to(self.device)
        image_crit.to(self.device)
        if cfg.ngpu > 1:
            model = torch.nn.DataParallel(model, device_ids=range(cfg.ngpu))
            image_crit = torch.nn.DataParallel(image_crit, device_ids=range(cfg.ngpu))
        if resume_this is not '':
            print('loading PEAN (w/o TPEM) from %s ' % resume_this)
            if self.config.TRAIN.ngpu == 1:
                model.load_state_dict(torch.load(resume_this))
            else:
                model.load_state_dict(
                    {'module.' + k: v for k, v in torch.load(resume_this).items()})
        elif self.resume is not '':
            print('loading PEAN (w/o TPEM) from %s ' % self.resume)
            if self.config.TRAIN.ngpu == 1:
                model.load_state_dict(torch.load(self.resume))
            else:
                model.load_state_dict(
                    {'module.' + k: v for k, v in torch.load(self.resume).items()})
        return {'model': model, 'crit': image_crit}

    def optimizer_init(self, model):
        cfg = self.config.TRAIN
        model_params = []
        if type(model) == list:
            for m in model:
                model_params += list(m.parameters())
        else:
            model_params += list(model.parameters())
        optimizer = optim.AdamW(model_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))
        return optimizer
    
    def init_diffusion_model(self):
        opt = Logger.parse()
        opt = Logger.dict_to_nonedict(opt)
        diffusion = Model.create_model(opt)
        diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
        return diffusion

    def tripple_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, index):
        for i in (range(min(image_in.shape[0], self.config.TRAIN.VAL.n_vis) )):

            tensor_in = image_in[i][:3,:,:]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:]
            tensor_target = image_target[i][:3,:,:]
            images = ([tensor_in, tensor_out.cpu(), tensor_target.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            out_root = os.path.join(self.vis_dir)
            if not os.path.exists(out_root):
                os.mkdir(out_root)
            out_path = os.path.join(out_root, str(index))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            torchvision.utils.save_image(vis_im, os.path.join(out_path, im_name), padding=0)

    def test_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, str_filt, out_root):
        visualized = 0
        visualized += 1
        tensor_in = image_in.cpu()
        tensor_out = image_out.cpu()
        tensor_target = image_target.cpu()
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
             transforms.ToTensor()])
        tensor_in = transform(tensor_in)
        images = ([tensor_in, tensor_out, tensor_target])
        vis_im = torch.stack(images)
        vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)

        if not os.path.exists(out_root):
            os.mkdir(out_root)
        if not os.path.exists(out_root):
            os.mkdir(out_root)
        im_name = pred_str_lr + '_' + pred_str_sr + '_' + label_strs + '_.png'
        im_name = im_name.replace('/', '')
        torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
        return visualized

    def save_checkpoint(self, netG, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, metric="sum"):
        ckpt_path = os.path.join(self.vis_dir, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        if is_best:
            torch.save(netG.state_dict(), os.path.join(ckpt_path, 'model_best_{}_{}.pth'.format(epoch, metric)))
        else:
            torch.save(netG.state_dict(), os.path.join(ckpt_path, 'checkpoint_{}_{}.pth'.format(epoch, iters)))

    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        if cfg.ngpu > 1:
            MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        model_path = self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model

    def parse_crnn_data(self, imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        print('loading pretrained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        if cfg.ngpu > 1:
            aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict

    def PARSeq_init(self, path=None):
        model = torch.hub.load('./model/parseq', 'parseq', pretrained=False, source='local').eval()
        model_path = self.config.TRAIN.VAL.parseq_pretrained if path is None else path
        print('loading pretrained parseq model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        return model

    def parse_parseq_data(self, imgs_input):
        img_transform = SceneTextDataModule.get_transform([32, 128])
        imgs_input = transforms.ToPILImage()(imgs_input).convert('RGB')
        imgs = img_transform(imgs_input).unsqueeze(0)
        imgs = imgs.to(self.device)
        return imgs


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
