import os
import torch
import glob
import numpy as np
import imageio
import cv2
import math
import time
import argparse
from model.cdvd_tsp import CDVD_TSP
from pprint import pprint

'''
Trying out replacing Inference.predict() to something else, but STOPPED PURSUING
'''

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')


class Inference:
    def __init__(self, args):

        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.result_path = args.result_path
        self.n_seq = 5
        self.size_must_mode = 4
        self.device = args.device

        if not os.path.exists(self.result_path):
            # os.mkdir(self.result_path)
            os.makedirs(self.result_path, exist_ok=True)
            print('mkdir: {}'.format(self.result_path))

        if args.default_data == 'DVD':
            self.input_path = os.path.join(self.data_path, "input")
            self.GT_path = os.path.join(self.data_path, "GT")
        else:
            self.input_path = os.path.join(self.data_path, "blur")
            self.GT_path = os.path.join(self.data_path, "gt")
        print (f'input_path: {self.input_path}')
        print (f'GT_path: {self.GT_path}')

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = CDVD_TSP(
            in_channels=3, n_sequence=5, out_channels=3, n_resblock=3, n_feat=32,
            is_mask_filter=True, device=self.device
        )
        self.net.load_state_dict(torch.load(self.model_path), strict=False)
        self.net = self.net.to(self.device)
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()
    
    # Predict a video sequence
    def predict(self):
        pass

    def infer_(self):
        with torch.no_grad():
            total_psnr = {}
            total_ssim = {}
            videos = sorted(os.listdir(self.data_path))
            # pprint (videos)
            # pprint (sorted(glob.glob(os.path.join(self.input_path, "*"))))
            # assert False
            for v in videos:
                if v == '.DS_Store':
                    continue
                video_psnr = []
                video_ssim = []
                input_frames = sorted(glob.glob(os.path.join(self.data_path, v, 'input', "*")))
                gt_frames = sorted(glob.glob(os.path.join(self.data_path, v, 'GT', "*")))
                # pprint (input_frames)
                # pprint (gt_frames)
                # assert False

                input_seqs = self.gene_seq(input_frames, n_seq=self.n_seq)
                gt_seqs = self.gene_seq(gt_frames, n_seq=self.n_seq)
                # pprint (input_seqs)
                # pprint (gt_seqs)
                # assert False


                for in_seq, gt_seq in zip(input_seqs, gt_seqs):
                    # pprint (in_seq)
                    # pprint (gt_seq)
                    # assert False

                    start_time = time.time()
                    filename = os.path.basename(in_seq[self.n_seq // 2]).split('.')[0]
                    inputs = [imageio.imread(p) for p in in_seq]
                    gt = imageio.imread(gt_seq[self.n_seq // 2])

                    h, w, c = inputs[self.n_seq // 2].shape
                    new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                    inputs = [im[:new_h, :new_w, :] for im in inputs]
                    gt = gt[:new_h, :new_w, :]

                    in_tensor = self.numpy2tensor(inputs).to(self.device)
                    preprocess_time = time.time()
                    _, output_stage1, _, output, _ = self.net(in_tensor)
                    forward_time = time.time()
                    output_img = self.tensor2numpy(output)

                    psnr, ssim = self.get_PSNR_SSIM(output_img, gt)
                    video_psnr.append(psnr)
                    video_ssim.append(ssim)
                    total_psnr[v] = video_psnr
                    total_ssim[v] = video_ssim

                    if self.save_image:
                        if not os.path.exists(os.path.join(self.result_path, v)):
                            os.mkdir(os.path.join(self.result_path, v))
                        imageio.imwrite(os.path.join(self.result_path, v, '{}.png'.format(filename)), output_img)
                    postprocess_time = time.time()

                    self.logger.write_log(
                        '> {}-{} PSNR={:.5}, SSIM={:.4} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                            .format(v, filename, psnr, ssim,
                                    preprocess_time - start_time,
                                    forward_time - preprocess_time,
                                    postprocess_time - forward_time,
                                    postprocess_time - start_time))

            sum_psnr = 0.
            sum_ssim = 0.
            n_img = 0
            for k in total_psnr.keys():
                self.logger.write_log("# Video:{} AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(
                    k, sum(total_psnr[k]) / len(total_psnr[k]), sum(total_ssim[k]) / len(total_ssim[k])))
                sum_psnr += sum(total_psnr[k])
                sum_ssim += sum(total_ssim[k])
                n_img += len(total_psnr[k])
            self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(sum_psnr / n_img, sum_ssim / n_img))
            assert False

    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[:half]
            img_list_temp.extend(img_list)
            img_list_temp.extend(img_list[-half:])
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list

    def numpy2tensor(self, input_seq, rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor)
        stacked = torch.stack(tensor_list).unsqueeze(0)
        return stacked

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]

if __name__ == '__main__':

    '''
    1)
    python inference.py --default_data DVD
        # --default_data: the dataset you want to test, optional: DVD, GOPRO

    Namespace(
        border=False, 
        data_path='../dataset/DVD/test', 
        default_data='DVD', 
        model_path='../pretrain_models/CDVD_TSP_DVD_Convergent.pt',
        result_path='../infer_results/DVD', 
        save_image=True
        )

    2)
    cd ./code
    python inference.py --data_path path/to/data --model_path path/to/pretrained/model
        # --data_path: the path of your dataset.
        # --model_path: the path of the downloaded pretrained model.

    Namespace(
        border=False, 
        data_path='../dataset/DVD/test', 
        default_data='.', 
        model_path='../pretrain_models/CDVD_TSP_DVD_Convergent.pt',
        result_path='../infer_results', 
        save_image=True
        )
    '''


    # parser = argparse.ArgumentParser(description='CDVD-TSP-Inference')

    # parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    # parser.add_argument('--border', action='store_true', help='restore border images of video if true')

    # parser.add_argument('--default_data', type=str, default='.',
    #                     help='quick test, optional: DVD, GOPRO')
    # parser.add_argument('--data_path', type=str, default='../dataset/DVD/test',
    #                     help='the path of test data')
    # parser.add_argument('--model_path', type=str, default='../pretrain_models/CDVD_TSP_DVD_Convergent.pt',
    #                     help='the path of pretrain model')
    # parser.add_argument('--result_path', type=str, default='../infer_results',
    #                     help='the path of deblur result')
    # args = parser.parse_args()

    # if args.default_data == 'DVD':
    #     args.data_path = '../dataset/DVD/test'
    #     args.model_path = '../pretrain_models/CDVD_TSP_DVD_Convergent.pt'
    #     args.result_path = '../infer_results/DVD'
    # elif args.default_data == 'GOPRO':
    #     args.data_path = '../dataset/GOPRO/test'
    #     args.model_path = '../pretrain_models/CDVD_TSP_GOPRO.pt'
    #     args.result_path = '../infer_results/GOPRO'
    
    # print (args)
    # assert False

    dataset_name = 'DVD'
    dataset_type = 'quantitative_datasets'
    dataset_sample = 'IMG_0200'

    args = AttrDict()
    args.border = False 
    args.data_path = f'/nethome/hkwon64/Datasets/public/DVD/DeepVideoDeblurring_Dataset/{dataset_type}' 
    args.default_data = dataset_name
    args.model_path = '/nethome/hkwon64/Research/imuTube/repos_v2/motion_blur/CDVD-TSP/pretrain_models/CDVD_TSP_DVD_Convergent.pt'
    args.result_path = f'/nethome/hkwon64/Research/imuTube/repos_v2/motion_blur/CDVD-TSP/infer_results/{dataset_name}/{dataset_type}' 
    args.save_image = True
    args.device = torch.device('cuda', 0)

    Infer = Inference(args)
    # assert False

    Infer.infer_()
