import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
from decord import VideoReader, cpu
import json
import pathlib
from pathlib import Path
from torchvision import transforms
from config import rgb_dirs, pose_dirs, description_dirs
from temporal_alignment import DescriptionLoader, TemporalAligner

# load sub-pose
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]

            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[...,2]<=thr] = 0
            
        kps_with_scores[part] = torch.tensor(result)
        
    return kps_with_scores


# input: T, N, 3
# input is un-normed joints
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]


# bbox of hands
def bbox_4hands(left_keypoints, right_keypoints, hw):
    # keypoints --> T,21,2
    # keypoints --> T,21,2
    
    def compute_bbox(keypoints):
        min_x = np.min(keypoints[..., 0], axis=1)
        min_y = np.min(keypoints[..., 1], axis=1)
        max_x = np.max(keypoints[..., 0], axis=1)
        max_y = np.max(keypoints[..., 1], axis=1)
        
        return (max_x+min_x)/2, (max_y+min_y)/2, (max_x-min_x), (max_y-min_y)
    H,W = hw
    
    if left_keypoints is None:
        left_keypoints = np.zeros([1,21,2])
        
    if right_keypoints is None:
        right_keypoints = np.zeros([1,21,2])
    # [T, 21, 2]
    left_mean_x, left_mean_y, left_diff_x, left_diff_y = compute_bbox(left_keypoints)
    left_mean_x = W*left_mean_x
    left_mean_y = H*left_mean_y
    
    left_diff_x = W*left_diff_x
    left_diff_y = H*left_diff_y
    
    left_diff_x = max(left_diff_x)
    left_diff_y = max(left_diff_y)
    left_box_hw = max(left_diff_x,left_diff_y)
    
    right_mean_x, right_mean_y, right_diff_x, right_diff_y = compute_bbox(right_keypoints)
    right_mean_x = W*right_mean_x
    right_mean_y = H*right_mean_y
    
    right_diff_x = W*right_diff_x
    right_diff_y = H*right_diff_y
    
    right_diff_x = max(right_diff_x)
    right_diff_y = max(right_diff_y)
    right_box_hw = max(right_diff_x,right_diff_y)
    
    box_hw = int(max(left_box_hw, right_box_hw) * 1.2 / 2) * 2
    box_hw = max(box_hw, 0)

    left_new_box = np.stack([left_mean_x - box_hw/2, left_mean_y - box_hw/2, left_mean_x + box_hw/2, left_mean_y + box_hw/2]).astype(np.int16)
    right_new_box = np.stack([right_mean_x - box_hw/2, right_mean_y - box_hw/2, right_mean_x + box_hw/2, right_mean_y + box_hw/2]).astype(np.int16)
    
    return left_new_box.transpose(1,0), right_new_box.transpose(1,0), box_hw

def load_support_rgb_dict(tmp, skeletons, confs, full_path, data_transform):
    support_rgb_dict = {}
    
    confs = np.array(confs)
    skeletons = np.array(skeletons) 

    # sample index of low scores
    left_confs_filter = confs[:,0,91:112].mean(-1)
    left_confs_filter_indices = np.where(left_confs_filter > 0.3)[0]

    if len(left_confs_filter_indices) == 0:
        left_sampled_indices = None
        left_skeletons = None
    else:
        
        left_confs = confs[left_confs_filter_indices]
        left_confs = left_confs[:,0,[95,99,103,107,111]].min(-1)
        
        left_weights = np.max(left_confs) - left_confs + 1e-5
        left_probabilities = left_weights / np.sum(left_weights)
        
        left_sample_size = int(np.ceil(0.1 * len(left_confs_filter_indices)))
        
        left_sampled_indices = np.random.choice(left_confs_filter_indices.tolist(), 
                                                size=left_sample_size, 
                                                replace=False, 
                                                p=left_probabilities)
        # left_sampled_indices: values: 0-255(0,max_len)
        # tmp: values: 0-(end-start)
        left_sampled_indices = np.sort(left_sampled_indices)
        
        left_skeletons = skeletons[left_sampled_indices,0,91:112]

    right_confs_filter = confs[:,0,112:].mean(-1)
    right_confs_filter_indices = np.where(right_confs_filter > 0.3)[0]
    if len(right_confs_filter_indices) == 0:
        right_sampled_indices = None
        right_skeletons = None
        
    else:
        right_confs = confs[right_confs_filter_indices]
        right_confs = right_confs[:,0,[95+21,99+21,103+21,107+21,111+21]].min(-1)

        right_weights = np.max(right_confs) - right_confs + 1e-5
        right_probabilities = right_weights / np.sum(right_weights)
        
        right_sample_size = int(np.ceil(0.1 * len(right_confs_filter_indices)))
        
        right_sampled_indices = np.random.choice(right_confs_filter_indices.tolist(), 
                                                 size=right_sample_size, 
                                                 replace=False, 
                                                 p=right_probabilities)
        right_sampled_indices = np.sort(right_sampled_indices)
        
        right_skeletons = skeletons[right_sampled_indices,0,112:133]
        
    image_size = 112
    all_indices = []
    if not left_sampled_indices is None:
        all_indices.append(left_sampled_indices)
    if not right_sampled_indices is None:
        all_indices.append(right_sampled_indices)
    if len(all_indices) == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        
        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    sampled_indices = np.concatenate(all_indices)
    sampled_indices = np.unique(sampled_indices)
    sampled_indices_real = tmp[sampled_indices]

    # load image sample
    imgs = load_video_support_rgb(full_path, sampled_indices_real)

    # get hand bbox
    left_new_box, right_new_box, box_hw = bbox_4hands(left_skeletons,
                                                        right_skeletons,
                                                        imgs[0].shape[:2])
    
    # crop left and right hand
    image_size = 112
    if box_hw == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        
        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    factor = image_size / box_hw
    
    if left_sampled_indices is None:
        left_hands = torch.zeros(1, 3, image_size, image_size)
        left_skeletons_norm = torch.zeros(1, 21, 2)
        
    else:
        left_hands = torch.zeros(len(left_sampled_indices), 3, image_size, image_size)
            
        left_skeletons_norm = left_skeletons * imgs[0].shape[:2][::-1] - left_new_box[:, None, [0,1]]
        left_skeletons_norm = left_skeletons_norm / box_hw
        left_skeletons_norm = left_skeletons_norm.clip(0,1)

    if right_sampled_indices is None:
        right_hands = torch.zeros(1, 3, image_size, image_size)
        right_skeletons_norm = torch.zeros(1, 21, 2)
        
    else:
        right_hands = torch.zeros(len(right_sampled_indices), 3, image_size, image_size)
        
        right_skeletons_norm = right_skeletons * imgs[0].shape[:2][::-1] - right_new_box[:, None, [0,1]]
        right_skeletons_norm = right_skeletons_norm / box_hw
        right_skeletons_norm = right_skeletons_norm.clip(0,1)
    left_idx = 0
    right_idx = 0

    for idx, img in enumerate(imgs):
        mapping_idx = sampled_indices[idx]
        if not left_sampled_indices is None and left_idx < len(left_sampled_indices) and mapping_idx == left_sampled_indices[left_idx]:
            box = left_new_box[left_idx]
            
            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3],box[0]:box[2],:]
            img_draw = np.pad(img_draw, ((0, max(0, box_hw-img_draw.shape[0])), (0, max(0, box_hw-img_draw.shape[1])), (0, 0)), mode='constant', constant_values=0)
            
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            left_hands[left_idx] = f_img
            left_idx += 1
            
        if not right_sampled_indices is None and right_idx < len(right_sampled_indices) and mapping_idx == right_sampled_indices[right_idx]:
            box = right_new_box[right_idx]
            
            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3],box[0]:box[2],:]
            img_draw = np.pad(img_draw, ((0, max(0, box_hw-img_draw.shape[0])), (0, max(0, box_hw-img_draw.shape[1])), (0, 0)), mode='constant', constant_values=0)
            
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            right_hands[right_idx] = f_img
            right_idx += 1
   
    if left_sampled_indices is None:
        left_sampled_indices = np.array([-1])
        
    if right_sampled_indices is None:
        right_sampled_indices = np.array([-1])

    # get index, images and keypoints priors
    support_rgb_dict['left_sampled_indices'] = torch.tensor(left_sampled_indices)
    support_rgb_dict['left_hands'] = left_hands
    support_rgb_dict['left_skeletons_norm'] = torch.tensor(left_skeletons_norm)
    
    support_rgb_dict['right_sampled_indices'] = torch.tensor(right_sampled_indices)
    support_rgb_dict['right_hands'] = right_hands
    support_rgb_dict['right_skeletons_norm'] = torch.tensor(right_skeletons_norm)

    return support_rgb_dict


# use split rgb video for save time
def load_video_support_rgb(path, tmp):
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    
    vr.seek(0)
    buffer = vr.get_batch(tmp).asnumpy()
    batch_image = buffer
    del vr

    return batch_image

# build base dataset
class Base_Dataset(Dataset.Dataset):
    def collate_fn(self, batch):
        # 初始化批次数据
        tgt_batch, src_length_batch, name_batch, pose_tmp, gloss_batch = [], [], [], [], []
        descriptions_batch = []
        has_description_batch = []
        
        # 解包批次数据，支持新旧格式兼容
        for item in batch:
            if len(item) == 7:
                # 新格式：包含描述和缺失指示符
                (name_sample, pose_sample, text, gloss, support_rgb_dict,
                 descriptions, has_description) = item[:7]
            else:
                # 原格式（向后兼容）
                name_sample, pose_sample, text, gloss, support_rgb_dict = item[:5]
                descriptions = None
                has_description = None
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)
            descriptions_batch.append(descriptions)
            has_description_batch.append(has_description)

        src_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch
                
        if self.rgb_support:
            support_rgb_dicts = {key:[] for key in batch[0][4].keys()}  # 第5个元素是support_rgb_dict
            for item in batch:
                support_rgb_dict = item[4]  # 获取第5个元素（support_rgb_dict）
                for key in support_rgb_dict.keys():
                    support_rgb_dicts[key].append(support_rgb_dict[key])
            
            for part in ['left', 'right']:
                index_key = f'{part}_sampled_indices'
                skeletons_key = f'{part}_skeletons_norm'
                rgb_key = f'{part}_hands'
                len_key = f'{part}_rgb_len'

                index_batch = torch.cat(support_rgb_dicts[index_key], 0)
                skeletons_batch = torch.cat(support_rgb_dicts[skeletons_key], 0)
                img_batch = torch.cat(support_rgb_dicts[rgb_key], 0)
                
                src_input[index_key] = index_batch
                src_input[skeletons_key] = skeletons_batch
                src_input[rgb_key] = img_batch
                src_input[len_key] = [len(index) for index in support_rgb_dicts[index_key]]

        # 打包描述文本
        if descriptions_batch and descriptions_batch[0] is not None:
            src_input['descriptions'] = descriptions_batch
            # 打包缺失指示符
            if has_description_batch and has_description_batch[0] is not None:
                max_desc_len = max((len(d) for d in descriptions_batch 
                                   if d is not None), default=0)
                has_description_padded = []
                for has_desc in has_description_batch:
                    if has_desc is not None:
                        if len(has_desc) < max_desc_len:
                            padded = torch.cat([
                                torch.tensor(has_desc, dtype=torch.float32),
                                torch.zeros(max_desc_len - len(has_desc), dtype=torch.float32)
                            ])
                        else:
                            padded = torch.tensor(has_desc, dtype=torch.float32)
                        has_description_padded.append(padded)
                if has_description_padded:
                    src_input['has_description'] = torch.stack(has_description_padded)
        else:
            src_input['descriptions'] = None
            src_input['has_description'] = None

        tgt_input = {}
        tgt_input['gt_sentence'] = tgt_batch
        tgt_input['gt_gloss'] = gloss_batch

        return src_input, tgt_input


class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase
        
        # 保存采样的帧索引（用于描述对齐）
        self._last_frame_indices = None

        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]
            
        elif "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        elif "How2Sign" in self.args.dataset:
            if phase == 'dev':
                raise NotImplementedError("How2Sign dev set is not supported")
            self.pose_dir = pose_dirs[args.dataset].format(phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        elif "OpenASL" in self.args.dataset:
            self.pose_dir = pose_dirs[args.dataset].format(phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        else:
            raise NotImplementedError

        self.list = list(self.raw_data.keys())

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])
        
        # 初始化描述加载器
        self.use_descriptions = getattr(args, 'use_descriptions', False)
        if self.use_descriptions:
            # 从 config 中获取描述文件基础路径
            if args.dataset in description_dirs:
                # 使用 config 中的路径 + phase 子文件夹
                desc_base_dir = Path(description_dirs[args.dataset])
                desc_dir = desc_base_dir / phase  # 例如 ./description/CSL_Daily/split_data/train
                
                # 检查描述文件夹是否存在
                if desc_dir.exists():
                    self.desc_loader = DescriptionLoader(str(desc_dir))
                else:
                    print(f"[警告] 描述文件夹不存在: {desc_dir}")
                    print(f"[警告] 禁用描述文本加载")
                    self.desc_loader = None
                    self.use_descriptions = False
            else:
                print(f"[警告] config.py 中未配置 {args.dataset} 的描述文件路径")
                print(f"[警告] 禁用描述文本加载")
                self.desc_loader = None
                self.use_descriptions = False
        else:
            self.desc_loader = None

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''
        
        name_sample = sample['name']
        pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])
        
        # 加载和对齐描述文本
        descriptions = None
        has_description = None
        if self.use_descriptions and self.desc_loader:
            descriptions, has_description = self._load_and_align_descriptions(
                name_sample, pose_sample
            )
        
        # 返回扩展的元组（包含描述文本）
        return name_sample, pose_sample, text, gloss, support_rgb_dict, descriptions, has_description
    
    def load_pose(self, path):
        """加载姿态数据，并保存采样的帧索引用于描述对齐"""
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
        
        # 保存采样的帧索引（原始帧号），用于描述对齐
        self._last_frame_indices = tmp.copy()
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            full_path = os.path.join(self.rgb_dir, path)
            support_rgb_dict = load_support_rgb_dict(tmp, skeletons, confs, full_path, self.data_transform)
            
        return kps_with_scores, support_rgb_dict
    
    def _load_and_align_descriptions(self, sample_id: str, pose_sample: dict):
        """
        加载并对齐描述文本
        
        参数：
            sample_id (str): 样本 ID，例如 'S000196_P0000_T00'
            pose_sample (dict): 姿态字典，包含时间维度信息
        
        返回：
            aligned_descriptions (list): 对齐后的描述文本列表 [str or None, ...]
            has_description (list): 缺失指示符 [int, ...]
                                   1 = 有真实描述，0 = 插值/缺失/最近邻
        """
        try:
            # 获取样本 ID（无扩展名）
            if isinstance(sample_id, str):
                sample_id = Path(sample_id).stem
            
            # 加载原始描述
            original_descriptions, metadata = self.desc_loader.load(sample_id)
            
            if not metadata['success'] or not original_descriptions:
                return None, None
            
            # 获取时间维度（采样帧数）
            # 从 pose_sample 字典中的任意 key 获取时间维度
            T_sampled = next(iter(pose_sample.values())).shape[0]
            
            # 获取采样后的帧索引（在 load_pose 中保存）
            if self._last_frame_indices is not None:
                # 使用实际的采样帧索引
                sampled_frame_indices = self._last_frame_indices.tolist()
            else:
                # 降级方案：假设是顺序采样
                sampled_frame_indices = list(range(T_sampled))
            
            # 智能插值对齐
            aligner = TemporalAligner(
                original_descriptions,
                sampled_frame_indices,
                use_nearest_neighbor=True,
                use_linear_interpolation=False
            )
            aligned_descriptions, has_desc = aligner.align()
            
            return aligned_descriptions, has_desc
        
        except Exception as e:
            print(f"[错误] 加载描述文本失败 - 样本 ID: {sample_id}, 错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def __str__(self):
        return f'#total {len(self)}'

class S2T_Dataset_news(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset_news, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(path)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
       
        if self.args.dataset == "CSL_News":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]
      
        else:
            raise NotImplementedError
        sum_sample = len(self.annotation)
        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

        if phase == 'train':
            self.start_idx = int(sum_sample * 0.0)
            self.end_idx = int(sum_sample * 0.99)
        else:
            self.start_idx = int(sum_sample * 0.99)
            self.end_idx = int(sum_sample)
        
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, index):
        num_retries = 10  

        # skip some invalid video sample
        for _ in range(num_retries):
            sample = self.annotation[self.start_idx:self.end_idx][index]

            text = sample['text']
            name_sample = sample['video']
           
            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'], sample['video'])
    
            except:
                import traceback

                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            break
           
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return name_sample, pose_sample, text, _, support_rgb_dict
    
    def load_pose(self, pose_name, rgb_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), 'rb'))
        full_path = os.path.join(self.rgb_dir, rgb_name)
        
        duration = len(pose['scores'])

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp)
            
        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)
        
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
                
        kps_with_scores = load_part_kp(skeletons, confs)
        
        support_rgb_dict = {}
        if self.rgb_support:
            support_rgb_dict = load_support_rgb_dict(tmp, skeletons, confs, full_path, self.data_transform)

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'

class S2T_Dataset_online(Base_Dataset):
    def __init__(self, args):
        super(S2T_Dataset_online, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.max_length = args.max_length

        # place holder
        self.rgb_data = None
        self.pose_data = None

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        text = ''
        gloss = ''
        name_sample = 'online_data'

        pose_sample, support_rgb_dict = self.load_pose()

        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self):
        pose = self.pose_data

        duration = len(pose['scores'])
        start = 0

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))

        tmp = np.array(tmp) + start

        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp

        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            full_path = self.rgb_data
            support_rgb_dict = load_support_rgb_dict(tmp, skeletons, confs, full_path, self.data_transform)

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'
