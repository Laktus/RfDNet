import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.registers import MODULES

from pcdet.models.backbones_3d import BiResNet
from pcdet.models.dense_heads import CAGroup3DHead
from pcdet.models.roi_heads import CAGroup3DRoIHead

class CAGroup3D(nn.Module):
    def __init__(self, model_cfg):
        super(CAGroup3D, self).__init__()
        self.cfg = model_cfg
        self.backbone_3d = self.build_backbone_3d()
        self.add_module("backbone_3d", self.backbone_3d)
        self.dense_head = self.build_dense_head()
        self.add_module("dense_head", self.dense_head)
        self.roi_head = self.build_roi_head()
        self.add_module("roi_head", self.roi_head)

        # set hparams
        self.voxel_size = self.cfg.config['model']['detection']['voxel_size']
        self.semantic_min_threshold = self.cfg.config['model']['detection']['semantic_min_thr']
        self.semantic_iter_value = self.cfg.config['model']['detection']['semantic_iter_value']
        self.semantic_value = self.cfg.config['model']['detection']['semantic_thr']
    
    def voxelization(self, points):
        """voxelize input points."""
        # points Nx7 (bs_id, x, y, z, r, g, b)
        coordinates = points[:, :4].clone()
        coordinates[:, 1:] /= self.voxel_size
        features = points[:, 4:].clone()
        sp_tensor = ME.SparseTensor(coordinates=coordinates, features=features)
        return sp_tensor
    
    def build_backbone_3d(self):
        return BiResNet(
            cfg=self.cfg.config['model']['detection']['backbone'],
        )
    
    def build_dense_head(self):
        return CAGroup3DHead(
            cfg=self.cfg.config['model']['detection']['dense'],
        )

    def build_roi_head(self):
        return CAGroup3DRoIHead(
            cfg=self.cfg.config['model']['detection']['roi'],
        )
    
    def forward(self, batch_dict):
        # adjust semantic value
        cur_epoch = batch_dict.get('cur_epoch', None)
        assert cur_epoch is not None
        self.dense_head.semantic_threshold = max(self.semantic_value - int(cur_epoch) * self.semantic_iter_value, self.semantic_min_threshold)
        # normalize point features
        # batch_dict['points'][:, -3:] = batch_dict['points'][:, -3:] / 255.
        sp_tensor = self.voxelization(batch_dict['points'])
        batch_dict['sp_tensor'] = sp_tensor
        
        for cur_module in self.children():
            results = cur_module(batch_dict)
            batch_dict.update(results)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {
                'loss': loss
            }
            disp_dict['cur_semantic_value'] = self.dense_head.semantic_threshold
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def post_processing(self, batch_dict):
        post_process_cfg = self.cfg.config['model']['detection']['post_processing']
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            recall_dict = self.generate_recall_record(
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg['recall_thresh_list'] # [0.25, 0.5]
            )
            record_dict = {
                'pred_boxes':  batch_dict['batch_box_preds'][index],
                'pred_scores': batch_dict['batch_score_preds'][index],
                'pred_labels': batch_dict['batch_cls_preds'][index]
            }
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        return recall_dict
    
    @staticmethod
    def convert2list(points):
        batch_size = points[:, 0].max().int() + 1
        points_list = []
        for i in range(batch_size):
            p = points[points[:, 0] == i, 1:]
            points_list.append(p)
        return points_list

    def get_training_loss(self, batch_dict):
        batch_size = batch_dict['batch_size']
        device = batch_dict['points'].device
        if 'semantic_mask' in batch_dict.keys():
            pts_semantic_mask = [torch.from_numpy(x).to(device) for x in batch_dict['semantic_mask']]
        else:
            pts_semantic_mask = None
        if 'instance_mask' in batch_dict.keys():
            pts_instance_mask = [torch.from_numpy(x).to(device) for x in batch_dict['instance_mask']]
        else:
            pts_instance_mask = None
        
        if batch_dict.get('gt_bboxes_3d', None) is not None:
            gt_bboxes_3d = batch_dict['gt_bboxes_3d']
            gt_labels_3d = batch_dict['gt_labels_3d']
        else:
            gt_bboxes_3d = []
            gt_labels_3d = []
            for b in range(len(batch_dict['gt_boxes'])):
                gt_bboxes_b = []
                gt_labels_b = []
                for _item in batch_dict['gt_boxes'][b]:
                    if not (_item == 0.).all(): 
                        gt_bboxes_b.append(_item[:7])  
                        gt_labels_b.append(_item[7:8]) 
                if len(gt_bboxes_b) == 0:
                    gt_bboxes_b = torch.zeros((0, 7), dtype=torch.float32).to(device)
                    gt_labels_b = torch.zeros((0,), dtype=torch.long).to(device)
                else:
                    gt_bboxes_b = torch.stack(gt_bboxes_b)
                    gt_labels_b = torch.cat(gt_labels_b).long()
                gt_bboxes_3d.append(gt_bboxes_b)
                gt_labels_3d.append(gt_labels_b)
            batch_dict['gt_bboxes_3d'] = gt_bboxes_3d
            batch_dict['gt_labels_3d'] = gt_labels_3d
        img_metas = [None for _ in range(batch_size)]

        # one-stage loss
        x, semantic_scores, voxel_offset = batch_dict['one_stage_results']
        centernesses, bbox_preds, cls_scores, voxel_points = x
        losses_inputs = (centernesses, bbox_preds, cls_scores, voxel_points, semantic_scores, voxel_offset, \
                                            gt_bboxes_3d, gt_labels_3d, self.convert2list(batch_dict['points']), img_metas,
                                            pts_semantic_mask, pts_instance_mask) 
        disp_dict = {}
        loss_one_stage, tb_dict = self.dense_head.loss(*losses_inputs)

        # two-stage loss
        loss_two_stage, tb_dict_two_stage = self.roi_head.loss(batch_dict) 
        tb_dict.update(tb_dict_two_stage)
        for k in tb_dict.keys():
            disp_dict[k] = tb_dict[k]
        loss_all = loss_one_stage + loss_two_stage

        tb_dict = {
            'loss_all': loss_all.item(),
            **tb_dict
        }
        
        return loss_all, tb_dict, disp_dict