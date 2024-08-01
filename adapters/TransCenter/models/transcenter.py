import os
import torch
import torch.nn as nn

from tracking.mot17_private import get_args_parser as get_args_parser_mot17
from tracking.mot20_private import get_args_parser as get_args_parser_mot20
from tracking.deformable_detr import build as build_model
from post_processing.decode import generic_decode

# We add TransCenter_official to the python path. We have to do it in this "relative" way so later obj_detect correctly detects NestedTensor
import sys
__CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(__CURRENT_FOLDER, '..', '..', 'TransCenter_official'))
print(os.path.join(__CURRENT_FOLDER, '..', '..', 'TransCenter_official'))
from util.misc import NestedTensor
# Now we remove the TransCenter_official from the python path
sys.path.pop(0)


class TransCenterDETR(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()

        # From TransCenter_official. Differences between mot17_private and mot20_private:
        # - main_args.K (max number of output objects): mot17 = 300 | mot20 = 500
        # - main_args.clip (prevent dets from going out of bounds): mot17 = False | mot20 = True
        # - main_args.fuse_scores (wether to multiply IoU dists by the confidence score): mot17 = True | mot20 = False
        # - main_args.track_thresh (threshold for considering 1st-round detections): mot17 = 0.3 | mot20 = 0.4
        # - main_args.noprehm (*I believe it's an unused argument*): mot17 = NULL | mot20 = True

        if dataset_name.upper() in set(['MOT17', 'MOT-2017', 'MOT16', 'MOT-2016']):
            main_args = get_args_parser_mot17().parse_args(args=[])
            main_args.K = 300
            main_args.clip = False
            main_args.fuse_scores = True
            main_args.track_thresh = 0.3
            main_args.noprehm = None
        
        elif dataset_name.upper() in set(['MOT20', 'MOT-2020']):
            main_args = get_args_parser_mot20().parse_args(args=[])
            main_args.K = 500
            main_args.clip = True
            main_args.fuse_scores = False
            main_args.track_thresh = 0.4
            main_args.noprehm = True
        
        else:
            raise ValueError(f'Invalid dataset name: {dataset_name}')

        # load model

        main_args.pre_hm = True
        main_args.tracking = True
        main_args.iou_recover = True

        # device = torch.device(main_args.device)

        main_args.output_h = main_args.input_h // main_args.down_ratio
        main_args.output_w = main_args.input_w // main_args.down_ratio
        main_args.input_res = max(main_args.input_h, main_args.input_w)
        main_args.output_res = max(main_args.output_h, main_args.output_w)

        model, criterion, postprocessors = build_model(main_args)
        n_parameters = sum(p.numel() for p in model.parameters())
        print('number of params:', n_parameters)

        # model = model.cuda()
        # model.eval()
        # model.half()

        self.obj_detect = model
        self.main_args = main_args

    @torch.no_grad()
    def forward(self, x, current_pos=None):  # Extracted from "detect_tracking_duel_vit" (TransCenter)
        
        padding_mask = torch.ones_like(x).to(x.device)
        padding_mask = 1 - padding_mask
        padding_mask = padding_mask[0, 0, :, :].unsqueeze(0).to(bool)

        samples = NestedTensor(x, padding_mask)

        batch = {'frame_name': None, 'video_name': None, 'img': None,
                'samples': samples, 'orig_size': None,
                'dets': torch.zeros(0, 4),
                'trans': [1.0, 0, 0]  # [ratio, padw, padh]
                }

        self.sample = batch['samples']

        if self.pre_sample is None:
            self.pre_sample = self.sample
        

        if current_pos is None:
            mypos = torch.zeros(size=(0, 4), device=self.sample.tensors.device).float()
        else:
            mypos = current_pos

        [ratio, padw, padh] = batch['trans']

        no_pre_cts = False

        if mypos.shape[0] > 0:
            # make pre_cts #
            # bboxes to centers
            hm_h, hm_w = self.sample.tensors.shape[2], self.sample.tensors.shape[3]
            bboxes = mypos.clone()
            # bboxes

            bboxes[:, 0] += bboxes[:, 2]
            bboxes[:, 1] += bboxes[:, 3]
            pre_cts = bboxes[:, 0:2] / 2.0

            # to input image plane
            pre_cts *= ratio
            pre_cts[:, 0] += padw
            pre_cts[:, 1] += padh
            pre_cts[:, 0] = torch.clamp(pre_cts[:, 0], 0, hm_w - 1)
            pre_cts[:, 1] = torch.clamp(pre_cts[:, 1], 0, hm_h - 1)

            # to output image plane
            pre_cts /= self.main_args.down_ratio
        else:
            pre_cts = torch.zeros(size=(2, 2), device=mypos.device, dtype=mypos.dtype)

            no_pre_cts = True
            print("No Pre Cts!")


        outputs = self.obj_detect(samples=self.sample, pre_samples=self.pre_sample,
                                  pre_cts=pre_cts.clone().unsqueeze(0))
        # post processing #
        output = {k: v[-1] for k, v in outputs.items() if k != 'boxes'}

        # 'hm' is not _sigmoid!
        output['hm'] = torch.clamp(output['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)

        decoded = generic_decode(output, K=self.main_args.K, opt=self.main_args)

        out_scores = decoded['scores'][0]
        labels_out = decoded['clses'][0].int() + 1

        # reid features #

        if no_pre_cts:
            pre2cur_cts = torch.zeros_like(mypos)[..., :2]
        else:
            pre2cur_cts = self.main_args.down_ratio * (decoded['tracking'][0] + pre_cts)
            pre2cur_cts[:, 0] -= padw
            pre2cur_cts[:, 1] -= padh
            pre2cur_cts /= ratio

        # extract reid features #
        boxes = decoded['bboxes'][0].clone()
        reid_cts = torch.stack([0.5*(boxes[:, 0]+boxes[:, 2]), 0.5*(boxes[:, 1]+boxes[:, 3])], dim=1)
        reid_cts[:, 0] /= outputs['reid'][0].shape[3]
        reid_cts[:, 1] /= outputs['reid'][0].shape[2]
        reid_cts = torch.clamp(reid_cts, min=0.0, max=1.0)
        reid_cts = (2.0 * reid_cts - 1.0)

        out_boxes = decoded['bboxes'][0] * self.main_args.down_ratio
        out_boxes[:, 0::2] -= padw
        out_boxes[:, 1::2] -= padh
        out_boxes /= ratio

        # filtered by scores #
        filtered_idx = labels_out == 1
        out_scores = out_scores[filtered_idx]
        out_boxes = out_boxes[filtered_idx]

        reid_cts = reid_cts[filtered_idx]
        if self.main_args.clip:  # for mot20 clip box
            _, _, orig_h, orig_w  = self.sample.tensors.shape
            out_boxes[:, 0::2] = torch.clamp(out_boxes[:, 0::2], 0, orig_w-1)
            out_boxes[:, 1::2] = torch.clamp(out_boxes[:, 1::2], 0, orig_h-1)

        # post processing #

        # We convert the boxes to the format that the tracker expects
        # out_boxes are in the format [x1, y1, x2, y2] and we need to convert them to [x, y, w, h]
        out_boxes[:, 2] = out_boxes[:, 2] - out_boxes[:, 0]
        out_boxes[:, 3] = out_boxes[:, 3] - out_boxes[:, 1]
        out_boxes[:, 0] = out_boxes[:, 0] + out_boxes[:, 2] / 2
        out_boxes[:, 1] = out_boxes[:, 1] + out_boxes[:, 3] / 2

        # Normal YOLOX outputs something like `output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)`
        # As later it will aggregate the class_conf and obj_conf (multiply them), we will set class_conf to 1
        out_scores = out_scores.unsqueeze(1)
        
        output = torch.cat([out_boxes, out_scores, torch.ones_like(out_scores)], dim=1)  # We repeat scores twice because we don't have class/obj scores
        output = output.unsqueeze(0)  # We add a batch dimension

        self.pre_sample = self.sample

        return output
    
    def reset(self):
        self.pre_sample = None
        self.obj_detect.pre_memory = None
        self.sample = None
        self.obj_detect.masks_flatten = None

