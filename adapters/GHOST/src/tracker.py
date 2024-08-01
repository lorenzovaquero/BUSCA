from collections import defaultdict
import torch
import os
import numpy as np
import logging
from lapsolver import solve_dense
import json
from src.tracking_utils import get_proxy, Track, multi_predict, get_iou_kalman
from src.base_tracker import BaseTracker
from tqdm import tqdm

from busca.network import BUSCA
from busca.tracking import center_distance

logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker(BaseTracker):
    def __init__(
            self,
            tracker_cfg,
            encoder,
            net_type='resnet50',
            output='plain',
            data='tracktor_preprocessed_files.txt',
            device='cpu'):
        super(
            Tracker,
            self).__init__(
            tracker_cfg,
            encoder,
            net_type,
            output,
            data,
            device)
        self.short_experiment = defaultdict(list)
        self.inact_patience = self.tracker_cfg['inact_patience']

        # vvv BUSCA vvv
        if 'use_busca' in self.tracker_cfg and self.tracker_cfg['use_busca'] is True:
            self.use_busca = True
        else:
            self.use_busca = False
        
        self.save_memory = False
            
        if self.use_busca:
            busca_args = self.tracker_cfg['busca_args'].transformer

            busca_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.transformer_tracker = BUSCA(busca_args).to(busca_args.device)
            self.transformer_tracker.load_pretrained(self.tracker_cfg['busca_ckpt'], ignore_reid_fc=True)
            self.transformer_tracker.eval()

            if self.tracker_cfg['busca_args'].transformer_update_mems_only_first_round:
                # We only want to update the mems of the first round
                # But as GHOST needs to update some of its stuff (and there is overlapping),
                # we apply a filter so BUSCA only sees the high-confidence detections
                busca_mem_thresh = self.tracker_cfg[self.tracker_cfg['busca_args'].minimum_conf_base] + self.tracker_cfg['busca_args'].minimum_conf_modifier
                Track.set_busca_conf_threshold(conf_thres=busca_mem_thresh)
            
            if hasattr(self.tracker_cfg['busca_args'], 'avoid_memory_leak') and self.tracker_cfg['busca_args'].avoid_memory_leak:
                print('Enabling GHOST memory saving mode!', flush=True)
                self.save_memory = True
            
        else:
            self.transformer_tracker = None
        # ^^^ BUSCA ^^^

    def track(self, seq, first=False, log=True):
        '''
        first - feed all bounding boxes through net first for bn stats update
        seq -   sequence instance for iteratration and with meta information
                like name or lentgth
        '''
        self.log = log
        if self.log:
            logger.info(
                "Tracking sequence {} of lenght {}".format(
                    seq.name, seq.num_frames))

        # initalize variables for sequence
        self.setup_seq(seq, first)

        # batch norm experiemnts I - before iterating over sequence
        self.normalization_before(seq, first=first)
        self.prev_frame = 0
        i = 0
        # iterate over frames
        for frame_data in tqdm(seq, total=len(seq)):
            frame, path, boxes, gt_ids, vis, \
                random_patches, whole_im, conf, label = frame_data
            # log if in training mode
            if i == 0:
                logger.info(f'Network in training mode: {self.encoder.training}')
            self.i = i

            # batch norm experiments II on the fly
            self.normalization_experiments(random_patches, frame, i, seq)

            # get frame id
            if 'bdd' in path:
                self.frame_id = int(path.split(
                    '/')[-1].split('-')[-1].split('.')[0])
            else:
                self.frame_id = int(path.split(os.sep)[-1][:-4])

            # detections of current frame
            detections = list()

            # forward pass
            feats = self.get_features(frame)

            # Here frame is the already cropped bboxes
            # frame.shape = torch.Size([N, 3, 384, 128])
            det_images = frame.cpu().detach().permute(0, 2, 3, 1).numpy()
            det_images = det_images[..., ::-1]  # Also we convert from RGB to BGR
            det_images = self._normalize_embeddings_batch(det_images, denormalize=True)  # We denormalize them into normal uint8 images

            # if just feeding for bn stats update
            if first:
                continue

            # iterate over bbs in current frame
            for f, b, gt_id, v, c, l, bbox_img in zip(feats, boxes, gt_ids, vis, conf, label, det_images):
                if (b[3] - b[1]) / (b[2] - b[0]
                                    ) < self.tracker_cfg['h_w_thresh']:
                    if c < self.tracker_cfg['det_conf']:
                        continue
                    
                    # vvv BUSCA vvv
                    if 'use_busca' in self.tracker_cfg and self.tracker_cfg['use_busca']:
                        # We get the images of the detections
                        det_image = bbox_img
                    else:
                        det_image = None
                    # ^^^ BUSCA ^^^

                    detection = {
                        'bbox': b,
                        'feats': f,
                        'im_index': self.frame_id,
                        'gt_id': gt_id,
                        'vis': v,
                        'conf': c,
                        'frame': self.frame_id,
                        'label': l,
                        'image': det_image,  # BUSCA
                        }
                    detections.append(detection)

                    # store features
                    if self.store_feats:
                        self.add_feats_to_storage(detections)

            # apply motion compensation to stored track positions
            if self.motion_model_cfg['motion_compensation']:
                self.motion_compensation(whole_im, i)

            # association over frames
            whole_im_cv2 = whole_im.cpu().detach().permute(1, 2, 0).numpy() * 255.0
            whole_im_cv2 = whole_im_cv2[..., ::-1].copy().astype(np.uint8)  # Also we convert from RGB to BGR
            tr_ids = self._track(detections, i, frame=whole_im_cv2)

            # visualize bounding boxes
            if self.store_visualization:
                self.visualize(detections, tr_ids, path, seq.name, i+1)

            # get previous frame
            self.prev_frame = self.frame_id

            # increase count
            i += 1

        # just fed for bn stats update
        if first:
            logger.info('Done with pre-tracking feed...')
            return

        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)

        # write results
        self.write_results(self.output_dir, seq.name)

        # reset thresholds if every / tbd for next sequence
        self.reset_threshs()

        # store features
        if self.store_feats:
            os.makedirs('features', exist_ok=True)
            path = os.path.join('features', self.experiment + 'features.json')
            logger.info(f'Storing features to {path}...')
            if os.path.isfile(path):
                with open(path, 'r') as jf:
                    features_ = json.load(jf)
                self.features_.update(features_)

            with open(path, 'w') as jf:
                json.dump(self.features_, jf)
            self.features_ = defaultdict(dict)

    def _track(self, detections, i, frame=None):
        # get inactive tracks with inactive < patience
        self.curr_it = {k: track for k, track in self.inactive_tracks.items()
                        if track.inactive_count <= self.inact_patience}

        # just add all bbs to self.tracks / intitialize in the first frame
        if len(self.tracks) == 0 and len(self.curr_it) == 0:
            tr_ids = list()
            for detection in detections:
                self.tracks[self.id] = Track(
                    track_id=self.id,
                    **detection,
                    kalman=self.kalman,
                    kalman_filter=self.kalman_filter)  # BUSCA
                tr_ids.append(self.id)
                self.id += 1

        # association over frames for frame > 0
        elif i > 0:
            # get hungarian matching
            if len(detections) > 0:

                # get proxy features of tracks first and compute distance then
                if not self.tracker_cfg['avg_inact']['proxy'] == 'each_sample':
                    dist, row, col, ids = self.get_hungarian_with_proxy(
                        detections, sep=self.tracker_cfg['assign_separately'])

                # get aveage of distances to all detections in track --> proxy dist
                else:
                    dist, row, col, ids = self.get_hungarian_each_sample(
                        detections, sep=self.tracker_cfg['assign_separately'])
            else:
                dist, row, col, ids = 0, 0, 0, 0

            if dist is not None:
                # get bb assignment
                tr_ids = self.assign(
                    detections=detections,
                    dist=dist,
                    row=row,
                    col=col,
                    ids=ids,
                    sep=self.tracker_cfg['assign_separately'],
                    frame_img=frame)  # BUSCA
        
        # vvv BUSCA vvv
        if self.save_memory:
            if len(self.inactive_tracks) > 0:
                for t_id in list(self.inactive_tracks):
                    inact_track = self.inactive_tracks[t_id]
                    if inact_track.inactive_count > self.inact_patience + 5:
                        inact_track._images_mem = [None] * len(inact_track._images_mem)
                        inact_track.past_feats = [None] * len(inact_track.past_feats)
            
            for detection in detections:
                detection['image'] = None
        # ^^^ BUSCA ^^^

        return tr_ids

    def last_frame(self, ids, tracks, x, nan_over_classes, labels_dets):
        """
        Get distance of detections to last frame of tracks
        """
        y = torch.stack([t.feats for t in tracks.values()])
        ids.extend([i for i in tracks.keys()])
        dist = self.dist(x, y).T

        # set distance between matches of different classes to nan
        if nan_over_classes:
            labels = np.array([t.label[-1] for t in tracks.values()])
            label_mask = np.atleast_2d(labels).T == np.atleast_2d(labels_dets)
            dist[~label_mask] = np.nan
        return dist

    def proxy_dist(self, tr, x, nan_over_classes, labels_dets):
        """
        Compute proxy distances using all detections in given track
        """
        # get distance between detections and all dets of track
        y = torch.stack(tr.past_feats)
        dist = self.dist(x, y)

        # reduce
        if self.tracker_cfg['avg_inact']['num'] == 1:
            dist = np.min(dist, axis=1)
        elif self.tracker_cfg['avg_inact']['num'] == 2:
            dist = np.mean(dist, axis=1)
        elif self.tracker_cfg['avg_inact']['num'] == 3:
            dist = np.max(dist, axis=1)
        elif self.tracker_cfg['avg_inact']['num'] == 4:
            dist = (np.max(dist, axis=1) + np.min(dist, axis=1))/2
        elif self.tracker_cfg['avg_inact']['num'] == 5:
            dist = np.median(dist, axis=1)

        # nan over classes
        if nan_over_classes:
            label_mask = np.atleast_2d(np.array(tr.label[-1])) == \
                np.atleast_2d(labels_dets).T
            dist[~label_mask.squeeze()] = np.nan

        return dist

    def get_hungarian_each_sample(
            self, detections, nan_over_classes=True, sep=False):
        """
        Get distances using proxy distance, i.e., the average distance
        to all detections in given track
        """

        # get new detections
        x = torch.stack([t['feats'] for t in detections])
        dist_all, ids = list(), list()

        # if setting dist values between classes to nan before hungarian
        if nan_over_classes:
            labels_dets = np.array([t['label'] for t in detections])

        # distance to active tracks
        if len(self.tracks) > 0:

            # just compute distance to last detection of active track
            if not self.tracker_cfg['avg_act']['do'] and len(detections) > 0:
                dist = self.last_frame(
                    ids, self.tracks, x, nan_over_classes, labels_dets)
                dist_all.extend([d for d in dist])

            # if use each sample for active frames
            else:
                for id, tr in self.tracks.items():
                    # get distance between detections and all dets of track
                    ids.append(id)
                    dist = self.proxy_dist(
                        tr, x, nan_over_classes, labels_dets)
                    dist_all.append(dist)

        # get number of active tracks
        num_active = len(ids)

        # get distances to inactive tracklets (inacht thresh = 100000)
        curr_it = self.curr_it
        if len(curr_it) > 0:
            if not self.tracker_cfg['avg_inact']['do']:
                dist = self.last_frame(
                    ids, curr_it, x, nan_over_classes, labels_dets)
                dist_all.extend([d for d in dist])
            else:
                for id, tr in curr_it.items():
                    ids.append(id)
                    dist = self.proxy_dist(
                        tr, x, nan_over_classes, labels_dets)
                    dist_all.append(dist)

        # stack all distances
        dist = np.vstack(dist_all).T

        dist, row, col = self.solve_hungarian(
            dist, num_active, detections, curr_it, sep)

        return dist, row, col, ids

    def solve_hungarian(self, dist, num_active, detections, curr_it, sep):
        """
        Solve hungarian assignment
        """
        # update thresholds
        self.update_thresholds(dist, num_active, len(curr_it))

        # get motion distance
        if self.motion_model_cfg['apply_motion_model']:

            # simple linear motion model
            if not self.kalman:
                self.motion()
                iou = self.get_motion_dist(detections, curr_it)

            # kalman fiter
            else:
                self.motion(only_vel=True)
                stracks = multi_predict(
                    self.tracks,
                    curr_it,
                    self.shared_kalman)
                iou = get_iou_kalman(stracks, detections)

            # combine motion distances
            dist = self.combine_motion_appearance(iou, dist)

        # set values larger than thershold to nan --> impossible assignment
        if self.nan_first:
            dist[:, :num_active] = np.where(
                dist[:, :num_active] <= self.act_reid_thresh, dist[:, :num_active], np.nan)
            dist[:, num_active:] = np.where(
                dist[:, num_active:] <= self.inact_reid_thresh, dist[:, num_active:], np.nan)

        # solve at once
        if not sep:
            row, col = solve_dense(dist)

        # solve active first and inactive later
        else:
            dist_act = dist[:, :num_active]
            row, col = solve_dense(dist_act)
            if num_active > 0:
                dist_inact = dist[:, num_active:]
            else:
                dist_inact = None
            dist = [dist_act, dist_inact]

        return dist, row, col

    def get_hungarian_with_proxy(self, detections, sep=False):
        """
        Use proxy feature vectors for distance computation
        """
        # instantiate
        ids = list()
        y_inactive, y = None, None

        x = torch.stack([t['feats'] for t in detections])

        # Get active track proxies
        if len(self.tracks) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y = get_proxy(
                    curr_it=self.tracks,
                    mode='act',
                    tracker_cfg=self.tracker_cfg,
                    mv_avg=self.mv_avg)
            else:
                y = torch.stack(
                    [track.feats for track in self.tracks.values()])
            ids += list(self.tracks.keys())
        # get num active tracks
        num_active = len(ids)

        # get inactive tracks with inactive < patience
        curr_it = {k: track for k, track in self.inactive_tracks.items()
                   if track.inactive_count <= self.inact_patience}
        # get inactive track proxies
        if len(curr_it) > 0:
            if self.tracker_cfg['avg_inact']['do']:
                y_inactive = get_proxy(
                    curr_it=curr_it,
                    mode='inact',
                    tracker_cfg=self.tracker_cfg,
                    mv_avg=self.mv_avg)
            else:
                y_inactive = torch.stack([track.feats
                                         for track in curr_it.values()])

            if len(self.tracks) > 0:
                y = torch.cat([y, y_inactive])
            else:
                y = y_inactive

            ids += [k for k in curr_it.keys()]

        # if no active or inactive tracks --> return and instantiate all dets
        # new
        elif len(curr_it) == 0 and len(self.tracks) == 0:
            for detection in detections:
                self.tracks[self.id] = Track(
                    track_id=self.id,
                    **detection,
                    kalman=self.kalman,
                    kalman_filter=self.kalman_filter)
                self.id += 1
            return None, None, None, None

        # get distance between proxy features and detection features
        dist = self.dist(x, y)

        # solve hungarian
        dist, row, col = self.solve_hungarian(
            dist, num_active, detections, curr_it, sep)

        return dist, row, col, ids

    def assign(self, detections, dist, row, col, ids, sep=False, frame_img=None):
        """
        Filter hungarian assignments using matching thresholds
        either assigning active and inactive together or separately
        """
        # assign tracks from hungarian
        active_tracks = list()
        tr_ids = [None for _ in range(len(detections))]
        if len(detections) > 0:
            if not sep:
                assigned = self.assign_act_inact_same_time(
                    row, col, dist, detections, active_tracks, ids, tr_ids)
            else:
                assigned = self.assign_separatly(
                    row, col, dist, detections, active_tracks, ids, tr_ids)

        else:
            assigned = list()
        
        # vvv BUSCA vvv
        ''' vvv Step 3b (BUSCA): Third association, between Kalman and Tracks vvv'''
        # The goal of this association is to refine `u_track` (unmatched tracks after second round) and check if we really have to deactivate them
        third_round_indices = []
        third_round_stracks = []
        keys = list(self.tracks.keys())
        for k in keys:
            if k not in active_tracks:
                unconfirmed = len(
                    self.tracks[k]) >= 2 if self.tracker_cfg['remove_unconfirmed'] else True
                if unconfirmed:
                    # We are going to filter those tracks with negative area or Kalman predictions with negative area.
                    tlwh_area = self.tracks[k].tlwh[2] * self.tracks[k].tlwh[3]
                    pos_area = (self.tracks[k].pos[2] - self.tracks[k].pos[0]) * (self.tracks[k].pos[3] - self.tracks[k].pos[1])
                    if tlwh_area <= 0.0 or pos_area <= 0.0:
                        print('WARNING: Found a track ({}) with negative area! Ignoring it...'.format(k), flush=True)
                        continue

                    third_round_indices.append(k)
                    third_round_stracks.append(self.tracks[k])

        if 'use_busca' in self.tracker_cfg and self.tracker_cfg['use_busca'] and hasattr(self.tracker_cfg['busca_args'], 'busca_thresh') and self.tracker_cfg['busca_args'].busca_thresh > 0 and len(third_round_indices) > 0:
            main_args = self.tracker_cfg['busca_args']  # For convenience
            
            extra_kalman_candidates = self.get_extra_kalman_candidates(strack=third_round_stracks, frame_img=frame_img,
                                                                        det_conf=Track._conf_thres)  # We set the confidence to the minimum that will make them appear on the Track mems

            all_considered_dets = self.get_all_considered_dets(detections)
            
            third_round_matches, third_round_u_track = self.third_round_association(strack_pool=third_round_stracks, considered_dets=all_considered_dets,
                                                                                    extra_kalman_candidates=extra_kalman_candidates, asoc_thresh=main_args.busca_thresh)

            for itracked, prob, idet in third_round_matches:
                track = third_round_stracks[itracked]

                if idet is None:  # It means it is kalman prediction
                    # We just associate it, like normal BUSCA
                    det = extra_kalman_candidates[itracked]  # itracked and "idet" are the same, as `third_round_stracks` and `extra_kalman_candidates` refer to the same tracks
                else:
                    raise NotImplementedError('Third round with BUSCA and DETS is not implemented yet')

                if main_args.transformer_update_mems_only_first_round:
                    # We previously set `Track.set_busca_conf_threshold()`
                    # We update new_img with whatever (as it will not be used by BUSCA due to the low confidence)
                    # And we update GHOST new_feats with either the last one from the track or the new one from the detection
                    if main_args.update_feats_third_round:
                        new_feats = det.feats
                    else:
                        new_feats = track.feats

                    new_img = track.images_mem[-1]  # Although it should never be used
                    new_conf = 0.10000001  # So it does not appear in the mems
                else:
                    new_feats = det.feats
                    new_img = det.image
                    new_conf = det.conf  # It will be the same as `self.tracker_cfg['det_conf']

                track.add_detection(bbox=det.pos, feats=new_feats, im_index=det.im_index, gt_id=det.gt_id, vis=det.gt_vis,
                                    conf=new_conf, frame=det.past_frames[-1], label=det.label[-1], image=new_img,
                                    save_memory=self.save_memory)
                
                active_tracks.append(third_round_indices[itracked])
            
            assigned = set(assigned)

        ''' ^^^ Step 3b (BUSCA): Third association, between Kalman and Transformer ^^^ '''
        # ^^^ BUSCA ^^^

        # move tracks not used to inactive tracks
        keys = list(self.tracks.keys())
        for k in keys:
            if k not in active_tracks:
                unconfirmed = len(
                    self.tracks[k]) >= 2 if self.tracker_cfg['remove_unconfirmed'] else True
                if unconfirmed:
                    self.inactive_tracks[k] = self.tracks[k]
                    self.inactive_tracks[k].inactive_count = 0
                del self.tracks[k]

        # increase inactive count by one
        for k in self.inactive_tracks.keys():
            self.inactive_tracks[k].inactive_count += self.frame_id - \
                self.prev_frame

        # start new track with unassigned detections if conf > thresh
        for i in range(len(detections)):
            if i not in assigned and detections[i]['conf'] > self.tracker_cfg['new_track_conf']:
                self.tracks[self.id] = Track(
                    track_id=self.id,
                    **detections[i],
                    kalman=self.kalman,
                    kalman_filter=self.kalman_filter)
                tr_ids[i] = self.id
                self.id += 1
        return tr_ids

    def assign_act_inact_same_time(
            self,
            row,
            col,
            dist,
            detections,
            active_tracks,
            ids,
            tr_ids):
        """
        Assign active and inactive at the same time
        """
        # assigned contains all new detections that have been assigned
        assigned = list()
        for r, c in zip(row, col):

            # assign tracks to active tracks if reid distance < thresh
            if ids[c] in self.tracks.keys() and \
                    dist[r, c] < self.act_reid_thresh:

                self.tracks[ids[c]].add_detection(**detections[r], save_memory=self.save_memory)
                active_tracks.append(ids[c])
                assigned.append(r)
                tr_ids[r] = ids[c]

            # assign tracks to inactive tracks if reid distance < thresh
            elif ids[c] in self.inactive_tracks.keys() and \
                    dist[r, c] < self.inact_reid_thresh:
                # move inactive track to active
                self.tracks[ids[c]] = self.inactive_tracks[ids[c]]
                del self.inactive_tracks[ids[c]]
                self.tracks[ids[c]].inactive_count = 0

                self.tracks[ids[c]].add_detection(**detections[r], save_memory=self.save_memory)
                active_tracks.append(ids[c])
                assigned.append(r)
                tr_ids[r] = ids[c]

        return set(assigned)

    def assign_separatly(
            self,
            row,
            col,
            dist,
            detections,
            active_tracks,
            ids,
            tr_ids):
        """
        Assign active and inactive one after another
        """
        # assign active tracks first
        assigned = self.assign_act_inact_same_time(
            row,
            col,
            dist[0],
            detections,
            active_tracks,
            ids[:dist[0].shape[1]],
            tr_ids)

        # assign inactive tracks
        if dist[1] is not None:
            # only use detections that have not been assigned yet
            # u = unassigned
            u = sorted(
                list(set(list(range(dist[0].shape[0]))) - assigned))

            if len(u) != 0:
                dist[1] = dist[1][u, :]

                row_inact, col_inact = solve_dense(dist[1])
                assigned_2 = self.assign_act_inact_same_time(
                    row=row_inact,
                    col=col_inact,
                    dist=dist[1],
                    detections=[t for i, t in enumerate(detections) if i in u],
                    active_tracks=active_tracks,
                    ids=ids[dist[0].shape[1]:],
                    tr_ids=tr_ids)
                assigned_2 = set(
                    [u for i, u in enumerate(u) if i in assigned_2])
                assigned.update(assigned_2)

        return assigned
    
    def get_extra_kalman_candidates(self, strack, frame_img, det_conf=0.10000001):  # With det_conf=0.10000001, we will force the "detection" having the lowest possible score for being considered for second round
        # This function is used to get the extra candidates from the Kalman Filter
        # It is used in the third round of association
        extra_kalman_candidates = []

        for t in strack:
            det_tlbr = t.tlbr.copy()

            det_conf = det_conf
            det_img = self.transformer_tracker.get_image_crops(image=frame_img, bboxes=[det_tlbr], normalize=False)
            
            det_img_tensor = self._normalize_embeddings_batch(det_img, denormalize=False)  # We normalize it so GHOST feature extractor likes it
            det_img_tensor = det_img_tensor[..., ::-1].copy()  # We convert from BGR to RGB (we do array.copy() to avoid problems with the memory)
            det_img_tensor = torch.from_numpy(det_img_tensor).to(self.device).permute(0, 3, 1, 2)
            det_feats = self.get_features(det_img_tensor)
            det_feats = det_feats[0]  # We only have one image

            det_img = det_img[0]  # We only have one image
            
            det = Track(track_id=-1, bbox=det_tlbr, feats=det_feats, im_index=self.frame_id, gt_id=-1, vis=-1, conf=det_conf,
                        frame=self.frame_id, label=t.label[-1], image=det_img, kalman=self.kalman, kalman_filter=self.kalman_filter)

            extra_kalman_candidates.append(det)
        
        return extra_kalman_candidates
    
    def get_all_considered_dets(self, detections):
        # We convert all the detections to Track objects
        all_considered_dets = []
        for det in detections:
            det_tlbr = det['bbox'].copy()
            det_conf = det['conf']
            det_img = det['image']
            det_feats = det['feats']

            if self.tracker_cfg['busca_args'].transformer_update_mems_only_first_round:
                det_conf = max(det_conf, Track._conf_thres)  # We set the confidence to the minimum that will make them appear on the Track mems

            det = Track(track_id=None, bbox=det_tlbr, feats=det_feats, im_index=None, gt_id=None, vis=None, conf=det_conf,
                        frame=None, label=None, image=det_img, kalman=False, kalman_filter=None)

            all_considered_dets.append(det)

        return all_considered_dets

    def third_round_association(self, strack_pool, considered_dets, extra_kalman_candidates, asoc_thresh):
        main_args = self.tracker_cfg['busca_args']  # For convenience
        
        # We compute BUSCA for all tracks and dets
        if asoc_thresh <= 0.0:
            matches = []
            u_track = strack_pool

            return matches, u_track

        strack_pool_pos = np.array([tr.pos for tr in strack_pool])
        considered_dets_pos = np.array([tr.pos for tr in considered_dets])
        extra_kalman_candidates_pos = np.array([tr.pos for tr in extra_kalman_candidates])


        with torch.no_grad():  # To turn off gradients computation:
            # We need to compute the distance matrix between all tracks and all dets
            tracks_dets_dists = center_distance(strack_pool_pos, considered_dets_pos)

            embedding_probs, \
            reliable_predictions = self.transformer_tracker.associate_embeddings(tracks_embeddings=strack_pool,
                                                                                 dets_embeddings=considered_dets,
                                                                                 dists_matrix=tracks_dets_dists,
                                                                                 seq_len=main_args.seq_len,
                                                                                 num_candidates=main_args.num_candidates,
                                                                                 use_broader_memory=main_args.use_broader_memory,
                                                                                 extra_kalman_candidates=extra_kalman_candidates,
                                                                                 select_highest_candidate=main_args.select_highest_candidate,
                                                                                 highest_candidate_minimum_thresh=main_args.highest_candidate_minimum_thresh if hasattr(main_args, 'highest_candidate_minimum_thresh') else None,
                                                                                 keep_highest_value=main_args.keep_highest_value if hasattr(main_args, 'keep_highest_value') else False,
                                                                                 plot_results=self.tracker_cfg['online_visualization'],
                                                                                 normalize_ims=True)

        if embedding_probs is not None:
            if hasattr(main_args, "match_dets_too") and main_args.match_dets_too:
                matches, u_track = self.recover_kalman_and_dets(embedding_probs=embedding_probs, reliable_predictions=reliable_predictions,
                                                                strack_pool=strack_pool, considered_dets=considered_dets, asoc_thresh=asoc_thresh)
            else:
                matches, u_track = self.recover_only_kalman(embedding_probs=embedding_probs, reliable_predictions=reliable_predictions,
                                                                strack_pool=strack_pool, considered_dets=considered_dets, asoc_thresh=asoc_thresh)
            
        else:
            matches = []
            u_track = strack_pool
            
        return matches, u_track
    
    def recover_only_kalman(self, embedding_probs, reliable_predictions, strack_pool, considered_dets, asoc_thresh):
        # First we will get the indices of the unmatched tracks (will be indices from 0 to strack_pool-1)
        strack_inds = [i for i in range(len(strack_pool))]

        # Then we will get the indices of the kalman predictions for those tracks
        # We assume that the Kalman predictions will be the len(strack_pool) last predictions
        num_all_dets = len(considered_dets)
        extra_kalman_inds = [i + num_all_dets for i in strack_inds]

        # We are interested only in the Kalman predictions, not the unmatched dets
        embedding_probs_kalman = []
        for t, d in zip(strack_inds, extra_kalman_inds):
            embedding_probs_kalman.append(embedding_probs[t, d])
        
        reliable_predictions_kalman = reliable_predictions[strack_inds]

        matches = []
        u_track = []
        for i, prob in enumerate(embedding_probs_kalman):
            if reliable_predictions_kalman[i] and prob > asoc_thresh:
                matches.append([i, prob, None])
            else:
                u_track.append(i)
        
        return matches, u_track
    
    def _normalize_embeddings_batch(self, embeddings_batch, denormalize=False, ghost_normalization=True):
        if ghost_normalization:
            input_pixel_mean = np.array([0.406, 0.456, 0.485])  # BGR
            input_pixel_std = np.array([0.225, 0.224, 0.299])  # BGR
        else:
            input_pixel_mean = np.array([0.406, 0.456, 0.485])  # BGR
            input_pixel_std = np.array([0.225, 0.224, 0.229])  # BGR
        
        if not denormalize:
            embeddings_batch = embeddings_batch.astype(np.float32) / 255.0
            embeddings_batch -= input_pixel_mean
            embeddings_batch /= input_pixel_std
        
        else:
            embeddings_batch *= input_pixel_std
            embeddings_batch += input_pixel_mean
            embeddings_batch *= 255.0
            embeddings_batch = embeddings_batch.astype(np.uint8)
        
        return embeddings_batch