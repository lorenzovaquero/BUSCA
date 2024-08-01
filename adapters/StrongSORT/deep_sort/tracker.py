# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .detection import Detection
from opts import opt

import os
import torch
import cv2

from busca.tracking import center_distance
from busca.network import BUSCA

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, tracker_cfg=None):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.tracks = []
        self._next_id = 1

        if tracker_cfg is None:
            self.tracker_cfg = {}
        else:
            self.tracker_cfg = tracker_cfg

        # vvv *BUSCA* vvv
        if hasattr(self.tracker_cfg, 'use_busca') and self.tracker_cfg.use_busca is True:
            self.use_busca = True
        else:
            self.use_busca = False
        
        self.save_memory = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        if self.use_busca:
            
            busca_args = self.tracker_cfg.busca_args.transformer  # For convenience

            busca_args.device = torch.device("cuda:0" if self.tracker_cfg.busca_args.device == 'gpu' and torch.cuda.is_available() else "cpu")
            
            self.busca_tracker = BUSCA(busca_args).to(busca_args.device)
            self.busca_tracker.load_pretrained(self.tracker_cfg.busca_ckpt, ignore_reid_fc=True)
            self.busca_tracker.eval()

            if self.tracker_cfg.busca_args.transformer_update_mems_only_first_round:
                # We only want to update the mems of the first round
                # But as StrongSORT needs to update some of its stuff (and there is overlapping),
                # we apply a filter so BUSCA only sees the first-rouns (high-confidence) detections
                busca_mem_thresh = self.tracker_cfg.min_confidence + self.tracker_cfg.busca_args.minimum_conf_modifier
                Track.set_busca_conf_threshold(conf_thres=busca_mem_thresh)
            
            if hasattr(self.tracker_cfg.busca_args, 'avoid_memory_leak') and self.tracker_cfg.busca_args.avoid_memory_leak:
                print('WARNING: Enabling StrongSORT memory saving mode!', flush=True)
                self.save_memory = True
            
        else:
            self.busca_tracker = None
        # ^^^ *BUSCA* ^^^

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def camera_update(self, video, frame):
        for track in self.tracks:
            track.camera_update(video, frame)

    def update(self, detections, image_file=None):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        
        # vvv *BUSCA* vvv
        # We add images to detections
        if hasattr(self.tracker_cfg, 'use_busca') and self.tracker_cfg.use_busca:
            print('Current frame: {}'.format(image_file))
            current_frame = cv2.imread(image_file, cv2.IMREAD_COLOR)
            for det in detections:
                det_tlbr = det.to_tlbr().copy()
                det.add_image(self.busca_tracker.get_image_crops(image=current_frame, bboxes=[det_tlbr], normalize=False)[0])


        ''' vvv Third association, between Kalman and Tracks vvv'''
        # The goal of this association is to refine `u_track` (unmatched tracks after second round) and check if we really have to deactivate them
        third_round_indices = []
        third_round_stracks = []

        for track_idx in unmatched_tracks:
            t = self.tracks[track_idx]
            if not t.is_confirmed() or t.time_since_update > 1:  # We are only interested in the active tracks
                continue
            third_round_indices.append(track_idx)
            third_round_stracks.append(t)

        if hasattr(self.tracker_cfg, 'use_busca') and self.tracker_cfg.use_busca and hasattr(self.tracker_cfg.busca_args, 'busca_thresh') and self.tracker_cfg.busca_args.busca_thresh > 0 and len(third_round_indices) > 0:
            main_args = self.tracker_cfg.busca_args  # For convenience
        
            # Unreliable dets
            if hasattr(main_args, 'reliable_thresh') and not self.is_reliable(current_frame=current_frame, active_stracks=self.tracked_stracks, p=main_args.reliable_thresh):
                third_round_stracks = []
                
            else:
                extra_kalman_candidates = self.get_extra_kalman_candidates(strack=third_round_stracks, frame_img=current_frame,
                                                                           det_conf=Track._conf_thres)  # We set the confidence to the minimum that will make them appear on the Track mems

                all_considered_dets = self.get_all_considered_dets(detections, frame_img=current_frame)
                
                third_round_matches, third_round_u_track = self.third_round_association(strack_pool=third_round_stracks, considered_dets=all_considered_dets,
                                                                                        extra_kalman_candidates=extra_kalman_candidates, asoc_thresh=main_args.busca_thresh,
                                                                                        update_vis_info=None, previous_strack_pool=None)
                for itracked, prob, idet in third_round_matches:
                    track = third_round_stracks[itracked]
                    track_idx = third_round_indices[itracked]

                    if idet is None:  # It means it is kalman prediction
                        # We just associate it, like normal BUSCA
                        det = extra_kalman_candidates[itracked]  # itracked and "idet" are the same, as `third_round_stracks` and `extra_kalman_candidates` refer to the same tracks

                    else:
                        raise NotImplementedError('Third round with BUSCA and DETS is not implemented yet')
                    

                    if main_args.transformer_update_mems_only_first_round:
                        if main_args.update_feats_third_round:
                            new_feats = det.features[-1]
                        else:
                            new_feats = track.features[-1]

                        new_img = track.images_mem[-1]  # Although it should never be used
                        new_conf = 0.10000001  # So it does not appear in the mems
                    else:
                        new_feats = det.features[-1]
                        new_img = det.image
                        new_conf = det.conf  # It will be the same as `self.tracker_cfg['det_conf']
                    
                    det_update = Detection(tlwh=det.tlwh, confidence=new_conf, feature=new_feats, image=new_img)
                    track.update(det_update)
                        
                        
                    # I remove the track from `unmatched_tracks`
                    unmatched_tracks.remove(track_idx)

        ''' ^^^ Third association, between Kalman and Tracks ^^^ '''
        # ^^^ *BUSCA* ^^^

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            if not opt.EMA:
                track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, self.n_init, self.max_age,
            detection.feature, detection.confidence,
            scale=1.0, frame_id=None, image=detection.image))  # For *BUSCA*
        self._next_id += 1


    
    def get_extra_kalman_candidates(self, strack, frame_img, det_conf=0.10000001):  # With det_conf=0.10000001, we will force the "detection" having the lowest possible score for being considered for second round
        # This function is used to get the extra candidates from the Kalman Filter
        # It is used in the third round of association
        extra_kalman_candidates = []

        for t in strack:
            det_tlbr = t.tlbr.copy()
            det_xyah = t.xyah.copy()

            det_conf = det_conf
            det_img = self.busca_tracker.get_image_crops(image=frame_img, bboxes=[det_tlbr], normalize=False)
            det_img = det_img[0]  # We only have one image
            
            det_feats = None
            
            det = Track(detection=det_xyah, track_id=-1, n_init=self.n_init, max_age=self.max_age, feature=det_feats, score=det_conf, scale=1.0, frame_id=None, image=det_img)

            extra_kalman_candidates.append(det)
        
        return extra_kalman_candidates
    
    def get_all_considered_dets(self, detections, frame_img):
        # We convert all the detections to Track objects
        all_considered_dets = []
        for det in detections:
            det_tlbr = det.to_tlbr().copy()
            det_xyah = det.to_xyah().copy()
            det_conf = det.confidence
            det_img = self.busca_tracker.get_image_crops(image=frame_img, bboxes=[det_tlbr], normalize=False)
            det_img = det_img[0]  # We only have one image
            det_feats = det.feature

            if self.tracker_cfg.busca_args.transformer_update_mems_only_first_round:
                det_conf = max(det_conf, Track._conf_thres)  # We set the confidence to the minimum that will make them appear on the Track mems
            
            det = Track(detection=det_xyah, track_id=-1, n_init=self.n_init, max_age=self.max_age, feature=det_feats, score=det_conf, scale=1.0, frame_id=None, image=det_img)

            all_considered_dets.append(det)

        return all_considered_dets

    def third_round_association(self, strack_pool, considered_dets, extra_kalman_candidates, asoc_thresh, update_vis_info=None, previous_strack_pool=None):
        main_args = self.tracker_cfg.busca_args  # For convenience
        
        if asoc_thresh <= 0.0:
            matches = []
            u_track = strack_pool
            if update_vis_info is not None:
                update_vis_info['transformer_asoc_all'] = {}

            return matches, u_track

        strack_pool_pos = np.array([tr.pos for tr in strack_pool])
        considered_dets_pos = np.array([tr.pos for tr in considered_dets])
        extra_kalman_candidates_pos = np.array([tr.pos for tr in extra_kalman_candidates])

        with torch.no_grad():  # To turn off gradients computation:
            # To compute the multicandidate Transformer, we need to compute the distance matrix between all tracks and all dets
            tracks_dets_dists = center_distance(strack_pool_pos, considered_dets_pos)

            embedding_probs, \
            reliable_predictions = self.busca_tracker.associate_embeddings(tracks_embeddings=strack_pool,
                                                                            dets_embeddings=considered_dets,
                                                                            dists_matrix=tracks_dets_dists,
                                                                            seq_len=main_args.seq_len,
                                                                            num_candidates=main_args.num_candidates,
                                                                            use_broader_memory=main_args.use_broader_memory,
                                                                            extra_kalman_candidates=extra_kalman_candidates,
                                                                            select_highest_candidate=main_args.select_highest_candidate,
                                                                            highest_candidate_minimum_thresh=main_args.highest_candidate_minimum_thresh if hasattr(main_args, 'highest_candidate_minimum_thresh') else None,
                                                                            keep_highest_value=main_args.keep_highest_value if hasattr(main_args, 'keep_highest_value') else False,
                                                                            plot_results=self.tracker_cfg.online_visualization,
                                                                            normalize_ims=True)

        if embedding_probs is not None:
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
