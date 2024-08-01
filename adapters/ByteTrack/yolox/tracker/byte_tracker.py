import numpy as np
from collections import deque
import os
import cv2
import copy
import torch
import torch.nn.functional as F

try:
    from .kalman_filter import KalmanFilter
    from yolox.tracker import matching
    from .basetrack import BaseTrack, TrackState
except ImportError:
    print("Some modules were not found. This may be because byte_tracker.py has been imported from another project (e.g. CenterTrack). Resorting to secondary paths.")
    from .mot_online.kalman_filter import KalmanFilter
    from .mot_online import matching
    from .mot_online.basetrack import BaseTrack, TrackState

from busca.network import BUSCA
from busca.tracking import center_distance
from busca.visualization import plot_box

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, image=None, scale=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.scale = scale  # We won't keep a historic of this value. We will only keep the last one.
        self.tracklet_len = 0
   
        self.tlwh_mem = []
        self.tlwh_mem.append(self._tlwh.copy())
        
        self.images_mem = []
        if image is not None:
            self.images_mem.append(image)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, update_mems=True):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.scale = new_track.scale

        if len(new_track.tlwh_mem) > 0 and update_mems:
            self.tlwh_mem.extend(new_track.tlwh_mem)
        
        if len(new_track.images_mem) > 0 and update_mems:
            self.images_mem.extend(new_track.images_mem)
        

    def update(self, new_track, frame_id, update_mems=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.scale = new_track.scale

        if len(new_track.tlwh_mem) > 0 and update_mems:
            self.tlwh_mem.extend(new_track.tlwh_mem)
        
        if len(new_track.images_mem) > 0 and update_mems:
            self.images_mem.extend(new_track.images_mem)
    
    def apply_camera_motion(self, warp_matrix):
        """Applies the camera motion to the tracklet"""
        if self.mean is None:
            pos =  self._tlwh[:2].copy()
        else:
            pos = self.mean[:2].copy()
        
        pos = pos * self.scale  # We apply the scale to the frame size
        new_pos = BYTETracker.warp_pos(pos=pos, warp_matrix=warp_matrix)
        new_pos = new_pos / self.scale  # We apply the scale to the ByteTrack size

        # We now update the 8-dimensional state space (x, y, a, h, vx, vy, va, vh)
        if self.mean is None:
            self._tlwh[:2] = new_pos
        else:
            self.mean[:2] = new_pos

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args

        if hasattr(self.args, 'use_busca') and self.args.use_busca is True:
            self.use_busca = True
        else:
            self.use_busca = False

        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.last_image = None

        if self.use_busca:
            busca_args = self.args.transformer
            busca_args.device = torch.device("cuda" if self.args.device == "gpu" else "cpu")
            self.busca_tracker = BUSCA(busca_args).to(busca_args.device)
            self.busca_tracker.load_pretrained(self.args.busca_ckpt, ignore_reid_fc=True)
            self.busca_tracker.eval()
        else:
            self.busca_tracker = None

    def update(self, output_results, img_info, img_size, current_frame=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:  # Aggregates obj_conf and class_conf
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale  # If the image was rescaled (for the detector), the bboxes are updated

        remain_inds = scores > self.args.track_thresh  # Indices for first-round detections

        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)  # Indices for second-round detections
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]

        inds_all_considered = np.logical_or(remain_inds, inds_second)  # Indices for all considered detections (1st+2nd). Excludes dets with score < 0.1
        bboxes_all_considered = bboxes[inds_all_considered]
        scores_all_considered = scores[inds_all_considered]
        inds_all_considered_1st = remain_inds[inds_all_considered]  # Indices for which elements from all_considered belong to 1st round
        inds_all_considered_2nd = inds_second[inds_all_considered]  # Indices for which elements from all_considered belong to 2nd round

        # We will also link the idet_first with idet_all_considered
        inds_all_considered_1st_idet = [None] * len(inds_all_considered_1st)
        counter = 0
        for i, is_first in enumerate(inds_all_considered_1st):
            if is_first:
                inds_all_considered_1st_idet[i] = counter
                counter += 1
        
        # We will also link the idet_second with idet_all_considered
        inds_all_considered_2nd_idet = [None] * len(inds_all_considered_2nd)
        counter = 0
        for i, is_second in enumerate(inds_all_considered_2nd):
            if is_second:
                inds_all_considered_2nd_idet[i] = counter
                counter += 1

        if self.use_busca and hasattr(self.args, 'busca_thresh') and self.args.busca_thresh > 0:
            # We extract the images of the detections
            images_second = self.busca_tracker.get_image_crops(image=current_frame, bboxes=dets_second*scale, normalize=False)
            images_first = self.busca_tracker.get_image_crops(image=current_frame, bboxes=dets*scale, normalize=False)
            images_all_considered = self.busca_tracker.get_image_crops(image=current_frame, bboxes=bboxes_all_considered*scale, normalize=False)

        else:
            images_second = [None] * len(dets_second)
            images_first = [None] * len(dets)
            images_all_considered = [None] * len(bboxes_all_considered)

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, image=im, scale=scale) for
                          (tlbr, s, im) in zip(dets, scores_keep, images_first)]
        else:
            detections = []
        
        if len(bboxes) > 0:
            '''Both first-round and second-round detections'''
            all_considered_dets = [STrack(STrack.tlbr_to_tlwh(tlbr), s, image=im, scale=scale) for
                                    (tlbr, s, im) in zip(bboxes_all_considered, scores_all_considered, images_all_considered)]
        else:
            all_considered_dets = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                update_mems = True
                if detections[idet].score < self.det_thresh:
                    update_mems = False

                track.update(detections[idet], self.frame_id, update_mems=update_mems)
                activated_starcks.append(track)
            else:
                update_mems = True
                if det.score < self.det_thresh:
                    update_mems = False

                track.re_activate(det, self.frame_id, new_id=False, update_mems=update_mems)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, image=im, scale=scale) for
                          (tlbr, s, im) in zip(dets_second, scores_second, images_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        r_lost_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state != TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                update_mems = not (hasattr(self.args, 'transformer_update_mems_only_first_round') and self.args.transformer_update_mems_only_first_round)
                track.update(det, self.frame_id, update_mems=update_mems)
                activated_starcks.append(track)
            else:
                # This branch never runs on original ByteTrack, as only Tracked tracks are considered in `r_tracked_stracks` (second round detections can't reactivate Lost tracks)
                assert False, "This branch should never run in original ByteTrack!"
                track.re_activate(det, self.frame_id, new_id=False, update_mems=not self.args.transformer_update_mems_only_first_round)
                refind_stracks.append(track)
        
        unassigned_stracks = joint_stracks([r_tracked_stracks[it] for it in u_track], r_lost_stracks)  # These are the active tracks that are not matched with any detection (not even with second round detections) and inactive tracks that didn't match a first round detection
        u_track = list(range(len(unassigned_stracks)))

        ''' vvv Step 3b (BUSCA): Third association, between Kalman and BUSCA vvv'''
        # The goal of this association is to refine `u_track` (unmatched tracks after second round) and check if we really have to deactivate them
        if self.use_busca and hasattr(self.args, 'busca_thresh') and self.args.busca_thresh > 0:
            # Unreliable dets
            if hasattr(self.args, 'reliable_thresh') and not self.is_reliable(current_frame=current_frame, active_stracks=self.tracked_stracks, p=self.args.reliable_thresh):
                third_round_stracks = []

            else:
                third_round_stracks = unassigned_stracks

                if self.args.use_camera_motion_compensation:
                    cc = self.camera_motion_compensation(track_pool=third_round_stracks, current_frame=current_frame)

                extra_kalman_candidates = self.get_extra_kalman_candidates(strack_pool=third_round_stracks, frame_img=current_frame)

                third_round_matches, third_round_u_track = self.third_round_association(strack_pool=third_round_stracks, considered_dets=all_considered_dets,
                                                                                        extra_kalman_candidates=extra_kalman_candidates, asoc_thresh=self.args.busca_thresh)

                for itracked, prob in third_round_matches:
                    track = third_round_stracks[itracked]
                    det = extra_kalman_candidates[itracked]  # itracked and "idet" are the same, as `third_round_stracks` and `extra_kalman_candidates` refer to the same tracks
                    
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id, update_mems=False)
                        activated_starcks.append(track)
                
                u_track = third_round_u_track

        else:
            third_round_stracks = []
        ''' ^^^ Step 3b (BUSCA): Third association, between Kalman and Transformer ^^^ '''

        for it in u_track:
            track = unassigned_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, update_mems=True)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        fix_leak = True
        if fix_leak:
            self.removed_stracks = [track for track in self.removed_stracks if self.frame_id - track.end_frame < 10 * self.max_time_lost]
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
       
        self.last_image = np.copy(current_frame) if current_frame is not None else None
        
        if hasattr(self.args, 'online_visualization') and self.args.online_visualization:
            wait_for_key = True
            save_imgs_file_frame = None

            self.visualize_online_tracking(frame=current_frame, active_stracks=output_stracks, inactive_stracks=lost_stracks, wait_for_key=wait_for_key, save_file=save_imgs_file_frame)

        return output_stracks


    def is_reliable(self, current_frame, active_stracks, p):
        det_cov = self.get_detection_coverage(frame=current_frame, active_stracks=active_stracks, inactive_stracks=[])

        if det_cov['area_covered'] > det_cov['area_covered_per_obj'] * p[0] + p[1]:
            return True
        else:
            return False


    def get_extra_kalman_candidates(self, strack_pool, frame_img, det_conf=0.10000001):  # With det_conf=0.10000001, we will force the "detection" having the lowest possible score for being considered for second round
        # This function is used to get the extra candidates from the Kalman Filter
        # It is used in the third round of association
        extra_kalman_candidates = []
        for track in strack_pool:
            det_tlwh = track.tlwh
            det_score = np.float32(det_conf)
            det_img = self.busca_tracker.get_image_crops(image=frame_img, bboxes=[track.tlbr * track.scale], normalize=False)[0]
            det = STrack(tlwh=det_tlwh, score=det_score, image=det_img, scale=track.scale)
            extra_kalman_candidates.append(det)
        
        return extra_kalman_candidates
    
    def third_round_association(self, strack_pool, considered_dets, extra_kalman_candidates, asoc_thresh):
        if asoc_thresh <= 0.0:
            matches = []
            u_track = strack_pool
            return matches, u_track

        with torch.no_grad():  # To turn off gradients computation:
            # To compute the multicandidate Transformer, we need to compute the distance matrix between all tracks and all dets
            tracks_dets_dists = center_distance(strack_pool, considered_dets)

            embedding_probs, \
            reliable_predictions = self.busca_tracker.associate_embeddings(tracks_embeddings=strack_pool,
                                                                                 dets_embeddings=considered_dets,
                                                                                 dists_matrix=tracks_dets_dists,
                                                                                 seq_len=self.args.seq_len,
                                                                                 num_candidates=self.args.num_candidates,
                                                                                 use_broader_memory=self.args.use_broader_memory,
                                                                                 extra_kalman_candidates=extra_kalman_candidates,
                                                                                 select_highest_candidate=self.args.select_highest_candidate,
                                                                                 highest_candidate_minimum_thresh=self.args.highest_candidate_minimum_thresh if hasattr(self.args, 'highest_candidate_minimum_thresh') else None,
                                                                                 plot_results=True if hasattr(self.args, 'online_visualization') and self.args.online_visualization else False,
                                                                                 normalize_ims=True)

        if embedding_probs is not None:
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
                    matches.append([i, prob])
                else:
                    u_track.append(i)
            
        else:
            matches = []
            u_track = strack_pool
            
        return matches, u_track
    
    
    def visualize_online_tracking(self, frame, active_stracks, inactive_stracks, display_ids=False, max_size=960, wait_for_key=False, window_name='Tracking', save_file=None):
        frame = frame.astype(np.uint8)  # It copies the original image so we do not draw the rectangles on top of it

        # We compute it first so we can scale labels
        if max_size is not None and max_size > 0:
            frame_h, frame_w, _ = frame.shape
            resize_ratio = min(min(max_size/frame_h, max_size/frame_w), 1.0)
        
        else:
            resize_ratio = 1.0

        # We will plot bboxes for all active tracks
        for track in active_stracks:
            target_id = track.track_id
            target_bbox = np.array(track.tlbr) * track.scale
            plot_box(frame_image=frame, target_id=target_id, target_bbox=target_bbox, style='solid', thickness=2, display_id=display_ids, id_size=1/resize_ratio)
    
        # We will plot bboxes for all inactive tracks
        for track in inactive_stracks:
            target_id = track.track_id
            target_bbox = np.array(track.tlbr) * track.scale
            plot_box(frame_image=frame, target_id=target_id, target_bbox=target_bbox, style='dashed', thickness=2, display_id=display_ids, id_size=1/resize_ratio)
        
        if max_size is not None and max_size > 0:
            frame = cv2.resize(frame, (int(frame_w*resize_ratio), int(frame_h*resize_ratio)))

        if save_file is not None:
            cv2.imwrite(save_file, frame)
            return_value = 0

        else:
            cv2.imshow(window_name, frame)
            return_value = cv2.waitKey(0 if wait_for_key else 1)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0:  # If the "x" is pressed
                return_value = 27
        
        return return_value

    def get_detection_coverage(self, frame, active_stracks, inactive_stracks):
        # We will compute the area covered by the detections
        # We will use the active and inactive tracks to compute the area covered by the detections
        # To do this, we will plot bboxes on top of a black image and then we will count the number of non-black pixels
        base_frame = np.zeros_like(frame).astype(np.uint8)
        num_objs = 0
        max_bbox_area = 0.0
        bbox_areas = []

        # We will plot bboxes for all active tracks
        for track in active_stracks:
            target_bbox = np.array(track.tlbr) * track.scale
            cv2.rectangle(base_frame, (int(target_bbox[0]), int(target_bbox[1])), (int(target_bbox[2]), int(target_bbox[3])), (255, 255, 255), thickness=-1)
            num_objs += 1

            # We will get the bbox area with respect to the frame size
            bbox_area = max(min(((target_bbox[2] - target_bbox[0])/base_frame.shape[0]) * ((target_bbox[3] - target_bbox[1])/base_frame.shape[1]), 1.0), 0.0)
            max_bbox_area = max(max_bbox_area, bbox_area)
            bbox_areas.append(bbox_area)
            
        
        # We will plot bboxes for all inactive tracks
        for track in inactive_stracks:
            target_bbox = np.array(track.tlbr) * track.scale
            cv2.rectangle(base_frame, (int(target_bbox[0]), int(target_bbox[1])), (int(target_bbox[2]), int(target_bbox[3])), (255, 255, 255), thickness=-1)
            num_objs += 1

            # We will get the bbox area with respect to the frame size
            bbox_area = max(min(((target_bbox[2] - target_bbox[0])/base_frame.shape[0]) * ((target_bbox[3] - target_bbox[1])/base_frame.shape[1]), 1.0), 0.0)
            max_bbox_area = max(max_bbox_area, bbox_area)
            bbox_areas.append(bbox_area)

        # We reduce the image to a single channel
        base_frame = base_frame[:, :, 0]

        # We count the number of non-black pixels
        num_non_black_pixels = np.count_nonzero(base_frame)

        # We compute the percentage of the image covered by the detections
        percentage_covered = num_non_black_pixels / (base_frame.shape[0] * base_frame.shape[1])

        # We also compute the average area covered by each object
        if num_objs > 0:
            avg_area_covered = percentage_covered / num_objs
            average_bbox_area = np.sqrt(np.array(bbox_areas)).mean() ** 2
        else:
            avg_area_covered = 0.0
            average_bbox_area = 0.0

        return {'area_covered': percentage_covered, 'area_covered_per_obj': avg_area_covered, 'max_bbox_area': max_bbox_area, 'average_bbox_area': average_bbox_area, 'bbox_areas': bbox_areas}
    
    # https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/experiments/cfgs/tracktor.yaml
    def camera_motion_compensation(self, track_pool, current_frame, number_of_iterations=100, termination_eps=0.00001, warp_mode='MOTION_EUCLIDEAN'):
        """Aligns the positions of tracks depending on camera motion."""
        cc = 1.0
        if self.frame_id > 1:
            im1 = self.last_image
            im2 = current_frame

            if warp_mode == 'MOTION_EUCLIDEAN':
                warp_mode = cv2.MOTION_EUCLIDEAN
            elif warp_mode == 'MOTION_AFFINE':
                warp_mode = cv2.MOTION_AFFINE
            else:
                raise ValueError('Invalid warp_mode: {}'.format(warp_mode))
            
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
            cc, warp_matrix = cv2.findTransformECC(templateImage=im1_gray, inputImage=im2_gray, warpMatrix=warp_matrix, motionType=warp_mode, criteria=criteria)
            warp_matrix = torch.from_numpy(warp_matrix)

            for t in track_pool:
                t.apply_camera_motion(warp_matrix)

        return cc  # Correlation coefficient

    
    @staticmethod
    def warp_pos(pos, warp_matrix):  # Pos are (x1, y1)
        p1 = torch.Tensor([pos[0], pos[1], 1]).view(3, 1)
        p1_n = torch.mm(warp_matrix, p1).view(1, 2)
        return p1_n[0]


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
