import numpy as np
from .mot_online.kalman_filter import KalmanFilter
from .byte_tracker import BYTETracker
from .mot_online import matching


class Tracker(BYTETracker):
    def __init__(self, args, frame_rate=30):
        args.mot20 = True
        super().__init__(args, frame_rate)
        self.det_thresh = args.new_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.reset()
    
    # below has no effect to final output, just to be compatible to codebase
    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh and item['class'] == 1:
                self.id_count += 1
                item['active'] = 1
                item['age'] = 1
                item['tracking_id'] = self.id_count
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)
    
    def reset(self):
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.tracks = []    
        
        # below has no effect to final output, just to be compatible to codebase               
        self.id_count = 0
        
    def step(self, results, public_det=None, img_info=None, img_size=None, current_frame=None):
        # We need to convert results to the format of output_results (the one that byte_tracker expects)
        ped_items = [item for item in results if item['class'] == 1]
        if len(ped_items) == 0:
            scores = np.array([], np.float32)
            bboxes = np.zeros((0, 4), np.float32)
            bytetrack_results = np.zeros((0, 5), np.float32)
        
        else:
            scores = np.array([item['score'] for item in results if item['class'] == 1], np.float32)
            bboxes = np.vstack([item['bbox'] for item in results if item['class'] == 1])  # N x 4, x1y1x2y2
        
            bytetrack_results = []  # [x1, y1, x2, y2. score]
            for i in range(len(scores)):
                bytetrack_results.append([bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], scores[i]])
            
            bytetrack_results = np.array(bytetrack_results, np.float32)

        output_stracks = super().update(output_results=bytetrack_results, img_info=img_info, img_size=img_size, current_frame=current_frame)

        ret = []
        for track in output_stracks:
            track_dict = {}
            track_dict['score'] = track.score
            track_dict['bbox'] = track.tlbr
            bbox = track_dict['bbox']
            track_dict['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            track_dict['active'] = 1 if track.is_activated else 0
            track_dict['tracking_id'] = track.track_id
            track_dict['class'] = 1
            ret.append(track_dict)
        
        self.tracks = ret

        return ret
        

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


def remove_fp_stracks(stracksa, n_frame=10):
    remain = []
    for t in stracksa:
        score_5 = t.score_list[-n_frame:]
        score_5 = np.array(score_5, dtype=np.float32)
        index = score_5 < 0.45
        num = np.sum(index)
        if num < n_frame:
            remain.append(t)
    return remain

