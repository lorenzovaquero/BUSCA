# vim: expandtab:ts=4:sw=4
import numpy as np
from deep_sort.kalman_filter import KalmanFilter
from opts import opt

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    _conf_thres = 1.0  # We add this so we can filter MEMs in *BUSCA*

    @classmethod
    def set_busca_conf_threshold(cls, conf_thres):
        cls._conf_thres = conf_thres
        print("Setting BUSCA conf threshold for class Track to {}".format(Track._conf_thres), flush=True)

    def __init__(self, detection, track_id, n_init, max_age,
                 feature=None, score=None,
                 scale=1.0, frame_id=None, image=None):  # For *BUSCA*
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            feature /= np.linalg.norm(feature)
            self.features.append(feature)

        self.scores = []
        if score is not None:
            self.scores.append(score)

        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilter()

        self.mean, self.covariance = self.kf.initiate(detection)

        #  vvv *BUSCA* vvv
        self.scale = scale

        self._tlwh_mem = []
        self._tlwh_mem.append(self.to_tlwh())
        
        self.image = image
        self._images_mem = []
        if image is not None:
            self._images_mem.append(image)
        
        self.conf_mem = []  # I add this so BUSCA can pick only the detections with high conf
        self.conf_mem.append(score)
        #  ^^^ *BUSCA* ^^^


    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    # @jit(nopython=True)
    def tlwh(self):
        return self.to_tlwh()

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    @property
    # @jit(nopython=True)
    def tlbr(self):
        return self.to_tlbr()
    
    @property
    # @jit(nopython=True)
    def pos(self):
        # ByteTrack defines pos as tlbr
        return self.to_tlbr()
    
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    # @jit(nopython=True)
    def xyah(self):
        return self.to_xyah()
    
    @property
    # @jit(nopython=True)
    def tlwh_mem(self):
        """Get "tlwh_mem" much like we do in ByteTrack, but relying on the stored positions by StrongSORT
        """

        assert len(self.conf_mem) == len(self._tlwh_mem), "conf_mem and _tlwh_mem must have the same length"
        
        tlwh_mem = []
        for i, conf in enumerate(self.conf_mem):
            if conf >= Track._conf_thres:
                tlwh_mem.append(self._tlwh_mem[i])

        return tlwh_mem
    
    @property
    # @jit(nopython=True)
    def images_mem(self):
        assert len(self.conf_mem) == len(self._images_mem), "conf_mem and _images_mem must have the same length"
        
        images_mem = []
        for i, conf in enumerate(self.conf_mem):
            if conf >= Track._conf_thres:
                images_mem.append(self._images_mem[i])
        
        return images_mem

    def predict(self):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def get_matrix(dict_frame_matrix, frame):
        eye = np.eye(3)
        matrix = dict_frame_matrix[frame]
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def camera_update(self, video, frame):
        dict_frame_matrix = opt.ecc[video]
        frame = str(int(frame))
        if frame in dict_frame_matrix:
            matrix = self.get_matrix(dict_frame_matrix, frame)
            x1, y1, x2, y2 = self.to_tlbr()
            x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
            x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
            w, h = x2_ - x1_, y2_ - y1_
            cx, cy = x1_ + w / 2, y1_ + h / 2
            self.mean[:4] = [cx, cy, w / h, h]

    def update(self, detection, frame_id=None, save_memory=False):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.to_xyah(), detection.confidence)

        feature = detection.feature / np.linalg.norm(detection.feature)
        if opt.EMA:
            smooth_feat = opt.EMA_alpha * self.features[-1] + (1 - opt.EMA_alpha) * feature
            smooth_feat /= np.linalg.norm(smooth_feat)
            self.features = [smooth_feat]
        else:
            self.features.append(feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        
        self._tlwh_mem.append(detection.tlwh)
        self.conf_mem.append(detection.confidence)
        
        image = detection.image
        self.image = image
        if image is not None:
            if not save_memory:
                self._images_mem.append(image)
            else:  # For MOT20-03
                if self.conf >= Track._conf_thres:
                    self._images_mem.append(image)
                else:
                    self._images_mem.append(None)
            

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
