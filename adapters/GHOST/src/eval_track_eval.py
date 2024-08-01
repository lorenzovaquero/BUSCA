from TrackEvalForGHOST import trackeval
import argparse
import os.path as osp


def get_dict(mot_dir, detectors):
    if "MOT17" in mot_dir and detectors != 'all':
        seq_length = {
            'MOT17-02-FRCNN': None, 'MOT17-04-FRCNN': None,
            'MOT17-05-FRCNN': None, 'MOT17-09-FRCNN': None,
            'MOT17-10-FRCNN': None, 'MOT17-11-FRCNN': None,
            'MOT17-13-FRCNN': None}
    elif "MOT17" in mot_dir and detectors == 'all':
        seq_length = {
            'MOT17-02-FRCNN': None, 'MOT17-04-FRCNN': None,
            'MOT17-05-FRCNN': None, 'MOT17-09-FRCNN': None,
            'MOT17-10-FRCNN': None, 'MOT17-11-FRCNN': None,
            'MOT17-13-FRCNN': None,
            'MOT17-02-DPM': None, 'MOT17-04-DPM': None,
            'MOT17-05-DPM': None, 'MOT17-09-DPM': None,
            'MOT17-10-DPM': None, 'MOT17-11-DPM': None,
            'MOT17-13-DPM': None,
            'MOT17-02-SDP': None, 'MOT17-04-SDP': None,
            'MOT17-05-SDP': None, 'MOT17-09-SDP': None,
            'MOT17-10-SDP': None, 'MOT17-11-SDP': None,
            'MOT17-13-SDP': None}
    else:
        seq_length = {
            'MOT20-01': None, 'MOT20-02': None,
            'MOT20-03': None, 'MOT20-05': None}
    # seq_length = {
    #         'MOT17-13-FRCNN': None}
    return seq_length


def setup_trackeval():
    # default eval config
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False

    # default dataset config
    default_dataset_config = \
        trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()

    # default metrics config
    default_metrics_config = {
        'METRICS': [
            'HOTA',
            'CLEAR',
            'Identity'],
        'THRESHOLD': 0.5}

    config = {**default_eval_config, **default_dataset_config,
              **default_metrics_config}  # Merge default configs

    # get updated config dicts
    eval_config = {
        k: v for k,
        v in config.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k,
        v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {
        k: v for k,
        v in config.items() if k in default_metrics_config.keys()}

    return eval_config, dataset_config, metrics_config


def evaluate_track_eval(dir, tracker, dataset_cfg, gt_path, log=True, trackers_folder=None):
    # set config
    eval_config, dataset_config, metrics_config = setup_trackeval()

    dataset_config['GT_FOLDER'] = gt_path
    if trackers_folder is not None:
        dataset_config['TRACKERS_FOLDER'] = trackers_folder
    else:
        dataset_config['TRACKERS_FOLDER'] = 'out'
    dataset_config['TRACKERS_TO_EVAL'] = [tracker.experiment]
    dataset_config['OUTPUT_FOLDER'] = 'track_eval_output'
    dataset_config['PRINT_CONFIG'] = False
    eval_config['PRINT_CONFIG'] = False
    if 'Dance' not in gt_path:
        dataset_config['SEQ_INFO'] = get_dict(
            dataset_cfg['mot_dir'], dataset_cfg['detector'])
    else:
        if 'debug' in dataset_cfg['splits']:
            dataset_config['SEQMAP_FILE'] = '/storage/user/seidensc/datasets/DanceTrack/val_seqmap_debug.txt'
        else:
            dataset_config['SEQMAP_FILE'] = '/storage/user/seidensc/datasets/DanceTrack/val_seqmap.txt'
        dataset_config['SPLIT_TO_EVAL'] = 'val'
    dataset_config['SKIP_SPLIT_FOL'] = True

    dataset_config['TRACKER_SUB_FOLDER'] = ''
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['TIME_PROGRESS'] = False
    eval_config['USE_PARALLEL'] = True
    eval_config['NUM_PARALLEL_CORES'] = 8
    metrics_config['PRINT_CONFIG'] = False

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    if not log:
        evaluator.config['PRINT_RESULTS'] = False
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = []
    for metric in [
            trackeval.metrics.HOTA,
            trackeval.metrics.CLEAR,
            trackeval.metrics.Identity]:
        # trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    return output_res, output_msg
