from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from tracking.mot_evaluator import MOTEvaluator

from busca.option import load_args_from_config, merge_args

import argparse
import os
import random
import warnings
import importlib
import sys


def make_parser():
    parser = argparse.ArgumentParser("TransCenter Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # I add this to override the default output_dir
    parser.add_argument(
        "--output-dir", default=None, type=str, help="output directory (default is 'YOLOX_outputs')"
    )

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )

    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    parser.add_argument("--ignore-vertical-thresh", default=False, action="store_true", help="ignore vertical thresh")
    # *BUSCA* args
    parser.add_argument("--online-visualization",  default=False, action="store_true", help="visualize tracking online")
    parser.add_argument("--use-busca", default=False, action="store_true", help="use BUSCA to help with detections association")
    parser.add_argument("--busca-config", default="config/ByteTrack/MOT17/config_bytetrack_mot17.yml", type=str, help="config file for BUSCA")
    parser.add_argument("--busca-ckpt", default="models/BUSCA/motsynth/model_busca.pth", type=str, help="BUSCA ckpt for transformer")
    
    return parser




def get_exp(exp_file):
    print('My exp_file is: ', exp_file, flush=True)
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contain class named 'Exp'".format(exp_file))
    return exp



@logger.catch
def main(exp, args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    # set environment variables for distributed training
    cudnn.benchmark = True

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

    for param_k in list(exp.__dict__):
        if param_k.startswith("__"):
            continue
        if getattr(exp, param_k) is not None and hasattr(args, param_k):
            setattr(args, param_k, getattr(exp, param_k))

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    #logger.info("Model Structure:\n{}".format(str(model)))

    val_loader = exp.get_eval_loader(1, False, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(0)
    model.cuda(0)
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    loc = "cuda:{}".format(0)
    ckpt = torch.load(ckpt_file, map_location=loc)
    # load the model state dict
    model.obj_detect.load_state_dict(ckpt["model"])  # Notice how we load the submodule
    logger.info("loaded checkpoint done.")

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, exp.test_size, results_folder
    )
    logger.info("\n" + summary)

    logger.info('Completed')


if __name__ == "__main__":
    parse_args = make_parser().parse_args()
    tracker_args, _ = load_args_from_config(parse_args.busca_config)
    args = merge_args(tracker_args, parse_args)

    exp = get_exp(args.exp_file)
    exp.merge(args.opts)

    if args.output_dir is not None:
        exp.output_dir = args.output_dir

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    main(exp, args)
