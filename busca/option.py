import types
import copy
import yaml


def load_args_from_config(config_file):
    # We load the config file
    with open(config_file, "r") as stream:
        args = yaml.safe_load(stream)

        tracker_args = types.SimpleNamespace(**args['tracker'])
        trainer_args = types.SimpleNamespace(**args['trainer'])
        transformer_args = types.SimpleNamespace(**args['transformer'])
        dataset_args = types.SimpleNamespace(**args['dataset'])

        tracker_args.transformer = transformer_args
        trainer_args.transformer = transformer_args
        trainer_args.dataset = dataset_args
    
    return tracker_args, trainer_args


def merge_args(base_args, new_args, verbose=True):
    base_args = copy.deepcopy(base_args)
    for key, value in new_args.__dict__.items():
        if key in base_args.__dict__ and value is not None:
            if verbose:
                if key in base_args.__dict__:
                    print('Overriding {} from {} to {}'.format(key, base_args.__dict__[key], value), flush=True)
                else:
                    print('Setting {} to {}'.format(key, value), flush=True)
            setattr(base_args, key, value)
        
        elif key not in base_args.__dict__: 
            setattr(base_args, key, value)
            if verbose:
                print('Setting {} to {}'.format(key, value), flush=True)

    return base_args
