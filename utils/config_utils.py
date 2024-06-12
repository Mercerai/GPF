import argparse
import os
import torch
import yaml
import addict
import copy
import json

class ForceKeyErrorDict(addict.Dict):
    def __missing__(self, name):
        raise KeyError(name)

def save_config(datadict: ForceKeyErrorDict, path: str):
    datadict = copy.deepcopy(datadict)
    datadict.training.ckpt_file = None
    datadict.training.pop("exp_dir")
    with open(path, "w", encoding="utf8") as outfile:
        yaml.dump(datadict.to_dict(), outfile, default_flow_style=False)


def update_config(config, unknown):
    # update config given args
    for idx, arg in enumerate(unknown):
        if arg.startswith("--"):
            if (":") in arg:
                k1, k2 = arg.replace("--", "").split(":")
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx + 1].lower() == "true"
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx + 1])
                    else:
                        v = unknown[idx + 1]
                print(f"Changing {k1}:{k2} ---- {config[k1][k2]} to {v}")
                config[k1][k2] = v
            else:
                k = arg.replace("--", "")
                v = unknown[idx + 1]
                argtype = type(config[k])
                print(f"Changing {k} ---- {config[k]} to {v}")
                config[k] = v

    return config

def create_args_parser():
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument(
        "--resume_dir", type=str, default=None, help="Directory of experiment to load."
    )
    return parser




def read_yaml(path):
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def read_json(path):
    with open(path, "r") as f:
        config_dict = json.loads(f.read())
    return config_dict

def load_yaml(path, default_path=None):

    with open(path, encoding="utf8") as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)

    if default_path is not None and path != default_path:
        with open(default_path, encoding="utf8") as default_yaml_file:
            default_config_dict = yaml.load(default_yaml_file, Loader=yaml.FullLoader)
            main_config = ForceKeyErrorDict(**default_config_dict)

        # def overwrite(output_config, update_with):
        #     for k, v in update_with.items():
        #         if not isinstance(v, dict):
        #             output_config[k] = v
        #         else:
        #             overwrite(output_config[k], v)
        # overwrite(main_config, config)

        # simpler solution
        main_config.update(config)
        config = main_config

    return config

def load_config(args, unknown, base_config_path=None):
    """overwrite seq
    command line param --over--> args.config --over--> default config yaml
    """
    assert (args.config is not None) != (
        args.resume_dir is not None
    ), "you must specify ONLY one in 'config' or 'resume_dir' "

    # NOTE: '--local_rank=xx' is automatically given by torch.distributed.launch (if used)
    #       BUT: pytorch suggest to use os.environ['LOCAL_RANK'] instead, and --local_rank=xxx will be deprecated in the future.
    #            so we are not using --local_rank at all.
    found_k = None
    for item in unknown:
        if "local_rank" in item:
            found_k = item
            break
    if found_k is not None:
        unknown.remove(found_k)

    print("=> Parse extra configs: ", unknown)
    if args.resume_dir is not None:
        assert (
            args.config is None
        ), "given --config will not be used when given --resume_dir"
        assert (
            "--expname" not in unknown
        ), "given --expname with --resume_dir will lead to unexpected behavior."
        # ---------------
        # if loading from a dir, do not use base.yaml as the default;
        # ---------------
        config_path = os.path.join(args.resume_dir, "config.yaml")
        config = load_yaml(config_path, default_path=None)

        # use configs given by command line to further overwrite current config
        config = update_config(config, unknown)

        # use the loading directory as the experiment path
        config.training.exp_dir = args.resume_dir
        print("=> Loading previous experiments in: {}".format(config.training.exp_dir))
    else:
        # ---------------
        # if loading from a config file
        # use base.yaml as default
        # ---------------
        config = load_yaml(args.config, default_path=base_config_path)

        # use configs given by command line to further overwrite current config
        config = update_config(config, unknown)

        # use the expname and log_root_dir to get the experiement directory
        if "exp_dir" not in config.training:
            config.training.exp_dir = os.path.join(
                config.training.log_root_dir, config.expname
            )

    # add other configs in args to config
    other_dict = vars(args)
    other_dict.pop("config")
    other_dict.pop("resume_dir")
    config.update(other_dict)

    if hasattr(args, "ddp") and args.ddp:
        if config.device_ids != -1:
            print("=> Ignoring device_ids configs when using DDP. Auto set to -1.")
            config.device_ids = -1
    else:
        args.ddp = False
        # # device_ids: -1 will be parsed as using all available cuda device
        # # device_ids: [] will be parsed as using all available cuda device
        if (type(config.device_ids) == int and config.device_ids == -1) or (
            type(config.device_ids) == list and len(config.device_ids) == 0
        ):
            config.device_ids = list(range(torch.cuda.device_count()))
        # # e.g. device_ids: 0 will be parsed as device_ids [0]
        elif isinstance(config.device_ids, int):
            config.device_ids = [config.device_ids]
        # # e.g. device_ids: 0,1 will be parsed as device_ids [0,1]
        elif isinstance(config.device_ids, str):
            config.device_ids = [int(m) for m in config.device_ids.split(",")]
        print("=> Use cuda devices: {}".format(config.device_ids))

    return config