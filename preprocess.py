# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <name> <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""
from docopt import docopt
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams


def preprocess(mod, in_dir, out_root, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    sr = hparams.sample_rate
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    # args = docopt(__doc__)
    # name = args["<name>"]
    name = 'ljspeech'
    # in_dir = args["<in_dir>"]
    if name == 'cmu_arctic':
        in_dir = '/home/jinqiangzeng/work/data/speech/cmu_arctic'
        out_dir = './data/cmu_arctic'
    elif name == 'ljspeech':
        in_dir = '/home/jinqiangzeng/work/data/speech/ljspeech/LJSpeech-1.0'
        out_dir = './data/ljspeech_8bits'
    # out_dir = args["<out_dir>"]
    # out_dir = './data/cmu_arctic_mulaw-quantize'
    # num_workers = args["--num_workers"]
    num_workers = cpu_count() - 1
    num_workers = cpu_count() if num_workers is None else int(num_workers)
    preset = '/home/jinqiangzeng/work/pycharm/wavenet_vocoder/presets/ljspeech_8_bit_nv.json'

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    # hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    # Presets
    # if hparams.preset is not None and hparams.preset != "":
    #     preset = hparams.presets[hparams.preset]
    import json

    # hparams.parse_json(json.dumps(preset))
    # print("Override hyper parameters with preset \"{}\": {}".format(
    #     hparams.preset, json.dumps(preset, indent=4)))

    print("Sampling frequency: {}".format(hparams.sample_rate))

    # assert name in ["cmu_arctic", "ljspeech", "librivox", ]
    mod = importlib.import_module(name)
    preprocess(mod, in_dir, out_dir, num_workers)
