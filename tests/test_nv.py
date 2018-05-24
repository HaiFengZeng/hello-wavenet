import sys

sys.path.append('..')
from train_nv import build_model
from hparams import hparams

preset = '/home/tesla/work/pycharm/hello-wavenet/presets/ljspeech_8_bit_nv.json'
with open(preset) as f:
    hparams.parse_json(f.read())
nv_wavenet = build_model(hparams)
nv_wavenet.prepare_model_condition(ckp_path='',
                                   mel_path='',
                                   save_path=''
                                   )
