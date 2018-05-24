import sys

sys.path.append('..')
from train_nv import build_model
from hparams import hparams

preset = '../presets/ljspeech_8_bit_nv.json'
with open(preset) as f:
    hparams.parse_json(f.read())
nv_wavenet = build_model(hparams)
nv_wavenet.get_parameters_dict()
# nv_wavenet.prepare_model_condition(ckp_path='',
#                                    mel_path='',
#                                    save_path=''
#                                    )
