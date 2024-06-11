from copy import deepcopy
import math
import os, sys
import random
import traceback
from tqdm import tqdm
now_dir = os.getcwd()
sys.path.append(now_dir)
import ffmpeg
import os
from typing import Generator, List, Union
import numpy as np
import torch
import shutil
import psutil
import signal
import platform
import json
import torch.nn.functional as F
import yaml
import datetime
from subprocess import Popen
from transformers import AutoModelForMaskedLM, AutoTokenizer
from gpt_sovits_python.TTS_infer_pack.setup_logger import logger
from gpt_sovits_python.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from gpt_sovits_python.feature_extractor.cnhubert import CNHubert
from gpt_sovits_python.module.models import SynthesizerTrn
import librosa
from time import time as ttime
from gpt_sovits_python.utils import load_audio
from gpt_sovits_python.module.mel_processing import spectrogram_torch
from gpt_sovits_python.TTS_infer_pack.text_segmentation_method import splits
from gpt_sovits_python.TTS_infer_pack.TextPreprocessor import TextPreprocessor

# configs/tts_infer.yaml
"""
default:
  device: cpu
  is_half: false
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/s2G488k.pth
  
custom:
  device: cuda
  is_half: true
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/s2G488k.pth


"""

def set_seed(seed:int):
    seed = int(seed)
    seed = seed if seed != -1 else random.randrange(1 << 32)
    logger.debug(f"Set seed to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = True
            # 开启后会影响精度
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed
    
class TTS_Config:
    default_configs={
                "device": "cpu",
                "is_half": False,
                "t2s_weights_path": "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "vits_weights_path": "pretrained_models/s2G488k.pth",
                "pretrained_s2D": "pretrained_models/s2D488k.pth",
                "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
                "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
            }
    configs:dict = None
    def __init__(self, configs: Union[dict, str]=None):
        
        # 设置默认配置文件路径
        # configs_base_path:str = "GPT_SoVITS/configs/"
        # os.makedirs(configs_base_path, exist_ok=True)
        # self.configs_path:str = os.path.join(configs_base_path, "tts_infer.yaml")
        
        # if configs in ["", None]:
        #     if not os.path.exists(self.configs_path):
        #         self.save_configs()
        #         logger.debug(f"Create default config file at {self.configs_path}")
        #     configs:dict = {"default": deepcopy(self.default_configs)}
        
        # if isinstance(configs, str):
        #     self.configs_path = configs
        #     configs:dict = self._load_configs(self.configs_path)
            
        assert isinstance(configs, dict)
        default_configs:dict = configs.get("default", None)
        if default_configs is not None:
            self.default_configs = default_configs
            
        self.configs:dict = configs.get("custom", deepcopy(self.default_configs))
        
        
        self.device = self.configs.get("device", torch.device("cpu"))
        self.is_half = self.configs.get("is_half", False)
        self.bert_base_path = self.configs.get("bert_base_path", None)
        self.cnhuhbert_base_path = self.configs.get("cnhuhbert_base_path", None)
        self.pretrained_s2D = self.configs.get("pretrained_s2D", None)
        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)

        if (self.t2s_weights_path in [None, ""]) or (not os.path.exists(self.t2s_weights_path)):
            self.t2s_weights_path = self.default_configs['t2s_weights_path']
            logger.debug(f"fall back to default t2s_weights_path: {self.t2s_weights_path}")
        if (self.vits_weights_path in [None, ""]) or (not os.path.exists(self.vits_weights_path)):
            self.vits_weights_path = self.default_configs['vits_weights_path']
        if (self.bert_base_path in [None, ""]) or (not os.path.exists(self.bert_base_path)):
            self.bert_base_path = self.default_configs['bert_base_path']
            logger.debug(f"fall back to default bert_base_path: {self.bert_base_path}")
        if (self.cnhuhbert_base_path in [None, ""]) or (not os.path.exists(self.cnhuhbert_base_path)):
            self.cnhuhbert_base_path = self.default_configs['cnhuhbert_base_path']
            logger.debug(f"fall back to default cnhuhbert_base_path: {self.cnhuhbert_base_path}")
        if (self.pretrained_s2D in [None, ""]) or (not os.path.exists(self.pretrained_s2D)):
            self.pretrained_s2D = self.default_configs['pretrained_s2D']
            logger.debug(f"fall back to default pretrained_s2D: {self.pretrained_s2D}")
        self.update_configs()
        
        
        self.max_sec = None
        self.hz:int = 50
        self.semantic_frame_rate:str = "25hz"
        self.segment_size:int = 20480
        self.filter_length:int = 2048
        self.sampling_rate:int = 32000
        self.hop_length:int = 640
        self.win_length:int = 2048
        self.n_speakers:int = 300
        
        self.languages:list = ["auto", "en", "zh", "ja",  "all_zh", "all_ja"]
            
    def _load_configs(self, configs_path: str)->dict:
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
    
        return configs

    def save_configs(self, configs_path:str=None)->None:
        configs={
            "default":self.default_configs,
        }
        if self.configs is not None:
            configs["custom"] = self.update_configs()

        # if configs_path is None:
        #     configs_path = self.configs_path
        # with open(configs_path, 'w') as f:
        #     yaml.dump(configs, f)

    def update_configs(self):
        self.config = {
            "device"             : str(self.device),
            "is_half"            : self.is_half,
            "bert_base_path"     : self.bert_base_path,
            "cnhuhbert_base_path": self.cnhuhbert_base_path,
        }
        return self.config

    def __str__(self):
        self.configs = self.update_configs()
        string = "TTS Config".center(100, '-') + '\n'
        for k, v in self.configs.items():
            string += f"{str(k).ljust(20)}: {str(v)}\n"
        string += "-" * 100 + '\n'
        return string
    
    def __repr__(self):
        return self.__str__()

base_vits_model = None
base_t2s_model = None
global_vits_model = None
global_t2s_model = None
global_models = {
    "base_vits_model": None,
    "base_t2s_model": None,
    "global_vits_model": None,
    "global_t2s_model": None
}

class TTS:
    def __init__(self, configs: Union[dict, str, TTS_Config]):
        if isinstance(configs, TTS_Config):
            self.configs = configs
        else:
            self.configs:TTS_Config = TTS_Config(configs)
        self.bert_tokenizer:AutoTokenizer = None
        self.bert_model:AutoModelForMaskedLM = None
        self.cnhuhbert_model:CNHubert = None
        self.last_model_paths = {"vits_weights_path": self.configs.vits_weights_path, "t2s_weights_path": self.configs.t2s_weights_path}
        self.current_model_paths = {"vits_weights_path": self.configs.vits_weights_path, "t2s_weights_path": self.configs.t2s_weights_path}
        self._init_models()
        
        self.text_preprocessor:TextPreprocessor = \
                            TextPreprocessor(self.bert_model, 
                                            self.bert_tokenizer, 
                                            self.configs.device)
        
        
        self.prompt_cache:dict = {
            "ref_audio_path" : None,
            "prompt_semantic": None,
            "refer_spec"     : None,
            "prompt_text"    : None,
            "prompt_lang"    : None,
            "phones"         : None,
            "bert_features"  : None,
            "norm_text"      : None,
        }
        
        
        self.stop_flag:bool = False
        self.precision:torch.dtype = torch.float16 if self.configs.is_half else torch.float32

    def _init_models_in_run_and_train(self, t2s_weights_path, vits_weights_path):
        global global_models
        t0 =ttime()
        self.last_model_paths["vits_weights_path"] = vits_weights_path
        self.last_model_paths["t2s_weights_path"] = t2s_weights_path
        global_models["global_t2s_model"] = self.init_t2s_weights(t2s_weights_path)
        print("---------------->t2stime", ttime() - t0)
        t0 = ttime()
        global_models["global_vits_model"] = self.init_vits_weights(vits_weights_path)
        print("---------------->vitstime", ttime() - t0)

    def _init_models(self,):
        global global_models
        t0 = ttime()
        self.init_bert_weights(self.configs.bert_base_path)
        print("---------------->berttime", ttime() - t0)
        t0 = ttime()
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)
        print("---------------->cnhuhberttime", ttime() - t0)
        t0 = ttime()
        global_models["base_t2s_model"] = self.init_t2s_weights(self.configs.t2s_weights_path)
        print("---------------->t2stime", ttime() - t0)
        t0 = ttime()
        global_models["base_vits_model"] = self.init_vits_weights(self.configs.vits_weights_path)
        print("---------------->vitstime", ttime() - t0)
        # base_vits_model, base_t2s_model = self.enable_half_precision(base_vits_model, base_t2s_model, self.configs.is_half)
        
        
        
    def init_cnhuhbert_weights(self, base_path: str):
        logger.info(f"Loading CNHuBERT weights from {base_path}")
        self.cnhuhbert_model = CNHubert(base_path)
        self.cnhuhbert_model=self.cnhuhbert_model.eval()
        self.cnhuhbert_model = self.cnhuhbert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device)!="cpu":
            self.cnhuhbert_model = self.cnhuhbert_model.half()
        
        
        
    def init_bert_weights(self, base_path: str):
        logger.info(f"Loading BERT weights from {base_path}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        self.bert_model = self.bert_model.eval()
        self.bert_model = self.bert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device)!="cpu":
            self.bert_model = self.bert_model.half()
        
    def init_vits_weights(self, weights_path: str):
        logger.info(f"Loading VITS weights from {weights_path}")
        dict_s2 = torch.load(weights_path, map_location=self.configs.device)
        hps = dict_s2["config"]
        self.configs.filter_length = hps["data"]["filter_length"]
        self.configs.segment_size = hps["train"]["segment_size"]
        self.configs.sampling_rate = hps["data"]["sampling_rate"]       
        self.configs.hop_length = hps["data"]["hop_length"]
        self.configs.win_length = hps["data"]["win_length"]
        self.configs.n_speakers = hps["data"]["n_speakers"]
        self.configs.semantic_frame_rate = "25hz"
        kwargs = hps["model"]
        vits_model = SynthesizerTrn(
            self.configs.filter_length // 2 + 1,
            self.configs.segment_size // self.configs.hop_length,
            n_speakers=self.configs.n_speakers,
            **kwargs
        )
        # if ("pretrained" not in weights_path):
        if hasattr(vits_model, "enc_q"):
            del vits_model.enc_q
            
        vits_model = vits_model.to(self.configs.device)
        vits_model = vits_model.eval()
        vits_model.load_state_dict(dict_s2["weight"], strict=False)
        if self.configs.is_half and str(self.configs.device)!="cpu":
            vits_model = vits_model.half()
        return vits_model

        
    def init_t2s_weights(self, weights_path: str):
        logger.info(f"Loading Text2Semantic weights from {weights_path}")
        self.configs.hz = 50
        dict_s1 = torch.load(weights_path, map_location=self.configs.device)
        config = dict_s1["config"]
        self.configs.max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(self.configs.device)
        t2s_model = t2s_model.eval()
        if self.configs.is_half and str(self.configs.device)!="cpu":
            t2s_model = t2s_model.half()
        return t2s_model
        
    def enable_half_precision(self, vits_model, t2s_model, enable: bool = True):
        '''
            To enable half precision for the TTS model.
            Args:
                enable: bool, whether to enable half precision.
                
        '''
        if str(self.configs.device) == "cpu" and enable:
            logger.info("Half precision is not supported on CPU.")
            return
        
        self.configs.is_half = enable
        self.precision = torch.float16 if enable else torch.float32
        self.configs.save_configs()
        if enable:
            if t2s_model is not None:
                t2s_model = t2s_model.half()
            if vits_model is not None:
                vits_model = vits_model.half()
            if self.bert_model is not None:
                self.bert_model =self.bert_model.half()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.half()
        else:
            if t2s_model is not None:
                t2s_model = t2s_model.float()
            if vits_model is not None:
                vits_model = vits_model.float()
            if self.bert_model is not None:
                self.bert_model = self.bert_model.float()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.float()
        return vits_model, t2s_model
                
    def set_device(self, device: torch.device):
    
        '''
            To set the device for all models.
            Args:
                device: torch.device, the device to use for all models.
        '''
        global global_t2s_model, global_vits_model
        self.configs.device = device
        self.configs.save_configs()
        if global_t2s_model is not None:
            global_t2s_model = global_t2s_model.to(device)
        if global_vits_model is not None:
            global_vits_model = global_vits_model.to(device)
        if self.bert_model is not None:
            self.bert_model = self.bert_model.to(device)
        if self.cnhuhbert_model is not None:
            self.cnhuhbert_model = self.cnhuhbert_model.to(device)
        
    def set_ref_audio(self, ref_audio_path:str):
        '''
            To set the reference audio for the TTS model, 
                including the prompt_semantic and refer_spepc.
            Args:
                ref_audio_path: str, the path of the reference audio.
        '''
        self._set_prompt_semantic(ref_audio_path)
        self._set_ref_spec(ref_audio_path)
        
    def _set_ref_spec(self, ref_audio_path):
        audio = load_audio(ref_audio_path, int(self.configs.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.configs.filter_length,
            self.configs.sampling_rate,
            self.configs.hop_length,
            self.configs.win_length,
            center=False,
        )
        spec = spec.to(self.configs.device)
        if self.configs.is_half:
            spec = spec.half()
        # self.refer_spec = spec
        self.prompt_cache["refer_spec"] = spec
        
        
    def _set_prompt_semantic(self, ref_wav_path:str):
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * 0.3),
            dtype=np.float16 if self.configs.is_half else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise ValueError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.configs.device)
            zero_wav_torch = zero_wav_torch.to(self.configs.device)
            if self.configs.is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            hubert_feature = self.cnhuhbert_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            if self.current_model_paths["vits_weights_path"] == self.configs.vits_weights_path:
                key = "base_vits_model"
            else:
                key = "global_vits_model"
            global global_models
            codes = global_models[key].extract_latent(hubert_feature)
            prompt_semantic = codes[0, 0].to(self.configs.device)
            self.prompt_cache["prompt_semantic"] = prompt_semantic
    
    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length:int=None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype:torch.dtype = seq.dtype
        pad_value = torch.tensor(pad_value, dtype=dtype)
        seq_lengths = [seq.shape[axis] for seq in sequences]
        if max_length is None:
            max_length = max(seq_lengths)
        else:
            max_length = max(seq_lengths) if max_length < max(seq_lengths) else max_length

        padded_sequences = []
        for seq, length in zip(sequences, seq_lengths):
            padding = [0] * axis + [0, max_length - length] + [0] * (ndim - axis - 1)
            padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            padded_sequences.append(padded_seq)
        batch = torch.stack(padded_sequences)
        return batch
    
    def to_batch(self, data:list, 
                 prompt_data:dict=None, 
                 batch_size:int=5, 
                 threshold:float=0.75, 
                 split_bucket:bool=True, 
                 device:torch.device=torch.device("cpu"),
                 precision:torch.dtype=torch.float32,
                 ):
        # 但是这里不能套，反而会负优化
        # with torch.no_grad():
        _data:list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        batch_index_list = []
        if split_bucket:
            index_and_len_list.sort(key=lambda x: x[1])
            index_and_len_list = np.array(index_and_len_list, dtype=np.int64)            
            
            batch_index_list_len = 0
            pos = 0
            while pos <index_and_len_list.shape[0]:
                # batch_index_list.append(index_and_len_list[pos:min(pos+batch_size,len(index_and_len_list))])
                pos_end = min(pos+batch_size,index_and_len_list.shape[0])
                while pos < pos_end:
                    batch=index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score=batch[(pos_end-pos)//2]/(batch.mean()+1e-8)
                    if (score>=threshold) or (pos_end-pos==1):
                        batch_index=index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end=pos_end-1
            
            assert batch_index_list_len == len(data)
            
        else:
            for i in range(len(data)):
                if i%batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)

                
        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            phones_len_list = []
            # bert_features_list = []
            all_phones_list = []
            all_phones_len_list = []
            all_bert_features_list = []
            norm_text_batch = []
            bert_max_len = 0
            phones_max_len = 0
            # 但是这里也不能套，反而会负优化
            # with torch.no_grad():
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = torch.cat([prompt_data["bert_features"], item["bert_features"]], 1)\
                                                .to(dtype=precision, device=device)
                    all_phones = torch.LongTensor(prompt_data["phones"]+item["phones"]).to(device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    # norm_text = prompt_data["norm_text"]+item["norm_text"]
                else:
                    all_bert_features = item["bert_features"]\
                                            .to(dtype=precision, device=device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    all_phones = phones
                    # norm_text = item["norm_text"]

                bert_max_len = max(bert_max_len, all_bert_features.shape[-1])
                phones_max_len = max(phones_max_len, phones.shape[-1])
                
                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])
                
            phones_batch = phones_list
            all_phones_batch = all_phones_list
            all_bert_features_batch = all_bert_features_list
            
            
            max_len = max(bert_max_len, phones_max_len)
            # phones_batch = self.batch_sequences(phones_list, axis=0, pad_value=0, max_length=max_len)
            #### 直接对phones和bert_features进行pad。（padding策略会影响T2S模型生成的结果，但不直接影响复读概率。影响复读概率的主要因素是mask的策略）
            # all_phones_batch = self.batch_sequences(all_phones_list, axis=0, pad_value=0, max_length=max_len)
            # all_bert_features_batch = all_bert_features_list
            # all_bert_features_batch = torch.zeros(len(item_list), 1024, max_len, dtype=precision, device=device)
            # for idx, item in enumerate(all_bert_features_list):
            #     all_bert_features_batch[idx, :, : item.shape[-1]] = item
            
            # #### 先对phones进行embedding、对bert_features进行project，再pad到相同长度，（padding策略会影响T2S模型生成的结果，但不直接影响复读概率。影响复读概率的主要因素是mask的策略）
            # all_phones_list = [global_t2s_model.model.ar_text_embedding(item.to(global_t2s_model.device)) for item in all_phones_list]
            # all_phones_list = [F.pad(item,(0,0,0,max_len-item.shape[0]),value=0) for item in all_phones_list]
            # all_phones_batch = torch.stack(all_phones_list, dim=0)
            
            # all_bert_features_list = [global_t2s_model.model.bert_proj(item.to(global_t2s_model.device).transpose(0, 1)) for item in all_bert_features_list]
            # all_bert_features_list = [F.pad(item,(0,0,0,max_len-item.shape[0]), value=0) for item in all_bert_features_list]
            # all_bert_features_batch = torch.stack(all_bert_features_list, dim=0)
            
            batch = {
                "phones": phones_batch,
                "phones_len": torch.LongTensor(phones_len_list).to(device),
                "all_phones": all_phones_batch,
                "all_phones_len": torch.LongTensor(all_phones_len_list).to(device),
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch,
                "max_len": max_len,
            }
            _data.append(batch)
        
        return _data, batch_index_list
    
    def recovery_order(self, data:list, batch_index_list:list)->list:
        '''
        Recovery the order of the audio according to the batch_index_list.
        
        Args:
            data (List[list(np.ndarray)]): the out of order audio .
            batch_index_list (List[list[int]]): the batch index list.
        
        Returns:
            list (List[np.ndarray]): the data in the original order.
        '''
        length = len(sum(batch_index_list, []))
        _data = [None]*length
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data

    def stop(self,):
        '''
        Stop the inference process.
        '''
        self.stop_flag = True
    
    # 使用装饰器
    @torch.no_grad()
    def run(self, inputs:dict):
        """
        Text to speech inference.
        
        Args:
            inputs (dict): 
                {
                    "text": "",                   # str.(required) text to be synthesized
                    "text_lang: "",               # str.(required) language of the text to be synthesized
                    "ref_audio_path": "",         # str.(required) reference audio path
                    "prompt_text": "",            # str.(optional) prompt text for the reference audio
                    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                    "top_k": 5,                   # int. top k sampling
                    "top_p": 1,                   # float. top p sampling
                    "temperature": 1,             # float. temperature for sampling
                    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
                    "batch_size": 1,              # int. batch size for inference
                    "batch_threshold": 0.75,      # float. threshold for batch splitting.
                    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                    "return_fragment": False,     # bool. step by step return the audio fragment.
                    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                    "seed": -1,                   # int. random seed for reproducibility.
                    "parallel_infer": True,       # bool. whether to use parallel inference.
                    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
                }
        returns:
            tuple[int, np.ndarray]: sampling rate and audio data.
        """
        global global_models
        self.current_model_paths = {"t2s_weights_path": inputs.get("t2s_weights_path"), "vits_weights_path": inputs.get("vits_weights_path")}
        if (self.last_model_paths["t2s_weights_path"] != inputs.get("t2s_weights_path")) & (self.configs.t2s_weights_path != inputs.get("t2s_weights_path")):
            self._init_models_in_run_and_train(inputs.get("t2s_weights_path"), inputs.get("vits_weights_path"))
        if self.configs.t2s_weights_path == inputs.get("t2s_weights_path"):
            t2s_key = "base_t2s_model"
            vits_key = "base_vits_model"
        else:
            t2s_key = "global_t2s_model"
            vits_key = "global_vits_model"
        ########## variables initialization ###########
        self.stop_flag:bool = False
        text:str = inputs.get("text", "")
        text_lang:str = inputs.get("text_lang", "")
        ref_audio_path:str = inputs.get("ref_audio_path", "")
        prompt_text:str = inputs.get("prompt_text", "")
        prompt_lang:str = inputs.get("prompt_lang", "")
        top_k:int = inputs.get("top_k", 5)
        top_p:float = inputs.get("top_p", 1)
        temperature:float = inputs.get("temperature", 1)
        text_split_method:str = inputs.get("text_split_method", "cut0")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        actual_seed = set_seed(seed)

        parallel_infer = inputs.get("parallel_infer", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)

        if parallel_infer:
            logger.debug("Parallel inference mode is enabled")
            global_models[t2s_key].model.infer_panel = global_models[t2s_key].model.infer_panel_batch_infer_with_flash_attn
        else:
            logger.debug("Parallel inference mode is turned off")
            global_models[t2s_key].model.infer_panel = global_models[t2s_key].model.infer_panel_0307

        if return_fragment:
            # split_bucket = False
            logger.debug("Segmented return mode is enabled")
            if split_bucket:
                split_bucket = False
                logger.debug("The segmented return mode does not support split processing.")

        if split_bucket:
            logger.debug("Bucket processing mode is turned off")

        if fragment_interval<0.01:
            fragment_interval = 0.01
            logger.info("The segment interval is too small and has been automatically set to 0.01")

        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True

        assert text_lang in self.configs.languages
        if not no_prompt_text:
            assert prompt_lang in self.configs.languages

        if ref_audio_path in [None, ""] and \
            ((self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spec"] is None)):
            raise ValueError("ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()")

        ###### setting reference audio and prompt text preprocessing ########
        t0 = ttime()
        if (ref_audio_path is not None) and (ref_audio_path != self.prompt_cache["ref_audio_path"]):
            self.set_ref_audio(ref_audio_path)

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_lang != "en" else "."
            logger.debug(f"Actual input reference text: {prompt_text}")
            if self.prompt_cache["prompt_text"] != prompt_text:
                self.prompt_cache["prompt_text"] = prompt_text
                self.prompt_cache["prompt_lang"] = prompt_lang
                phones, bert_features, norm_text = \
                    self.text_preprocessor.segment_and_extract_feature_for_text(
                                                                        prompt_text, 
                                                                        prompt_lang)
                self.prompt_cache["phones"] = phones
                self.prompt_cache["bert_features"] = bert_features
                self.prompt_cache["norm_text"] = norm_text

        ###### text preprocessing ########
        t1 = ttime()
        data:list = None
        if not return_fragment:
            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method)
            if len(data) == 0:
                yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                            dtype=np.int16)
                return

            batch_index_list:list = None
            data, batch_index_list = self.to_batch(data, 
                                prompt_data=self.prompt_cache if not no_prompt_text else None, 
                                batch_size=batch_size, 
                                threshold=batch_threshold,
                                split_bucket=split_bucket,
                                device=self.configs.device,
                                precision=self.precision
                                )
        else:
            logger.debug("############ Split text ############")
            texts = self.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
            data = []
            for i in range(len(texts)):
                if i%batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])
            
            def make_batch(batch_texts):
                batch_data = []
                logger.debug("############ Extract text Bert features ############")
                for text in tqdm(batch_texts):
                    phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(text, text_lang)
                    if phones is None:
                        continue
                    res={
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    batch_data.append(res)
                if len(batch_data) == 0:
                    return None
                batch, _ = self.to_batch(batch_data, 
                            prompt_data=self.prompt_cache if not no_prompt_text else None, 
                            batch_size=batch_size, 
                            threshold=batch_threshold,
                            split_bucket=False,
                            device=self.configs.device,
                            precision=self.precision
                            )
                return batch[0]


        t2 = ttime()
        try:
            logger.debug("############ inference ############")
            ###### inference ######
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            for item in data:
                t3 = ttime()
                if return_fragment:
                    item = make_batch(item)
                    if item is None:
                        continue

                batch_phones:List[torch.LongTensor] = item["phones"]
                # batch_phones:torch.LongTensor = item["phones"]
                batch_phones_len:torch.LongTensor = item["phones_len"]
                all_phoneme_ids:torch.LongTensor = item["all_phones"]
                all_phoneme_lens:torch.LongTensor  = item["all_phones_len"]
                all_bert_features:torch.LongTensor = item["all_bert_features"]
                norm_text:str = item["norm_text"]
                max_len = item["max_len"]
                
                logger.debug(f"Front-end processed text (each sentence): {norm_text}")
                if no_prompt_text :
                    prompt = None
                else:
                    prompt = self.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.configs.device)

                pred_semantic_list, idx_list = global_models[t2s_key].model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,
                    all_bert_features,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.configs.hz * self.configs.max_sec,
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
                t4 = ttime()
                t_34 += t4 - t3

                refer_audio_spec:torch.Tensor = self.prompt_cache["refer_spec"]\
                                                    .to(dtype=self.precision, device=self.configs.device)

                batch_audio_fragment = []

                # 这里要记得加 torch.no_grad() 不然速度慢一大截
                # with torch.no_grad():
            
                # ## vits并行推理 method 1
                # pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                # pred_semantic_len = torch.LongTensor([item.shape[0] for item in pred_semantic_list]).to(self.configs.device)
                # pred_semantic = self.batch_sequences(pred_semantic_list, axis=0, pad_value=0).unsqueeze(0)
                # max_len = 0
                # for i in range(0, len(batch_phones)):
                #     max_len = max(max_len, batch_phones[i].shape[-1])
                # batch_phones = self.batch_sequences(batch_phones, axis=0, pad_value=0, max_length=max_len)
                # batch_phones = batch_phones.to(self.configs.device)
                # batch_audio_fragment = (global_vits_model.batched_decode(
                #         pred_semantic, pred_semantic_len, batch_phones, batch_phones_len,refer_audio_spec
                #     ))

                # ## vits并行推理 method 2
                pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                upsample_rate = math.prod(global_models[vits_key].upsample_rates)
                audio_frag_idx = [pred_semantic_list[i].shape[0]*2*upsample_rate for i in range(0, len(pred_semantic_list))]
                audio_frag_end_idx = [ sum(audio_frag_idx[:i+1]) for i in range(0, len(audio_frag_idx))]
                all_pred_semantic = torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.configs.device)
                _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(self.configs.device)
                _batch_audio_fragment = (global_models[vits_key].decode(
                        all_pred_semantic, _batch_phones, refer_audio_spec
                    ).detach()[0, 0, :])
                audio_frag_end_idx.insert(0, 0)
                batch_audio_fragment= [_batch_audio_fragment[audio_frag_end_idx[i-1]:audio_frag_end_idx[i]] for i in range(1, len(audio_frag_end_idx))]

                # ## vits串行推理
                # for i, idx in enumerate(idx_list):
                #     phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                #     _pred_semantic = (pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0))   # .unsqueeze(0)#mq要多unsqueeze一次
                #     audio_fragment =(global_vits_model.decode(
                #             _pred_semantic, phones, refer_audio_spec
                #         ).detach()[0, 0, :])
                #     batch_audio_fragment.append(
                #         audio_fragment
                #     )  ###试试重建不带上prompt部分

                t5 = ttime()
                t_45 += t5 - t4
                if return_fragment:
                    logger.debug("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t4 - t3, t5 - t4))
                    yield self.audio_postprocess([batch_audio_fragment], 
                                                    self.configs.sampling_rate, 
                                                    None, 
                                                    speed_factor, 
                                                    False,
                                                    fragment_interval
                                                    )
                else:
                    audio.append(batch_audio_fragment)

                if self.stop_flag:
                    yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                            dtype=np.int16)
                    return

            if not return_fragment:
                logger.debug("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
                yield self.audio_postprocess(audio, 
                                                self.configs.sampling_rate, 
                                                batch_index_list, 
                                                speed_factor, 
                                                split_bucket,
                                                fragment_interval
                                                )

        except Exception as e:
            traceback.print_exc()
            # 必须返回一个空音频, 否则会导致显存不释放。
            yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                            dtype=np.int16)
            # 重置模型, 否则会导致显存释放不完全。
            del global_models[t2s_key]
            del global_models[vits_key]
            global_models[t2s_key] = None
            global_models[vits_key] = None
            if vits_key == "base_vits_model":
                global_models[vits_key] = self.init_t2s_weights(self.configs.t2s_weights_path)
                global_models[t2s_key] = self.init_vits_weights(self.configs.vits_weights_path)
            raise e
        finally:
            self.empty_cache()
    
    def empty_cache(self):
        try:    
            if "cuda" in str(self.configs.device):
                torch.cuda.empty_cache()
            elif str(self.configs.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass 
        
    def audio_postprocess(self, 
                          audio:List[torch.Tensor], 
                          sr:int, 
                          batch_index_list:list=None, 
                          speed_factor:float=1.0, 
                          split_bucket:bool=True,
                          fragment_interval:float=0.3
                          )->tuple[int, np.ndarray]:
        zero_wav = torch.zeros(
                        int(self.configs.sampling_rate * fragment_interval),
                        dtype=self.precision,
                        device=self.configs.device
                    )
        
        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                max_audio=torch.abs(audio_fragment).max()#简单防止16bit爆音
                if max_audio>1: audio_fragment/=max_audio
                audio_fragment:torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0)
                audio[i][j] = audio_fragment.cpu().numpy()
            
        
        if split_bucket:
            audio = self.recovery_order(audio, batch_index_list)
        else:
            # audio = [item for batch in audio for item in batch]
            audio = sum(audio, [])
            
            
        audio = np.concatenate(audio, 0)
        audio = (audio * 32768).astype(np.int16) 
        
        try:
            if speed_factor != 1.0:
                audio = speed_change(audio, speed=speed_factor, sr=int(sr))
        except Exception as e:
            logger.error(f"Failed to change speed of audio: \n{e}")
        
        return sr, audio
            
    def train(self, inputs:dict):
        self._init_models_in_run_and_train(inputs.get("t2s_weights_path"), inputs.get("vits_weights_path"))
        python_exec = sys.executable or "python"
        tmp = os.path.join(now_dir, "TEMP")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(tmp, exist_ok=True)
        os.makedirs(inputs.get("sovits_weight_root"), exist_ok=True)
        os.makedirs(inputs.get("gpt_weight_root"), exist_ok=True)
        os.environ["TEMP"] = tmp
        if(os.path.exists(tmp)):
            for name in os.listdir(tmp):
                if(name=="jieba.cache"):continue
                path="%s/%s"%(tmp,name)
                delete=os.remove if os.path.isfile(path) else shutil.rmtree
                try:
                    delete(path)
                except Exception as e:
                    print(str(e))
                    pass
        def clean_path(path_str):
            if platform.system() == 'Windows':
                path_str = path_str.replace('/', '\\')
            return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        
        # inp_text 文本标注路径
        def open1abc():
            ps1abc=[]
            inp_text = clean_path(inputs.get("inp_text"))
            inp_wav_dir = clean_path(inputs.get("inp_wav_dir"))
            if (ps1abc == []):
                opt_dir="%s/%s"%(inputs.get("exp_root"), inputs.get("exp_name"))
                os.makedirs(opt_dir, exist_ok=True)
                try:
                    #############################1a
                    path_text="%s/2-name2text.txt" % opt_dir
                    if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and len(open(path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
                        config={
                            "inp_text":inp_text,
                            "inp_wav_dir":inp_wav_dir,
                            "exp_name":inputs.get("exp_name"),
                            "opt_dir":opt_dir,
                            "bert_pretrained_dir":self.configs.bert_base_path,
                            "is_half": str(inputs.get("is_true", True))
                        }
                        gpu_names=inputs.get("gpu_numbers1a").split("-")
                        all_parts=len(gpu_names)
                        
                        for i_part in range(all_parts):
                            config.update(
                                {
                                    "i_part": str(i_part),
                                    "all_parts": str(all_parts),
                                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                                }
                            )
                            os.environ.update(config)
                            run_file = os.path.join(current_dir, "../prepare_datasets/1-get-text.py")
                            cmd = f"{python_exec} {run_file}"
                            print(cmd)
                            p = Popen(cmd, shell=True)
                            ps1abc.append(p)
                        # yield "进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                        print("--------------------> 进度：1a-ing")
                        for p in ps1abc:p.wait()

                        opt = []
                        for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                            with open(txt_path, "r",encoding="utf8") as f:
                                opt += f.read().strip("\n").split("\n")
                            os.remove(txt_path)
                        with open(path_text, "w",encoding="utf8") as f:
                            f.write("\n".join(opt) + "\n")

                    # yield "进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                    print("--------------------> 进度：1a-done")
                    ps1abc=[]
                    #############################1b
                    config={
                        "inp_text":inp_text,
                        "inp_wav_dir":inp_wav_dir,
                        "exp_name":inputs.get("exp_name"),
                        "opt_dir":opt_dir,
                        "cnhubert_base_dir":self.configs.cnhuhbert_base_path,
                    }
                    gpu_names=inputs.get("gpu_numbers1Ba").split("-")
                    all_parts=len(gpu_names)
                    for i_part in range(all_parts):
                        config.update(
                            {
                                "i_part": str(i_part),
                                "all_parts": str(all_parts),
                                "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                            }
                        )
                        os.environ.update(config)
                        run_file = os.path.join(current_dir, "../prepare_datasets/2-get-hubert-wav32k.py")
                        cmd = f"{python_exec} {run_file}"
                        print(cmd)
                        p = Popen(cmd, shell=True)
                        ps1abc.append(p)
                    # yield "进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                    print("--------------------> 进度：1a-done, 1b-ing")
                    for p in ps1abc:p.wait()
                    # yield "进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                    print("--------------------> 进度：1a1b-done")
                    ps1abc=[]
                    #############################1c
                    path_semantic = "%s/6-name2semantic.tsv" % opt_dir
                    if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<31)):
                        config={
                            "inp_text":inp_text,
                            "exp_name":inputs.get("exp_name"),
                            "opt_dir":opt_dir,
                            "pretrained_s2G":self.configs.vits_weights_path,
                            "s2config_path":f"{current_dir}/../configs/s2.json",
                        }
                        gpu_names=inputs.get("gpu_numbers1c").split("-")
                        all_parts=len(gpu_names)
                        for i_part in range(all_parts):
                            config.update(
                                {
                                    "i_part": str(i_part),
                                    "all_parts": str(all_parts),
                                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                                }
                            )
                            os.environ.update(config)
                            run_file = os.path.join(current_dir, "../prepare_datasets/3-get-semantic.py")
                            cmd = f"{python_exec} {run_file}"
                            print(cmd)
                            p = Popen(cmd, shell=True)
                            ps1abc.append(p)
                        # yield "进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                        print("--------------------> 进度：1a1b-done, 1cing")
                        for p in ps1abc:p.wait()

                        opt = ["item_name\tsemantic_audio"]
                        for i_part in range(all_parts):
                            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                            with open(semantic_path, "r",encoding="utf8") as f:
                                opt += f.read().strip("\n").split("\n")
                            os.remove(semantic_path)
                        with open(path_semantic, "w",encoding="utf8") as f:
                            f.write("\n".join(opt) + "\n")
                        # yield "进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                        print("--------------------> 进度：all-done")
                    ps1abc = []
                    # yield "一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
                except:
                    traceback.print_exc()
                    close1abc(ps1abc != [])
                    # yield "一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
                    print("--------------------> 进度：一键三连中途报错")
            else:
                print("--------------------> 已有正在进行的一键三连任务，需先终止才能开启下一次任务")
                # yield "已有正在进行的一键三连任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        def close1abc(ps1abc):
            for p1abc in ps1abc:
                try:
                    kill_process(p1abc.pid)
                except:
                    traceback.print_exc()
            print("--------------------> 已终止所有一键三连进程")

        def kill_process(pid):
            system=platform.system()
            if(system=="Windows"):
                cmd = "taskkill /t /f /pid %s" % pid
                os.system(cmd)
            else:
                kill_proc_tree(pid)

        def kill_proc_tree(pid, including_parent=True):  
            try:
                parent = psutil.Process(pid)
            except psutil.NoSuchProcess:
                # Process already terminated
                return

            children = parent.children(recursive=True)
            for child in children:
                try:
                    os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
                except OSError:
                    pass
            if including_parent:
                try:
                    os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
                except OSError:
                    pass
        
        def open1Ba():
            p_train_SoVITS = None
            if(p_train_SoVITS==None):  
                with open(f"{current_dir}/../configs/s2.json")as f:
                    data=f.read()
                    data=json.loads(data)
                s2_dir="%s/%s"%(inputs.get("exp_root"),inputs.get("exp_name"))
                os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
                if(inputs.get("is_true", True)==False):
                    data["train"]["fp16_run"]=False
                    batch_size=max(1,inputs.get("batch_size1Ba")//2)
                else:
                    batch_size = inputs.get("batch_size1Ba")
                data["train"]["batch_size"]=batch_size
                data["train"]["epochs"]=inputs.get("total_epoch1Ba")
                data["train"]["text_low_lr_rate"]=inputs.get("text_low_lr_rate1Ba")
                data["train"]["pretrained_s2G"]=self.configs.vits_weights_path
                data["train"]["pretrained_s2D"]=self.configs.pretrained_s2D
                data["train"]["if_save_latest"]=inputs.get("if_save_latest")
                data["train"]["if_save_every_weights"]=inputs.get("if_save_every_weights")
                data["train"]["save_every_epoch"]=inputs.get("save_every_epoch1Ba")
                data["train"]["gpu_numbers"]=inputs.get("gpu_numbers1Ba")
                data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
                data["save_weight_dir"]=inputs.get("sovits_weight_root")
                data["name"]=inputs.get("exp_name")
                tmp_config_path="%s/tmp_s2.json"%tmp
                with open(tmp_config_path,"w")as f:f.write(json.dumps(data))
                cmd = '"%s" %s/s2_train.py --config "%s"'%(python_exec,current_dir,tmp_config_path)
                # yield "SoVITS训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
                print("--------------------> SoVITS训练开始")
                print(cmd)
                p_train_SoVITS = Popen(cmd, shell=True)
                p_train_SoVITS.wait()
                p_train_SoVITS=None
                # yield "SoVITS训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
                print("--------------------> SoVITS训练完成")       
            else:
                # yield "已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}
                print("--------------------> 已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务")

        def open1Bb():
            p_train_GPT=None
            if(p_train_GPT==None):
                with open(f"{current_dir}/../configs/s1longer.yaml")as f:
                    data=f.read()
                    data=yaml.load(data, Loader=yaml.FullLoader)
                s1_dir="%s/%s"%(inputs.get("exp_root"),inputs.get("exp_name"))
                os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
                if(inputs.get("is_half")==False):
                    data["train"]["precision"]="32"
                    batch_size = max(1, inputs.get("batch_size1Bb") // 2)
                else:
                    batch_size = inputs.get("batch_size1Bb")
                data["train"]["batch_size"]=batch_size
                data["train"]["epochs"]=inputs.get("total_epoch1Bb")
                data["pretrained_s1"]=self.configs.t2s_weights_path
                data["train"]["save_every_n_epoch"]=inputs.get("save_every_epoch")
                data["train"]["if_save_every_weights"]=inputs.get("if_save_every_weights")
                data["train"]["if_save_latest"]=inputs.get("if_save_latest")
                data["train"]["if_dpo"]=inputs.get("if_dpo")
                data["train"]["half_weights_save_dir"]=inputs.get("gpt_weight_root")
                data["train"]["exp_name"]=inputs.get("exp_name")
                data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
                data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
                data["output_dir"]="%s/logs_s1"%s1_dir

                os.environ["_CUDA_VISIBLE_DEVICES"]=inputs.get("gpu_numbers1Bb").replace("-",",")
                os.environ["hz"]="25hz"
                tmp_config_path="%s/tmp_s1.yaml"%tmp
                with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
                # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
                cmd = '"%s" %s/s1_train.py --config_file "%s" '%(python_exec,current_dir,tmp_config_path)
                # yield "GPT训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
                print("--------------------> GPT训练开始：")
                print(cmd)
                p_train_GPT = Popen(cmd, shell=True)
                p_train_GPT.wait()
                p_train_GPT=None
                # yield "GPT训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
                print("--------------------> GPT训练完成")
            else:
                # yield "已有正在进行的GPT训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}
                print("--------------------> 已有正在进行的GPT训练任务，需先终止才能开启下一次任务")

        open1abc()
        open1Ba()
        open1Bb()

        
       
def speed_change(input_audio:np.ndarray, speed:float, sr:int):
    # 将 NumPy 数组转换为原始 PCM 流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio