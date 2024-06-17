import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks
import os
import time
from gpt_sovits_python import TTS, TTS_Config
from pydub import AudioSegment
from pathlib import Path
p = Path(os.path.abspath(__file__))
roleplay_dir = p.parent.parent.parent.parent.parent.joinpath("roleplay")
# print(roleplay_dir)
soviets_configs = {
    "default": {
    "t2s_weights_path": f"{roleplay_dir}/mymodels/gptsovits/base/base_gpt.ckpt",
    "vits_weights_path": f"{roleplay_dir}/mymodels/gptsovits/base/base_sovits.pth",
    "device": "cuda", # ["cpu", "cuda"]
    "is_half": True, # Set 'False' if you will use cpu
    "pretrained_s2D": f"{roleplay_dir}/mymodels/gptsovits/base/s2D488k.pth",
    "cnhuhbert_base_path": f"{roleplay_dir}/mymodels/gptsovits/base/chinese-hubert-base",
    "bert_base_path": f"{roleplay_dir}/mymodels/gptsovits/base/chinese-roberta-wwm-ext-large"
    }
}
tts_config = TTS_Config(soviets_configs)
tts_pipeline = TTS(tts_config)

def tts_inf(ref_audio, ref_text, ref_lang, inf_text, inf_lang, cut_way):
        print("=================>", cut_way)
        cut_dic = {"不切分": "cut0", "凑4句切分": "cut1", "凑50字切分": "cut2", "按中文句号切分": "cut3", "按英文句号切分": "cut4", "按标点符号切分": "cut5"}
        params = {
            "t2s_weights_path": soviets_configs["default"]["t2s_weights_path"],
            "vits_weights_path": soviets_configs["default"]["vits_weights_path"],
            "text": inf_text,
            "text_lang": inf_lang,
            "ref_audio_path": ref_audio,
            "prompt_text": ref_text,
            "prompt_lang": ref_lang,
            "top_k": 5,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": cut_dic[cut_way],
            "batch_size": 1,              # int. batch size for inference
            "batch_threshold": 0.75,      # float. threshold for batch splitting.
            "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
            "speed_factor": 1.0,          # float. control the speed of the synthesized audio.
            "fragment_interval": 0.3,     # float. to control the interval of the audio fragment.
            "seed": -1,                   # int. random seed for reproducibility.
            "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
            "streaming_mode": False,      # bool. whether to return a streaming response.
            "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
            "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
        }
        st0 = time.time()
        tts_generator = tts_pipeline.run(params)
        print("===============>get generator", time.time() - st0)
        st = time.time()
        sr, audio_data = next(tts_generator)
        print(type(audio_data))
        return (sr, audio_data)



class GptSovitsApi(scripts.Script):  
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "GptSovits"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):  
        with gr.Group():
            with gr.Row():
                ref_audio = gr.Audio(label="参考音频(3-9s)", sources=["upload", "microphone"], type="filepath")
                ref_text = gr.Textbox(label="参考音频对应的文字(可不填)")
                ref_lang = gr.Dropdown(label="参考音频对应的语言", choices=["zh", "en", "ja"], value="en") 
            
            with gr.Row():
                inf_text = gr.Textbox(label="推理的文字", lines=3)  
                inf_lang = gr.Dropdown(label="推理文字对应的语言", choices=["zh", "en", "ja"], value="en")  
                cut_way = gr.Dropdown(
                    label="切分方式",
                    choices=["不切分", "凑4句切分", "凑50字切分", "按中文句号切分", "按英文句号切分", "按标点符号切分"],
                    value="凑50字切分"
                )
            
                inf_res = gr.Audio(label="推理结果", interactive=False)  
            
            run_button = gr.Button("开始生成")  
            
         
            gr.Markdown("""
            ## 使用说明
            1. 上传一段 3-9 秒的参考音频，或使用麦克风录制。
            2. （可选）填写参考音频对应的文字和语言。
            3. 在推理文字框中输入您想要转换的文字。
            4. 选择推理文字的语言和切分方式。
            5. 点击"开始生成"按钮。
            6. 等待片刻，生成的音频将显示在下方。
            """)

            run_button.click(
                fn=tts_inf, 
                inputs=[ref_audio, ref_text, ref_lang, inf_text, inf_lang, cut_way],
                outputs=[inf_res]
            )

        
        return [ref_audio, ref_text, ref_lang, inf_text, inf_lang, cut_way, inf_res]


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as gptsovits_block:
        GptSovitsApi().ui(is_img2img=False)  

    return [(gptsovits_block, "GptSovits", "gptsovits_interface")]

script_callbacks.on_ui_tabs(on_ui_tabs)   
    