import os
import json
import base64
import magic
import requests
import time
import threading
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import dataclasses
from typing import Literal
from accelerate import Accelerator
from transformers import HfArgumentParser
import itertools
import io
import torch
import megfile
from bs4 import BeautifulSoup
import sys
import random
import markdown
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== 全局锁 ======
last_save_lock = threading.Lock()
last_save_time = [None]

@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 512
    height: int = 512
    ref_size: int = -1
    num_steps: int = 25
    guidance: float = 4
    seed: int = 3407
    save_path: str = "output/inference"
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 512
    data_resolution: int = 512
    pe: Literal['d', 'h', 'w', 'o'] = 'd'

def download_with_requests(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                timeout=30,
                headers={'User-Agent': 'Mozilla/5.0'},
                stream=True
            )
            response.raise_for_status()

            buffer = BytesIO()
            for chunk in response.iter_content(8192):
                if chunk:
                    buffer.write(chunk)

            buffer.seek(0)
            Image.open(buffer).verify()
            return Image.open(buffer)

        except Exception as e:
            print(f"❌ 下载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    return None

def extract_md_images(text):
    try:
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        return [img['src'] for img in soup.find_all('img')]
    except Exception as e:
        print(f"❌ 提取图片链接失败: {str(e)}")
        return []

sys.path.append("/data/Code/")
sys.path.append("/data/Code/models/")

# Configuration
base_url = 'https://models-proxy.stepfun-inc.com/v1'
appkey = "YOUR_API_KEY"
headers = {
    'Authorization': f"Bearer {appkey}",
    "Content-Type": "application/json"
}

def convert_image_to_base64(image_input):
    # try:
    if isinstance(image_input, Image.Image):
        img_byte_arr = io.BytesIO()
        image_input.save(img_byte_arr, format='PNG')
        png_binary_data = img_byte_arr.getvalue()
    elif isinstance(image_input, (bytes, bytearray)):
        png_binary_data = image_input
    else:
        raise ValueError('data type not support')

    mime_type = magic.from_buffer(png_binary_data, mime=True)
    base64_encoded_data = base64.b64encode(png_binary_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"
    # except Exception as e:
    #     print(f"❌ Base64转换失败: {str(e)}")
    #     return None

def make_api_request(data, max_retries=3):
    for attempt in range(max_retries):
        # try:
        time.sleep(random.randint(1, 5))
        response = requests.post(
            base_url + '/chat/completions',
            headers=headers,
            json=data,
            timeout=240
        )
        response.raise_for_status()

        res = ""
        for line in response.iter_lines(decode_unicode=False, chunk_size=10):
            res += line.decode('utf-8')

        res = json.loads(res)

            # 处理响应
        image_link_list = extract_md_images(res['choices'][0]['message']['content'])
        if not image_link_list:
            print(f"❌ 未找到图片链接")

        output_image = download_with_requests(image_link_list[0])
        if not output_image:
            print(f"❌ 未找到图片链接")

        # except Exception as e:
        #     print(f"  - 错误类型: {type(e).__name__}")
        #     print(f"❌ API请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
        #     response = response.json()
        #     print(response)
        #     if attempt < max_retries - 1:
        #         time.sleep(2)
        #     continue
    return None

def process_single_item(item, history=None, save_history=False):
    system_prompt = "You are a professional story generator. I will provide you with several character images and a prompt for each shot. For the second shot and beyond, the history of previous shots will also be available."
    # try:
    prompt = item['prompt']
    ref_imgs = item['ref_imgs']
    save_path = item['save_path']
    key_words_dict = item.get('key_words_dict', {})

    prompt = f'(Output image with width:height=16:9) {prompt}'
    
    print(f"📄 处理: {save_path} | 指令: {prompt}")

    print(f'--------------------item:{item}')

    # Prepare request data
    data = {
        'model': 'gpt-4o-all',
        'stream': False,
        "max_tokens": 4096,
        "messages": []
    }
    if history is not None:
        data["messages"].extend(history)

        # Build user message with images and prompt
        user_content = []
        # Add the main prompt
        user_content.append({
            'type': 'text',
            'text': prompt
        })
        data["messages"].append({
            'role': 'user',
            'content': user_content
        })
    else:
        data["messages"].append({
            'role': 'system',
            'content': system_prompt
        })
        # Build user message with images and prompt
        user_content = []
        print(f'key_words_dict:{key_words_dict}')
        for character_keywords, ref_img in key_words_dict.items():
            ref_img = Image.open(ref_img)
            current_ref_size = args.ref_size
            if current_ref_size == -1:
                current_ref_size = 512 if len(ref_imgs) == 1 else 320
            ref_img = preprocess_ref(ref_img, current_ref_size)
            print(f'character_keywords:{character_keywords}, ref_img:{ref_img}')
            user_content.append({
                'type': 'text',
                'text': f'Below is the image of character {character_keywords}'
            })
            user_content.append({
                'type': 'image_url',
                'image_url': {'url': convert_image_to_base64(ref_img)}
            })
        # Add the main prompt
        user_content.append({
            'type': 'text',
            'text': prompt
        })
        data["messages"].append({
            'role': 'user',
            'content': user_content
        })

    # Send API request
    res = make_api_request(data)
    if not res:
        return

    with megfile.smart_open(save_path, 'wb') as f:
        res.save(f, lossless=True)

    if save_history:
        generated_image_list = []
        generated_image_list.extend(
        [
        {
            'type': 'text',
            'text': f"This is generated {item[i]} shot"
        },
        {
                'type': 'image_url',
                'image_url': {'url': convert_image_to_base64(preprocess_ref(res, args.ref_size))}
        }
        ]
        )

        assistant_content = {
            'role': 'assistant',
            'content': generated_image_list
        }

        history_data = data["messages"].deepcopy()
        history_data.append(assistant_content)
        item['history'] = history_data

    print(f"💾 已保存: {save_path}")

    with last_save_lock:
        now = time.time()
        if last_save_time[0] is not None:
            print(f"⏱️ 与上次保存间隔: {now - last_save_time[0]:.2f} 秒")
        last_save_time[0] = now

    return history_data

    # except Exception as e:
    #     print(f"❌ 意外错误: {str(e)}")

def process_dataset(dataset, save_path, max_workers=3):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in dataset:
            future = executor.submit(process_single_item, item, save_path)
            futures.append(future)

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

# Placeholder for preprocess_ref (implement as needed)
def preprocess_ref(raw_image: Image.Image, long_size: int = 512):
    # 获取原始图像的宽度和高度
    image_w, image_h = raw_image.size
    # print(f'image_w, image_h:{image_w, image_h}')

    # 计算长边和短边
    if image_w >= image_h:
        new_w = long_size
        new_h = int((long_size / image_w) * image_h)
    else:
        new_h = long_size
        new_w = int((long_size / image_h) * image_w)

    # 按新的宽高进行等比例缩放
    # print(f'long_size / image_w:{long_size}|{image_w}|{long_size / image_w}')
    # print(f'(new_w, new_h):{(new_w, new_h)}')

    raw_image = raw_image.resize((new_w, new_h), resample=Image.LANCZOS)
    target_w = new_w // 16 * 16
    target_h = new_h // 16 * 16

    # 计算裁剪的起始坐标以实现中心裁剪
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # 进行中心裁剪
    raw_image = raw_image.crop((left, top, right, bottom))

    # 转换为 RGB 模式
    raw_image = raw_image.convert("RGB")
    return raw_image


if __name__ == "__main__":
    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory_en'
    method = "gpt4o"
    # save_path = "s3+b://jiaqioss/benchmarks/storfy_telling_benchmark/gpt4o/"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}"
    story_json_path = os.path.join(processed_dataset_path, "story_set.json")
    data_root = f""

    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]


    import time
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Starting new run with ID: {timestamp}")



    with open(story_json_path, 'r') as f:
        story_data = json.load(f)

    WildStory = story_data.get("WildStory", {})
    print(f'{dataset_name}: {len(WildStory)} stories loaded')

    for story_name, story_info in tqdm(WildStory.items(), desc="Processing Stories"):

        print(f"\nProcessing story: {story_name} at {timestamp}")

        save_path = f"{data_path}/outputs/{method}/{dataset_name}/{story_name}/{timestamp}"
        os.makedirs(f'{save_path}', exist_ok=True)

        # args.save_path =f"{data_path}/outputs/{method}/{dataset_name}/{story_name}/{timestamp}" # Old logic
        # os.makedirs(f'{args.save_path}', exist_ok=True) # Old logic

        if not story_info:
            print(f'Story {story_name} is empty, skipping')
            continue

        # print(f'当前处理的故事为{story_name}，{story_info}')
        print(f'Processing story: {story_name} with {len(story_info)} shots')

        assert story_info is not None or args.prompt is not None or args.eval_json_path is not None, "Please provide either prompt or eval_json_path"

        if story_info:
            data_dicts = story_info

        # data_dict_list = []
        # # for (tmp_i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
        # # for (tmp_i, data_dict) in itertools.product(enumerate(data_dicts)):
        # for tmp_i, data_dict in enumerate(data_dicts):
        #     print(f"正在处理第 {tmp_i} 个数据：{data_dict}")
        #     image_path = set()
        #     keywords_image_path_dict = {}
        #     for tmp_j, shot in enumerate(data_dict):
        #         print(f'shot:{shot}')
        #         image_path_list = shot["image_paths"]
        #         image_path.append(**image_path_list)
        #     for tmp_image_path in image_path:
        #         keywords = tmp_image_path.split("/")[-2]
        #         keywords_image_path_dict[keywords] = tmp_image_path

        #     data_dict_list.append(keywords_image_path_dict)



        # data_dict_list = []
        # for tmp_i, data_dict in enumerate(data_dicts):
        #     print(f"正在处理第 {tmp_i} 个数据：{data_dict}")
        #     image_path = set()
        #     keywords_image_path_dict = {}

        #     image_path_list = data_dict["image_paths"]
        #     image_path.append(**image_path_list)
        #     for tmp_image_path in image_path:
        #         keywords = tmp_image_path.split("/")[-2]
        #         keywords_image_path_dict[keywords] = tmp_image_path

        #     data_dict_list.append(keywords_image_path_dict)
        print(f'-----------------data_dicts:{data_dicts}')
        data_dict_list = []
        for tmp_i, data_dict in enumerate(data_dicts):
            print(f"正在处理第 {tmp_i+1} 个数据：{data_dict}")
            image_paths = set()  # 使用集合存储唯一路径
            keywords_image_path_dict = {}
            
            # 假设每个 data_dict 是包含多个 shot 的列表
            # 获取当前 shot 的 image_paths 列表（例如 ["path1.jpg", "path2.jpg"]）
            current_paths = data_dict["image_paths"]
            image_paths.update(current_paths)  # 用 update 添加多个路径到集合
            print(f'---------------len image_paths:{len(image_paths)}')
            
            # 提取关键词并构建字典
            for path in image_paths:
                keyword = path.split("/")[-2]  # 假设路径结构为 ".../keyword/image.jpg"
                keywords_image_path_dict[keyword] = path
            
            data_dict_list.append(keywords_image_path_dict)
        
        print(f'----------------len data_dict_list:{len(data_dict_list)}---------')


        history_data = None
        for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
            
            print(f'--------gei history_data | data_dict:{data_dict}')
            
            item = {}
            key_words_dict = data_dict_list[i]
            # key_words_dict_open = {}
            print(f'/////////////// key_words_dict:{key_words_dict}')
            # ref_imgs = [
            #     {char_key: Image.open(os.path.join(img_path))} if img_path # 绝对路径
            #     else key_words_dict_open[char_key] = ''
            #     for char_key, img_path in key_words_dict.items()
            # ]
            ref_imgs = [
                Image.open(os.path.join(img_path)) if story_info # 绝对路径
                else Image.open(os.path.join(data_root, img_path)) # 相对路径
                for img_path in data_dict["image_paths"]
            ]

            current_ref_size = args.ref_size
            if current_ref_size == -1:
                current_ref_size = 512 if len(ref_imgs) == 1 else 320

            processed_ref_imgs = [preprocess_ref(img, current_ref_size) for img in ref_imgs] ##should be pil

            item["index"] = i
            item["prompt"] = data_dict["prompt"]
            item["ref_imgs"] = processed_ref_imgs
            item["save_path"] = os.path.join(save_path, f"{i}_{j}.png")
            item["key_words_dict"] = key_words_dict

            history_data = process_single_item(item, history=history_data, save_history=True)

