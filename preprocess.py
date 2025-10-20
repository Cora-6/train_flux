# %%
import os
import json
import math
import random
import numpy as np
import torch
import torch.utils.data as td
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
from tqdm import tqdm
import pickle

# ---------- 1. 加载编码器（保持不变） ----------
def load_encoders():
    clip_tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="tokenizer", use_fast=False)
    clip_encoder = CLIPTextModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="text_encoder").eval().cuda()
    
    # 强制 T5 用 float16 减少显存（之前的优化保留）
    t5_tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="tokenizer_2", use_fast=False)
    t5_encoder = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2",
        torch_dtype=torch.float16
    ).eval().cuda()
    
    return [clip_tokenizer, t5_tokenizer], [clip_encoder, t5_encoder]

# ---------- 2. 生成文本（保持不变） ----------
def generate_text(cfg):
    camera_height = cfg['location'][2]
    tilt_rad = math.pi / 2 - cfg['rotation_euler'][0]
    tilt_deg = math.degrees(tilt_rad)
    lens = cfg['data']['lens']
    conf_str = f"{{camera_height:{camera_height},tilt_deg:{tilt_deg:.1f},lens:{lens}}}"
    return f"city topview from camera, camera conf as following. {conf_str}"

# ---------- 3. 数据集类（保持不变） ----------
class TextDS(td.Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding.input_ids

# ---------- 4. 核心修改：encode_all 嵌入保存步骤 ----------
@torch.no_grad()
def encode_all(
    texts, 
    tokenizers,  
    encoders,    
    save_dir,  # 新增：保存路径参数
    dir_names, # 新增：对应文本的二级目录名（用于命名文件）
    batch_size=1, 
    dtype=torch.float16, 
    proportion_empty_prompts=0.0, 
    is_train=True
):
    device = encoders[0].device
    clip_tokenizer, t5_tokenizer = tokenizers
    clip_encoder, t5_encoder = encoders

    # 第一步：文本预处理（保持不变）
    captions = []
    for caption in texts:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
    texts = captions

    # 第二步：CLIP 编码（新增：批量保存 CLIP 特征，不缓存全量）
    clip_max_len = clip_tokenizer.model_max_length
    ds_clip = TextDS(texts, clip_tokenizer, clip_max_len)
    dl_clip = td.DataLoader(ds_clip, batch_size=batch_size, shuffle=False, num_workers=0)

    # 存储 CLIP 特征的临时字典（batch 索引 → 特征）
    clip_pooled_dict = {}
    with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
        for batch_idx, batch_ids in enumerate(dl_clip):
            batch_ids = batch_ids.squeeze(1).to(device, non_blocking=True)
            clip_output = clip_encoder(batch_ids, output_hidden_states=True)
            pooled_embeds = clip_output.pooler_output  # [batch_size, 768]

            # 按样本索引保存 CLIP 特征（关键：不缓存全量，生成一个存一个）
            for idx_in_batch in range(pooled_embeds.shape[0]):
                global_idx = batch_idx * batch_size + idx_in_batch  # 全局样本索引
                if global_idx >= len(dir_names):
                    break  # 避免索引超出（防止 batch 最后一轮不足）
                dir_name = dir_names[global_idx]  # 获取当前样本的目录名
                # 保存 CLIP 池化特征
                torch.save(
                    pooled_embeds[idx_in_batch].unsqueeze(0).cpu(),  # 加 batch 维度
                    os.path.join(save_dir, f"{dir_name}_pooled_embeds.pt")
                )

            clip_pooled_dict[batch_idx] = pooled_embeds  # 临时存当前 batch 的 CLIP 特征（供 T5 同步）
            del batch_ids, clip_output, pooled_embeds
            torch.cuda.empty_cache()
    print("CLIP encoder & save finished!")

    # 第三步：T5 编码（同步 CLIP 临时特征，保存 T5 特征，不缓存全量）
    t5_max_len = t5_tokenizer.model_max_length
    ds_t5 = TextDS(texts, t5_tokenizer, t5_max_len)
    dl_t5 = td.DataLoader(ds_t5, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
        for batch_idx, batch_ids in enumerate(dl_t5):
            batch_ids = batch_ids.squeeze(1).to(device, non_blocking=True)  # [batch_size, 512]
            t5_output = t5_encoder(batch_ids)
            prompt_embeds = t5_output.last_hidden_state  # [batch_size, 512, 1024]
            
            t5_text_ids = batch_ids.unsqueeze(-1).expand(-1, -1, 3)  # [batch_size, 512, 3]
            print("lkj:shape",prompt_embeds.shape)

            # 按样本索引保存 T5 特征（与 CLIP 同步索引）# 1. 保存 T5 序列特征：[1, 512, 1024]
            for idx_in_batch in range(prompt_embeds.shape[0]):
                global_idx = batch_idx * batch_size + idx_in_batch
                if global_idx >= len(dir_names):
                    break
                dir_name = dir_names[global_idx]
                # 保存 T5 序列特征 # 2. 保存 text_ids：[1, 512, 3]
                torch.save(
                    prompt_embeds[idx_in_batch].unsqueeze(0).cpu(),
                    os.path.join(save_dir, f"{dir_name}_prompt_embeds.pt")
                )
                # 保存 T5 token ID
                torch.save(
                    t5_text_ids[idx_in_batch].unsqueeze(0).cpu(),
                    os.path.join(save_dir, f"{dir_name}_text_ids.pt")
                )

            del batch_ids, t5_output, prompt_embeds, t5_text_ids
            del clip_pooled_dict[batch_idx]  # 删除已匹配的 CLIP 临时特征
            torch.cuda.empty_cache()
    print("T5 encoder & save finished!")

    # 无需返回全量特征（已实时保存）
    return None

# ---------- 5. 主流程（删除原保存代码，调用新 encode_all） ----------
def main():
    root_dir = "/data/train"
    save_dir = f"{root_dir}/embeds_flux"
    os.makedirs(save_dir, exist_ok=True)

    # 5.1 筛选二层文件夹（保持不变，额外收集目录名）
    second_dirs = []
    dir_names = []  # 新增：收集二级目录名（用于 encode_all 命名文件）
    for d in os.listdir(root_dir):
        if d.startswith('embeds'):
            continue
        second_path = os.path.join(root_dir, d)
        if not os.path.isdir(second_path):
            continue
        subdirs = [os.path.join(second_path, sd) for sd in os.listdir(second_path) if os.path.isdir(os.path.join(second_path, sd))]
        if not subdirs:
            continue
        first_sub = subdirs[0]
        camconf_path = os.path.join(first_sub, 'camconf.json')
        if os.path.isfile(camconf_path):
            second_dirs.append(second_path)
            dir_names.append(d)  # 保存二级目录名（如 "dir1"）
    print(f"Valid second-level dirs: {len(second_dirs)}")
    print("First 5 dirs:", second_dirs[:5])

    # 5.2 生成文本（保持不变）
    texts = []
    for dir_path in second_dirs:
        subdirs = [os.path.join(dir_path, sd) for sd in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, sd))]
        first_sub = subdirs[0]
        camconf_path = os.path.join(first_sub, 'camconf.json')
        with open(camconf_path, 'r') as f:
            cfg = json.load(f)
        texts.append(generate_text(cfg))
    print("Text generation finished! Total texts:", len(texts))

    # 5.3 加载编码器并编码（调用新 encode_all，传入 save_dir 和 dir_names）
    tokenizers, encoders = load_encoders()
    print("Encoders loaded successfully!")
    encode_all(  # 不再接收返回值（已实时保存）
        texts,
        tokenizers,
        encoders,
        save_dir=save_dir,  # 新增参数：保存路径
        dir_names=dir_names,  # 新增参数：目录名（用于命名文件）
        batch_size=1,
        dtype=torch.float16,
        proportion_empty_prompts=0.1,
        is_train=True
    )
    print("Text encoding & save finished!")

    # 5.4 硬链接（保持不变，逻辑不受影响）
    cache_path = os.path.join(root_dir, '.dataset_cache.pkl')
    if not os.path.exists(cache_path):
        print(f"Warning: {cache_path} not found, skip linking!")
        return
    
    with open(cache_path, 'rb') as f:
        groups = pickle.load(f)['groups']
    
    for train_idx, group in enumerate(tqdm(groups, desc='Creating hard links')):
        second_level_dir = os.path.dirname(group[0])
        bname = os.path.basename(second_level_dir)
        
        src_prompt = os.path.join(save_dir, f"{bname}_prompt_embeds.pt")
        src_pooled = os.path.join(save_dir, f"{bname}_pooled_embeds.pt")
        src_text_ids = os.path.join(save_dir, f"{bname}_text_ids.pt")
        
        dst_prompt = os.path.join(save_dir, f"{train_idx}_prompt_embeds.pt")
        dst_pooled = os.path.join(save_dir, f"{train_idx}_pooled_embeds.pt")
        dst_text_ids = os.path.join(save_dir, f"{train_idx}_text_ids.pt")
        
        for src, dst in [(src_prompt, dst_prompt), (src_pooled, dst_pooled), (src_text_ids, dst_text_ids)]:
            if not os.path.exists(dst) and os.path.exists(src):
                os.link(src, dst)
    
    print(f"Finished! Unique embeddings: {len(second_dirs)}, Training links: {len(groups)}")

if __name__ == "__main__":
    main()


# %%
import os, glob, pickle

root = "/data/train"
embed_dir = f"{root}/embeds_flux"

# 1. 二层文件夹数（只拿第一个子目录）
second_dirs = []
for d in os.listdir(root):
    second = os.path.join(root, d)
    if os.path.isdir(second):
        first_sub = os.listdir(second)[0]
        if os.path.isfile(os.path.join(second, first_sub, 'camconf.json')):
            second_dirs.append(second)
print("二层文件夹数 :", len(second_dirs))
print(second_dirs[:5])  # 看前 5 个路径
# 2. 真实 embed 文件数
unique_files = glob.glob(f"{embed_dir}/*_prompt_embeds.pt")
print("唯一 embed  :", len(unique_files))

# 3. 链接总数
with open(f"{root}/.dataset_cache.pkl", 'rb') as f:
    groups = pickle.load(f)['groups']
print("需要链接数  :", len(groups))

link_files = glob.glob(f"{embed_dir}/[0-9]*_prompt_embeds.pt")
print("实际链接数  :", len(link_files))

# 4. 抽样验证硬链接是否指向同一 inode
if link_files:
    src = unique_files[0]
    dst = link_files[0]
    src_inode = os.stat(src).st_ino
    dst_inode = os.stat(dst).st_ino
    print("硬链接检查  :", "✅ 同一 inode" if src_inode == dst_inode else "❌ 不同")


