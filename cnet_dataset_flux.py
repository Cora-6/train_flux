import os, glob, json, math, pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from skimage.transform import resize
from transformers import PreTrainedTokenizer
from functools import partial
from typing import Optional, List

def load_image(path):
    img = cv2.imread(path)[..., ::-1] / 255.
    if img is None:
        raise FileNotFoundError(path)
    return img

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

# def generate_text(cfg):
#     """
#     cfg: 单个 camera_cfg
#     """
#     tilt_deg = round(math.degrees(math.pi/2 - cfg['rotation_euler'][0]))
#     views = ['front', 'right', 'back', 'left']
#     view_txt = ", ".join([f"{v} view" for v in views])
#     return (f"aerial oblique {tilt_deg}° of the same cityview, "
#             f"2×2 grid: {view_txt}")

def generate_text(config):
    """
    根据配置信息生成描述文本。

    Args:
        config (dict): 配置信息。

    Returns:
        str: 生成的描述文本。
    """
    camera_height = config['location'][2]  # Z坐标
    tilt_rad = math.pi / 2 - config['rotation_euler'][0]
    tilt_deg = math.degrees(tilt_rad)
    lens = config['data']['lens']
    conf_str = f"{{camera_height:{camera_height},tilt_deg:{tilt_deg:.1f},lens:{lens}}}"
    return f"city topview from camera, camera conf as following. {conf_str}"

def preprocess_image(img_np, size):
    if img_np.shape[:2] != (size, size):
        img_np = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_AREA)
    return img_np.transpose(2,0,1).astype(np.float32)*2-1

class HSIControlNetDataset(Dataset):
    """
    每次返回一个 4-in-1 样本：
        image            : (3,512,512)
        conditioning     : (48,512,512)
        text / input_ids : 与原来一致
    """
    def __init__(self,
                 root_dir: str,
                 image_size: int = 512,
                 ):
        self.root_dir   = root_dir
        self.preprocessed_text_dir = f"{root_dir}/embeds_flux"
        self.image_size = image_size
        print("dataset name:",root_dir)
         # 1. 缓存路径
        cache_file = os.path.join(root_dir, '.dataset_cache.pkl')
        incremental_marker = os.path.join(root_dir, '.last_scan')   # 记录上次扫描时间
        last_mtime = 0.0          # 默认当作“很久以前”
        if os.path.exists(incremental_marker):
            with open(incremental_marker) as f:
                content = f.read().strip()
                if content:                       # 非空才转
                    last_mtime = float(content)
                # 否则保持 0.0，一定会触发重新扫描

        # 2. 计算目录下最新 json 的修改时间（无论是否需要扫描，都先算出来）
        all_jsons = glob.glob(f"{root_dir}/**/*.json", recursive=True)
        newest_json = max(all_jsons, key=os.path.getmtime) if all_jsons else None
        newest_mtime = os.path.getmtime(newest_json) if newest_json else 0

        # 2. 如果需要增量扫描，先扫一遍新增文件夹
        need_full_scan = False
        incremental_marker = os.path.join(root_dir, '.last_scan')
        if os.path.exists(incremental_marker):
            # last_mtime = float(open(incremental_marker).read().strip())
            if newest_mtime > last_mtime:          # 有更新
                need_full_scan = True
        else:
            need_full_scan = True                  # 第一次

        # 3. 如果缓存不存在或需要增量/全量，就重新建
        if not os.path.exists(cache_file) or need_full_scan:
            print("[Dataset] 第一次运行 or 检测到新增样本，重建缓存...")
            jsons = sorted(glob.glob(f"{root_dir}/**/*.json", recursive=True))
            folders = [os.path.dirname(j) for j in jsons]
            groups = [folders[i:i+4] for i in range(0, len(folders), 4)
                      if len(folders[i:i+4]) == 4]

            # 落盘
            with open(cache_file, 'wb') as f:
                pickle.dump({'jsons': jsons, 'folders': folders, 'groups': groups}, f)
            with open(incremental_marker, 'w') as f:
                f.write(str(os.path.getmtime(newest_json) if newest_json else 0))
        else:
            print("[Dataset] 加载缓存...")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            jsons, folders, groups = cache['jsons'], cache['folders'], cache['groups']

        # 4. 赋给 self，后面逻辑完全不变
        self.groups = groups
        print("folders len", len(folders))
        print("groups  len", len(groups))
        # 3. 全局 HSI 数据（仅加载一次）
        self.hsi_path = '/data/try_small/result_new/Houston18.mat'
        with h5py.File(self.hsi_path, 'r') as f:
            self.hsi = np.array(f['ori_data'])          # (C,H,W)
        self.hsi_norm = self.hsi / (self.hsi.max((1,2), keepdims=True) + 1e-8)
        self.hsi_norm = self.hsi_norm.transpose(1, 2, 0)
        # 固定透视矩阵
        self.P = np.array([
            [4.80769231e-02, 2.55899318e+00, 4.44000000e+02],
            [2.55528846e+00, -4.82433141e-02, 1.21000000e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])

        # pos_P transform hsi_norm's shape?
        self.warped_hsi = cv2.warpPerspective(self.hsi_norm, self.P, (4096, 1024),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        out_size_per_subimage = image_size // 2
        scale = out_size_per_subimage / 512 # 512 is the canonical size
        self.scale_mat = np.array([
            [scale, 0., 0],
            [0., scale, 0.],
            [0., 0., 1.]
        ])

    def __len__(self):
        return len(self.groups)

    # ---------- 下面 3 个工具函数 -------------
    def _warp_hsi(self, H_mat, out_size):
        """把 (C,H,W) warp 成 (C,out_size,out_size)"""
        M = self.scale_mat @ H_mat
        warped = cv2.warpPerspective(self.warped_hsi, M, (out_size, out_size),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        res = warped.transpose(2,0,1)
        return res       # (C,H,W)

    @staticmethod
    def _stitch_2x2_np(cubes):
        """
        cubes: list[ndarray] 4 个 (C,H,W)
        return: (C,2H,2W)
        """
        a = np.concatenate([cubes[0], cubes[1]], 2)
        b = np.concatenate([cubes[2], cubes[3]], 2)
        return np.concatenate([a, b], 1)*2-1

    def _stitch_2x2_img(self, imgs):
        """把 4 张 RGB(H,W,3) 2×2 拼 -> resize -> (3,H,W)"""
        target_subimage_size = self.image_size // 2
        imgs = [cv2.resize(i, (target_subimage_size, target_subimage_size)) for i in imgs]
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        H0,W0 = imgs[0].shape[:2]
        for idx, im in enumerate(imgs):
            r,k = divmod(idx,2)
            canvas[r*H0:(r+1)*H0, k*W0:(k+1)*W0] = im
        #canvas = cv2.resize(canvas, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return canvas.transpose(2,0,1)*2-1
    # -----------------------------------------

    def __getitem__(self, idx):
        folders = self.groups[idx]                       # 4 视角目录
        target_subimage_size = self.image_size // 2

        # 1. 2×2 RGB 拼接
        imgs_3d = [load_image(os.path.join(f, 'render_3d.png')) for f in folders]
        image = self._stitch_2x2_img(imgs_3d)            # (3, 512, 512)

        # 2. 2×2 HSI 拼接
        Hs = [np.load(os.path.join(f, 'homography_matrix.npy')) for f in folders]
        warp_fn = partial(self._warp_hsi, out_size=target_subimage_size)
        cubes = list(map(warp_fn, Hs))
        cond = self._stitch_2x2_np(cubes)                # (48, 512, 512)

        # 3. 文本：只用第一个视角的相机配置
        # cfg = load_config(os.path.join(folders[0], 'camconf.json'))
        # text = generate_text(cfg)

        prompt_embeds = torch.load(
            os.path.join(self.preprocessed_text_dir, f"{idx}_prompt_embeds.pt"),
            weights_only=True
        )
        # 加载 CLIP 池化特征（pooled_embeds，对应 pooled_prompt_embeds）
        pooled_prompt_embeds = torch.load(
            os.path.join(self.preprocessed_text_dir, f"{idx}_pooled_embeds.pt"),
            weights_only=True
        )
        # 加载 T5 Token ID（text_ids）
        text_ids = torch.load(
            os.path.join(self.preprocessed_text_dir, f"{idx}_text_ids.pt"),
            weights_only=True
        )

        text_ids_path = os.path.join(self.preprocessed_text_dir, f"{idx}_text_ids.pt")
        text_ids = torch.load(text_ids_path, weights_only=True)

        # text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)
        return dict(
            pixel_values=torch.tensor(image),                    # RGB 2×2 拼图：(3, 512, 512)
            conditioning_pixel_values=torch.tensor(cond),        # HSI 条件：(48, 512, 512)
            prompt_embeds=prompt_embeds.squeeze(0),              # T5 序列特征：[512, 4096]
            pooled_prompt_embeds=pooled_prompt_embeds.squeeze(0),# CLIP 池化特征：(768)
            text_ids=text_ids.squeeze(0),                        # T5 Token ID：(512,3)
            time_ids=torch.tensor([self.image_size, self.image_size, 0.0]),  # 示例 time_ids，长度 3
        )
    
    def with_transform(self, transform_func):
        """
        模仿 datasets.Dataset.with_transform 的签名，
        返回一个 WrappedDataset，每次 __getitem__ 时先调原始数据再跑 transform_func。
        """
        return TransformedDataset(self, transform_func)
    
class TransformedDataset(torch.utils.data.Dataset):
    """薄封装：先取原始数据，再跑 transform_func(batch)。"""
    def __init__(self, base_ds, transform_func):
        self.base_ds = base_ds
        self.transform_func = transform_func

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        # 取一条样本（dict）
        sample = self.base_ds[idx]
        # transform_func 要求批处理，我们包一层列表再解包
        batch = self.transform_func([sample])
        return batch[0]


