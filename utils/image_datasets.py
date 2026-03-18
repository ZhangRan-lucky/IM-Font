import math
import random
from pathlib import Path

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from . import logger

# 🔥 新增：导入IDS解析模块
try:
    import cn
    IDS_AVAILABLE = True
except ImportError:
    logger.log("⚠️ 警告: 无法导入cn模块，IDS功能将不可用")
    IDS_AVAILABLE = False


# MPI shim for single machine
class MPIShim:
    class COMM_WORLD:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def Get_size():
            return 1

        @staticmethod
        def bcast(data, root=0):
            return data


MPI = MPIShim


def load_data(
        *,
        data_dir,
        batch_size,
        image_size,
        deterministic=False,
        random_crop=False,
        random_flip=False,
        classifier_free=False,
        total_char_file=None,
        use_ids=True,  # 🔥 默认启用IDS
        ids_max_len=40,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    chars_ids = None  # IDS数据
    ids_vocab = None  # IDS词表

    # 如果提供了 total_char_file，使用其定义的顺序
    if total_char_file is not None:
        logger.log('=' * 60)
        logger.log(f"使用 {total_char_file} 定义字符顺序")

        with open(total_char_file, 'r', encoding='utf-8') as f:
            total_chars = f.read().strip()

        logger.log(f"总字符数: {len(total_chars)}")
        logger.log(f"前30个字符: {total_chars[:30]}")

        # 🔥 新增：加载IDS数据
        if use_ids and IDS_AVAILABLE:
            logger.log('=' * 60)
            logger.log("开始加载IDS数据...")

            try:
                # 递归解析IDS
                ids_map, ids_set, ids_base = cn.resolve_IDS_babelstone(
                    total_chars,
                    level='stroke'
                )

                logger.log(f"✅ IDS解析完成!")
                logger.log(f"  - 成功解析: {len(ids_map)}/{len(total_chars)} 个字符")
                logger.log(f"  - 基础部件数量: {len(ids_base)}")

                # 构建IDS词表
                special_tokens = ['<PAD>', '<UNK>']
                base_components = sorted(ids_base.keys())
                ids_vocab = special_tokens + list(cn.IDC) + base_components
                ids2idx = {v: i for i, v in enumerate(ids_vocab)}

                logger.log(f"  - IDS词表大小: {len(ids_vocab)}")
                logger.log(f"  - 词表内容(前30): {ids_vocab[:30]}")

                # 转换IDS为索引序列
                chars_ids = {}
                failed_chars = []
                ids_lengths = []

                for char_idx, char in enumerate(total_chars):
                    ids_tuple = ids_map.get(char, (char,))

                    # 如果解析失败，记录
                    if ids_tuple == (char,) and char not in ids_base:
                        failed_chars.append(char)

                    # 转为索引
                    ids_indices = [ids2idx.get(c, ids2idx['<UNK>']) for c in ids_tuple]
                    ids_lengths.append(len(ids_indices))

                    # padding到固定长度
                    if len(ids_indices) < ids_max_len:
                        ids_indices += [ids2idx['<PAD>']] * (ids_max_len - len(ids_indices))
                    else:
                        ids_indices = ids_indices[:ids_max_len]

                    chars_ids[char_idx] = np.array(ids_indices, dtype=np.int64)

                # 统计信息
                ids_lengths = np.array(ids_lengths)
                logger.log(f"\n📊 IDS序列统计:")
                logger.log(f"  - 平均长度: {ids_lengths.mean():.2f}")
                logger.log(f"  - 中位数: {np.median(ids_lengths):.0f}")
                logger.log(f"  - 最短: {ids_lengths.min()}")
                logger.log(f"  - 最长: {ids_lengths.max()}")
                logger.log(f"  - 解析失败: {len(failed_chars)} 个")
                if failed_chars:
                    logger.log(f"    失败字符: {failed_chars[:20]}")

                # 计算非零比例
                total_elements = len(total_chars) * ids_max_len
                non_zero_elements = sum(ids_lengths)
                logger.log(f"  - 非零比例: {non_zero_elements / total_elements * 100:.1f}%")
                logger.log('=' * 60)

            except Exception as e:
                logger.log(f"❌ IDS加载失败: {e}")
                import traceback
                traceback.print_exc()
                logger.log("将禁用IDS功能")
                use_ids = False
                chars_ids = None
                ids_vocab = None

        elif use_ids and not IDS_AVAILABLE:
            logger.log("❌ 无法使用IDS: cn模块未导入")
            use_ids = False

        # 从文件名提取字符索引：00001.png -> 索引0
        classes = []
        for path in all_files:
            basename = bf.basename(path)
            try:
                char_idx = int(basename.split('.')[0]) - 1
                if 0 <= char_idx < len(total_chars):
                    classes.append(char_idx)
                else:
                    logger.log(f"警告: {path} 的索引 {char_idx} 超出范围")
                    classes.append(0)
            except:
                logger.log(f"警告: 无法解析 {path}")
                classes.append(0)
    else:
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    sty_class_names = [bf.dirname(path).split("/")[-1] for path in all_files]
    sty_sorted_classes = {x: i for i, x in enumerate(sorted(set(sty_class_names)))}
    sty_classes = [sty_sorted_classes[x] for x in sty_class_names]

    # 创建数据集
    if classifier_free:
        dataset = ImageDataset_Clsfree(
            resolution=image_size,
            image_paths=all_files,
            classes=classes,
            sty_classes=sty_classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            chars_ids=chars_ids,
            ids_vocab_size=len(ids_vocab) if ids_vocab else None,
        )
    else:
        dataset = ImageDataset(
            resolution=image_size,
            image_paths=all_files,
            classes=classes,
            sty_classes=sty_classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            chars_ids=chars_ids,
            ids_vocab_size=len(ids_vocab) if ids_vocab else None,
        )

    # 🔥 返回ids_vocab_size供模型使用
    dataset.ids_vocab_size = len(ids_vocab) if ids_vocab else None

    # 🔥 恢复原始配置 - 删除所有"优化"参数
    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,  # 恢复原值
            drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,  # 恢复原值
            drop_last=True,
        )

    # 返回可迭代的数据加载器
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            sty_classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
            chars_ids=None,
            ids_vocab_size=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.style_classes = None if sty_classes is None else sty_classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.chars_ids = chars_ids
        self.ids_vocab_size = ids_vocab_size

        if self.style_classes is not None:
            sty_classes_set = set(self.style_classes)
            self.sty_classes_idxs = {}
            for sty_class in sty_classes_set:
                self.sty_classes_idxs[sty_class] = np.where(np.array(self.style_classes) == sty_class)[0]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            out_dict["mask_y"] = False

        if self.style_classes is not None:
            sty_img_idxs = self.sty_classes_idxs[self.style_classes[idx]]
            rand_sty_idx = np.random.choice(sty_img_idxs)
            sty_img_path = self.local_images[rand_sty_idx]
            with bf.BlobFile(sty_img_path, "rb") as f:
                sty_pil_image = Image.open(f)
                sty_pil_image.load()
            sty_pil_image = sty_pil_image.convert("RGB")
            sty_arr = center_crop_arr(sty_pil_image, self.resolution)
            sty_arr = sty_arr.astype(np.float32) / 127.5 - 1

            # 添加IDS
            if self.chars_ids is not None:
                out_dict["ids"] = self.chars_ids[self.local_classes[idx]]
                out_dict["mask_ids"] = False

            return [np.transpose(arr, [2, 0, 1]), np.transpose(sty_arr, [2, 0, 1])], out_dict

        return np.transpose(arr, [2, 0, 1]), out_dict


class ImageDataset_Clsfree(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            sty_classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
            chars_ids=None,
            ids_vocab_size=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.style_classes = None if sty_classes is None else sty_classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.chars_ids = chars_ids
        self.ids_vocab_size = ids_vocab_size

        if self.style_classes is not None:
            sty_classes_set = set(self.style_classes)
            self.sty_classes_idxs = {}
            for sty_class in sty_classes_set:
                self.sty_classes_idxs[sty_class] = np.where(np.array(self.style_classes) == sty_class)[0]

        logger.log('=' * 20)
        logger.log("using classifier-free dataset")
        logger.log('=' * 20)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            out_dict["mask_y"] = random.random() < 0.3

        if self.style_classes is not None:
            sty_img_idxs = self.sty_classes_idxs[self.style_classes[idx]]
            rand_sty_idx = np.random.choice(sty_img_idxs)
            sty_img_path = self.local_images[rand_sty_idx]
            with bf.BlobFile(sty_img_path, "rb") as f:
                sty_pil_image = Image.open(f)
                sty_pil_image.load()
            sty_pil_image = sty_pil_image.convert("RGB")
            sty_arr = center_crop_arr(sty_pil_image, self.resolution)
            sty_arr = sty_arr.astype(np.float32) / 127.5 - 1

            # 添加IDS（classifier-free training）
            if self.chars_ids is not None:
                out_dict["ids"] = self.chars_ids[self.local_classes[idx]]
                out_dict["mask_ids"] = random.random() < 0.3

            return [np.transpose(arr, [2, 0, 1]), np.transpose(sty_arr, [2, 0, 1])], out_dict

        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]