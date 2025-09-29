#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_busi_zip.py
- 解压你下载好的 BUSI 压缩包到标准结构:
  datasets/BUSI/Dataset_BUSI_with_GT/{benign,malignant,normal}
- 可选删除 *_mask.png（做分类时建议开启）
- 统计每类图像数量

用法示例：
  python prepare_busi_zip.py
  python prepare_busi_zip.py --zip datasets/BUSI/archive.zip --out datasets/BUSI/Dataset_BUSI_with_GT --remove-masks --force
"""

import argparse
import os
import re
import shutil
import sys
import zipfile
from glob import glob

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="./datasets/BUSI/archive.zip",
                    help="BUSI 压缩包路径（.zip）")
    ap.add_argument("--out", default="./datasets/BUSI/Dataset_BUSI_with_GT",
                    help="输出标准目录")
    ap.add_argument("--tmp", default="./datasets/BUSI/_tmp_extract",
                    help="临时解压目录")
    ap.add_argument("--remove-masks", action="store_true",
                    help="删除 *_mask.png（分类任务建议开启）")
    ap.add_argument("--force", action="store_true",
                    help="若输出目录已存在，仍然覆盖/合并")
    ap.add_argument("--move", action="store_true",
                    help="移动而不是复制（省空间；原解压内容会被清理）")
    return ap.parse_args()

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def unzip_to(src_zip, dst_dir):
    if not os.path.isfile(src_zip):
        sys.exit(f"[Error] 找不到压缩包：{src_zip}")
    safe_mkdir(dst_dir)
    with zipfile.ZipFile(src_zip, 'r') as zf:
        zf.extractall(dst_dir)
    return dst_dir

def find_candidate_root(tmp_dir):
    """
    在临时解压目录里找真正的数据根目录：
    - 优先名字里含 'Dataset' & 'BUSI' & 'GT'
    - 否则找包含 benign/malignant/normal 三类子目录的路径
    """
    # 1) 名字匹配优先
    best = None
    for root, dirs, files in os.walk(tmp_dir):
        name = os.path.basename(root).lower()
        if all(k in name for k in ("dataset", "busi")) and ("gt" in name or "with_gt" in name or "with-gt" in name):
            best = root
            break
    if best:
        return best

    # 2) 结构匹配（含三类子目录）
    for root, dirs, files in os.walk(tmp_dir):
        sub = set(d.lower() for d in dirs)
        if {"benign", "malignant", "normal"} <= sub:
            return root

    # 3) 兜底：返回 tmp_dir 自身
    return tmp_dir

def merge_tree(src, dst, move=False):
    safe_mkdir(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
            if move:
                shutil.rmtree(s, ignore_errors=True)
        else:
            shutil.copy2(s, d)
            if move:
                try: os.remove(s)
                except: pass

def remove_masks(dst_root):
    removed = 0
    for path in glob(os.path.join(dst_root, "**", "*_mask.png"), recursive=True):
        try:
            os.remove(path); removed += 1
        except Exception:
            pass
    print(f"[INFO] 已删除掩码文件：{removed} 个")

def count_classes(dst_root):
    def cnt(sub):
        p = os.path.join(dst_root, sub)
        if not os.path.isdir(p): return 0
        return sum(1 for f in os.listdir(p)
                   if os.path.isfile(os.path.join(p, f))
                   and re.search(r"\.(png|jpg|jpeg)$", f, re.I)
                   and not f.endswith("_mask.png"))
    stats = {k: cnt(k) for k in ("benign","malignant","normal")}
    total = sum(stats.values())
    print(f"[DONE] 目标目录：{dst_root}")
    print(f"[STATS] benign={stats['benign']}, malignant={stats['malignant']}, normal={stats['normal']}, total={total}")
    return stats

def main():
    args = parse_args()

    # 0) 输出目录处理
    if os.path.exists(args.out):
        if not args.force:
            print(f"[WARN] 输出目录已存在：{args.out}（如需覆盖请加 --force）")
        else:
            print(f"[INFO] 输出目录已存在，启用合并/覆盖：{args.out}")

    # 1) 解压
    print(f"[INFO] 解压 {args.zip} -> {args.tmp}")
    unzip_to(args.zip, args.tmp)

    # 2) 查找数据根目录
    cand_root = find_candidate_root(args.tmp)
    print(f"[INFO] 检测到数据根目录：{cand_root}")

    # 3) 规范类名目录到目标结构
    #    若 cand_root 就是类文件夹所在目录，直接合并过去
    print(f"[INFO] 写入目标目录：{args.out}")
    merge_tree(cand_root, args.out, move=args.move)

    # 4) 可选：删除掩码
    if args.remove_masks:
        remove_masks(args.out)

    # 5) 统计
    count_classes(args.out)

    # 6) 清理临时解压
    try:
        shutil.rmtree(args.tmp, ignore_errors=True)
        print(f"[CLEAN] 已清理临时目录：{args.tmp}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
