#!/usr/bin/env python3
"""
终端图片显示工具 - 使用真彩色在命令行打印高清图片
支持 PNG, JPEG, GIF, BMP, WebP 等常见图片格式
"""

import sys
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    print("请先安装必要的库:")
    print("pip install Pillow numpy")
    sys.exit(1)


# 预设分辨率配置
RESOLUTION_PRESETS = {
    'low':    {'width': 80,  'desc': '低分辨率 (80 字符宽)'},
    'medium': {'width': 120, 'desc': '中分辨率 (120 字符宽)'},
    'high':   {'width': 180, 'desc': '高分辨率 (180 字符宽)'},
    'ultra':  {'width': 250, 'desc': '超高分辨率 (250 字符宽)'},
    'max':    {'width': 400, 'desc': '最大分辨率 (400 字符宽)'},
}


def print_image(
    image_path: str,
    width: int = None,
    height: int = None,
    resolution: str = None,
    scale: float = 1.0,
    use_color: bool = True,
    use_half_blocks: bool = True,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    brightness: float = 1.0,
):
    """
    在终端打印图片（真彩色高清版）
    
    Args:
        image_path: 图片文件路径
        width: 输出宽度（字符数），None 表示自动
        height: 输出高度（字符数），None 表示按比例计算
        resolution: 预设分辨率 (low/medium/high/ultra/max)
        scale: 缩放因子 (1.0=原始, 2.0=放大2倍)
        use_color: 是否使用颜色
        use_half_blocks: 是否使用半字符块（双倍垂直分辨率）
        contrast: 对比度调整 (1.0=原始, >1增强, <1降低)
        sharpness: 锐度调整 (1.0=原始, >1更锐利)
        brightness: 亮度调整 (1.0=原始)
    """
    try:
        # 打开并预处理图片
        img = Image.open(image_path)
        
        # 转换为RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 图像增强
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        
        # 确定输出尺寸
        orig_width, orig_height = img.size
        aspect_ratio = orig_height / orig_width
        
        # 优先级: resolution > width/height > 默认值
        if resolution and resolution in RESOLUTION_PRESETS:
            output_width = RESOLUTION_PRESETS[resolution]['width']
        elif width:
            output_width = width
        else:
            output_width = 150  # 默认宽度
        
        # 应用缩放因子
        output_width = int(output_width * scale)
        
        # 计算高度
        if height:
            # 用户指定了高度
            output_height = int(height * scale)
        else:
            # 按比例计算高度
            if use_half_blocks:
                # 半字符块模式：每个字符显示2个垂直像素
                output_height = int(output_width * aspect_ratio * 1.0)
            else:
                # 标准模式：字符宽高比约为1:2
                output_height = int(output_width * aspect_ratio * 0.5)
        
        # 高质量缩放
        img = img.resize((output_width, output_height), Image.Resampling.LANCZOS)
        
        # 显示分辨率信息
        print(f"分辨率: {output_width} x {output_height} (原图: {orig_width} x {orig_height})\n")
        
        # 转换为numpy数组（一次性操作，比逐像素访问快很多）
        pixels = np.array(img, dtype=np.uint8)
        
        # 构建输出
        reset = "\033[0m" if use_color else ""
        
        if use_half_blocks and use_color:
            # 【最高清晰度】真彩色 + 半字符块
            # 每个字符显示上下两个完整彩色像素
            _print_truecolor_half_blocks(pixels, reset)
        elif use_half_blocks:
            # 无颜色半字符块模式
            _print_grayscale_half_blocks(pixels)
        elif use_color:
            # 真彩色字符模式
            _print_truecolor_chars(pixels, reset)
        else:
            # 纯灰度ASCII模式
            _print_ascii(pixels)
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{image_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def _print_truecolor_half_blocks(pixels: np.ndarray, reset: str):
    """真彩色半字符块模式 - 最高清晰度"""
    height = len(pixels)
    
    for i in range(0, height - 1, 2):
        row_top = pixels[i]
        row_bottom = pixels[i + 1]
        
        # 使用列表推导式构建行（比循环append更快）
        line = "".join(
            f"\033[48;2;{t[0]};{t[1]};{t[2]}m\033[38;2;{b[0]};{b[1]};{b[2]}m▄"
            for t, b in zip(row_top, row_bottom)
        )
        print(line + reset)
    
    # 处理奇数行
    if height % 2 == 1:
        row = pixels[-1]
        line = "".join(
            f"\033[48;2;{p[0]};{p[1]};{p[2]}m\033[38;2;0;0;0m▄"
            for p in row
        )
        print(line + reset)


def _print_grayscale_half_blocks(pixels: np.ndarray):
    """灰度半字符块模式"""
    # 预计算灰度值（向量化）
    gray = (0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]).astype(np.uint8)
    height = len(gray)
    
    half_chars = {(0,0): ' ', (0,1): '▄', (1,0): '▀', (1,1): '█'}
    
    for i in range(0, height - 1, 2):
        top = gray[i] > 128
        bottom = gray[i + 1] > 128
        line = "".join(half_chars[(int(t), int(b))] for t, b in zip(top, bottom))
        print(line)


def _print_truecolor_chars(pixels: np.ndarray, reset: str):
    """真彩色字符模式"""
    chars = " ░▒▓█"
    num_chars = len(chars)
    
    # 预计算灰度值
    gray = (0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2])
    char_indices = np.clip((gray / 255 * num_chars).astype(int), 0, num_chars - 1)
    
    for row_pixels, row_indices in zip(pixels, char_indices):
        line = "".join(
            f"\033[38;2;{p[0]};{p[1]};{p[2]}m{chars[idx]}"
            for p, idx in zip(row_pixels, row_indices)
        )
        print(line + reset)


def _print_ascii(pixels: np.ndarray):
    """纯ASCII灰度模式"""
    chars = " .:-=+*#%@"
    num_chars = len(chars)
    
    # 向量化计算灰度和字符索引
    gray = (0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2])
    char_indices = np.clip((gray / 255 * num_chars).astype(int), 0, num_chars - 1)
    
    for row in char_indices:
        print("".join(chars[idx] for idx in row))


def main():
    # 构建分辨率预设说明
    resolution_help = "预设分辨率: " + ", ".join(
        f"{k}({v['width']})" for k, v in RESOLUTION_PRESETS.items()
    )
    
    parser = argparse.ArgumentParser(
        description="在终端显示图片（真彩色高清）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
分辨率预设:
  low    - 低分辨率 (80 字符宽)，适合小终端
  medium - 中分辨率 (120 字符宽)，日常使用
  high   - 高分辨率 (180 字符宽)，清晰显示
  ultra  - 超高分辨率 (250 字符宽)，大屏幕
  max    - 最大分辨率 (400 字符宽)，极致清晰

示例:
  python print.py image.jpg                # 显示图片（默认150宽）
  python print.py image.jpg -w 200         # 指定宽度 200
  python print.py image.jpg -H 50          # 指定高度 50
  python print.py image.jpg -r high        # 使用高分辨率预设
  python print.py image.jpg -r ultra       # 使用超高分辨率预设
  python print.py image.jpg --scale 1.5    # 放大 1.5 倍
  python print.py image.jpg -c 1.2 -s 1.5  # 增强对比度和锐度
  python print.py image.jpg --no-color     # 黑白模式
        """
    )
    
    parser.add_argument("image", nargs="?", help="图片文件路径")
    
    # 分辨率设置
    res_group = parser.add_argument_group("分辨率设置")
    res_group.add_argument("-w", "--width", type=int, help="输出宽度（字符数）")
    res_group.add_argument("-H", "--height", type=int, help="输出高度（字符数），默认按比例计算")
    res_group.add_argument("-r", "--resolution", choices=list(RESOLUTION_PRESETS.keys()),
                          help=resolution_help)
    res_group.add_argument("--scale", type=float, default=1.0, 
                          help="缩放因子 (默认1.0，如1.5表示放大50%%)")
    
    # 图像增强
    enhance_group = parser.add_argument_group("图像增强")
    enhance_group.add_argument("-c", "--contrast", type=float, default=1.0, 
                              help="对比度 (默认1.0，>1增强)")
    enhance_group.add_argument("-s", "--sharpness", type=float, default=1.0, 
                              help="锐度 (默认1.0，>1更锐利)")
    enhance_group.add_argument("-b", "--brightness", type=float, default=1.0, 
                              help="亮度 (默认1.0)")
    
    # 显示模式
    mode_group = parser.add_argument_group("显示模式")
    mode_group.add_argument("--no-color", action="store_true", help="禁用颜色（黑白模式）")
    mode_group.add_argument("--no-half-blocks", action="store_true", 
                           help="禁用半字符块（降低垂直分辨率）")
    
    args = parser.parse_args()
    
    # 查找图片文件
    image_path = None
    if args.image:
        image_path = Path(args.image)
    else:
        # 尝试默认文件名
        for name in ["cats_images.png", "cats_image.jpeg", "cats_image.jpg"]:
            if Path(name).exists():
                image_path = Path(name)
                break
    
    if image_path is None or not image_path.exists():
        if image_path:
            print(f"错误: 文件 '{image_path}' 不存在")
        else:
            parser.print_help()
        sys.exit(1)
    
    print(f"正在显示: {image_path}")
    
    print_image(
        str(image_path),
        width=args.width,
        height=args.height,
        resolution=args.resolution,
        scale=args.scale,
        use_color=not args.no_color,
        use_half_blocks=not args.no_half_blocks,
        contrast=args.contrast,
        sharpness=args.sharpness,
        brightness=args.brightness,
    )


if __name__ == "__main__":
    main()
