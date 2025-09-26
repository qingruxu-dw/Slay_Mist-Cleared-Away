import sys
print("Python version:", sys.version)

# 检查主要依赖
dependencies = [
    'torch',
    'torchvision', 
    'cv2',
    'skimage',
    'numpy',
    'PIL',
    'timm',
    'einops',
    'ftfy',
    'regex',
    'tqdm'
]

for dep in dependencies:
    try:
        if dep == 'cv2':
            import cv2
            print(f"✓ {dep} version: {cv2.__version__}")
        elif dep == 'torch':
            import torch
            print(f"✓ {dep} version: {torch.__version__}")
        elif dep == 'torchvision':
            import torchvision
            print(f"✓ {dep} version: {torchvision.__version__}")
        elif dep == 'skimage':
            import skimage
            print(f"✓ {dep} version: {skimage.__version__}")
        elif dep == 'numpy':
            import numpy
            print(f"✓ {dep} version: {numpy.__version__}")
        elif dep == 'PIL':
            from PIL import Image
            print(f"✓ {dep} available")
        elif dep == 'timm':
            import timm
            print(f"✓ {dep} version: {timm.__version__}")
        elif dep == 'einops':
            import einops
            print(f"✓ {dep} available")
        elif dep == 'ftfy':
            import ftfy
            print(f"✓ {dep} available")
        elif dep == 'regex':
            import regex
            print(f"✓ {dep} available")
        elif dep == 'tqdm':
            import tqdm
            print(f"✓ {dep} available")
    except ImportError as e:
        print(f"✗ {dep} not available: {e}")
    except Exception as e:
        print(f"? {dep} error: {e}")

# 检查特定功能
print("\n检查特定功能:")
try:
    from cv2.ximgproc import guidedFilter
    print("✓ cv2.ximgproc.guidedFilter available")
except Exception as e:
    print("✗ cv2.ximgproc.guidedFilter not available:", e)

try:
    import torch.nn as nn
    if hasattr(nn, 'GELU'):
        print("✓ torch.nn.GELU available")
    else:
        print("✗ torch.nn.GELU not available (available in PyTorch >= 1.3)")
except Exception as e:
    print("✗ torch.nn.GELU check error:", e)

try:
    from torchvision.transforms import InterpolationMode
    print("✓ torchvision.transforms.InterpolationMode available")
except Exception as e:
    print("✗ torchvision.transforms.InterpolationMode not available (available in torchvision >= 0.8)")

print("\n环境检查完成")