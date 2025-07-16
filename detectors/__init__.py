from .base_detector import BaseDetector

# 总是先尝试导入主要的检测器
print("Loading drift detectors...")

# 1. ADWIN检测器
try:
    from .adwin_detector import ADWINDetector
    ADWIN_AVAILABLE = True
except Exception as e:
    print(f"Warning: ADWIN detector failed to load: {e}")
    print("Falling back to simple ADWIN implementation")
    from .simple_detectors import SimpleADWIN as ADWINDetector
    ADWIN_AVAILABLE = False

# 2. DDM检测器
try:
    from .ddm_detector import DDMDetector
    DDM_AVAILABLE = True
except Exception as e:
    print(f"Warning: DDM detector failed to load: {e}")
    print("Falling back to simple DDM implementation")
    from .simple_detectors import SimpleDDM as DDMDetector
    DDM_AVAILABLE = False

# 3. QuadCDD检测器（可选）
try:
    from .quadcdd_detector import QuadCDDDetector
    QUADCDD_AVAILABLE = True
except Exception as e:
    print(f"Warning: QuadCDD detector failed to load: {e}")
    QuadCDDDetector = None
    QUADCDD_AVAILABLE = False

# 4. 简单检测器（总是可用）
from .simple_detectors import SimpleDriftDetector

# 导出列表
__all__ = ['BaseDetector', 'ADWINDetector', 'DDMDetector', 'SimpleDriftDetector']

if QUADCDD_AVAILABLE:
    __all__.append('QuadCDDDetector')

# 打印检测器可用性状态
print("\n" + "="*50)
print("DETECTOR AVAILABILITY STATUS")
print("="*50)
print(f"  ADWIN: {'✓ Available' if ADWIN_AVAILABLE else '✗ Using simple fallback'}")
print(f"  DDM: {'✓ Available' if DDM_AVAILABLE else '✗ Using simple fallback'}")
print(f"  QuadCDD: {'✓ Available' if QUADCDD_AVAILABLE else '✗ Not available'}")
print(f"  SimpleDriftDetector: ✓ Available")
print("="*50)