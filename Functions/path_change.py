from pathlib import Path

# קבלת הנתיב של הסקריפט הנוכחי
current_dir = Path(_file_).parent

# הגדרת נתיבים יחסיים
MODELS_PATH = current_dir / 'Models'
TRAINING_PATH = current_dir / 'Training'
INFERENCE_PATH = current_dir / 'Inference'

# הוספת הספריות ל-sys.path
sys.path.insert(0, str(MODELS_PATH))
sys.path.insert(0, str(TRAINING_PATH))
sys.path.insert(0, str(INFERENCE_PATH))