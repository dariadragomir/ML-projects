# SCORE_THRESH = {"fred" : 0.99, "daphne" : 0.9, "shaggy" : 0.87, "velma": 0.85, "unknown": 0.7} 81.5
SCORE_THRESH = {"fred" : 0.65, "daphne" : 0.9, "shaggy" : 0.87, "velma": 0.85, "unknown": 0.5} # 84

SCALES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
NUM_IMG = 115000
TRAIN_IMG = 1000
FOLDERS = ["fred", "daphne", "shaggy", "velma"]
SCOOBY_CLASSES = ["fred", "daphne", "shaggy", "velma", "unknown"]
LABELS = ["fred", "daphne", "shaggy", "velma", "unknown"]
WINDOW_SIZES = {
    "fred": (66, 48),
    "daphne": (50, 41), 
    "shaggy": (62, 39), 
    "velma": (50, 50), 
    "unknown": (41, 28) 
}
STRIDE = 16