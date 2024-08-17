from yacs.config import CfgNode as CN

def get_cfg_defaults():
    _C = CN()

    # Major Hyperparameters for fine-tuning train
    _C.TRAIN = CN()
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.BASE_LR = 0.004
    _C.TRAIN.WEIGHT_DECAY = 0.005
    _C.TRAIN.NUM_EPOCHS = 20
    _C.TRAIN.EPOCHS = 10    
    _C.TRAIN.DATA_ROOT = r"/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR/TD-TSR"
    _C.TRAIN.PRETRAINED_WEIGHTS = ""
    _C.TRAIN.RESULTS_CSV = "/kaggle/working/dlcv24-group-final-project-nicepeople/results"
    _C.TRAIN.PREDICTIONS = "/kaggle/working/dlcv24-group-final-project-nicepeople/predictions"
    _C.TRAIN.SAVE_MODEL_PATH = "/kaggle/working/dlcv24-group-final-project-nicepeople/models"
    _C.TRAIN.RUN_NAME = "fasterrcnn"
    _C.TRAIN.NO_CUDA = False
    _C.TRAIN.ANNOTATIONS_PATH = "/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR/TD-TSR/annotations/train.json"

    # Warmup Options
    _C.TRAIN.WARMUP_LR = 0.0004
    _C.TRAIN.WARMUP_EPOCHS = 1

    # Scheduling Options
    _C.TRAIN.PCT_START = 0.3
    _C.TRAIN.ANNEAL_STRATEGY = 'cos'

    # Train Data Augmentation Techniques
    _C.TRAIN.IMAGE_HEIGHT = 3509
    _C.TRAIN.IMAGE_WIDTH = 2480
    _C.TRAIN.HORIZONTAL_FLIP_PROB = 0
    _C.TRAIN.ROTATION_DEGREES = 0

    # Customizing evaluation on test data
    _C.TEST = CN()
    _C.TEST.SCORE_THRESHOLD=0
    _C.TEST.IOU_THRESHOLD=1
    _C.TEST.BATCH_SIZE = 4
    _C.TEST.ANNOTATIONS_PATH = "/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR/TD-TSR/annotations/test.json"
    _C.TEST.IMAGE_HEIGHT = 3509
    _C.TEST.IMAGE_WIDTH = 2480

    # Model Customization
    _C.MODEL = CN()
    _C.MODEL.NUM_CLASSES = 6
    _C.MODEL.TYPE = "fasterrcnn"  # 3 models available fasterrcnn, fcos and retinanet, with backbone resnet50
    _C.MODEL.BACKBONE = "resnet50"  # Available backbones: resnet18, resnet34, resnet50, resnet101

    # Ensemble Configuration
    _C.ENSEMBLE = CN()
    _C.ENSEMBLE.SCORE_THRESHOLD = 0
    _C.ENSEMBLE.IOU_THRESHOLD = 0.7
    _C.ENSEMBLE.USE_WBF = True
    _C.ENSEMBLE.FILENAMES = ['fasterrcnn.json', 'retina.json', 'fcos.json', 'fasterrcnn1.json', 'fcos1.json']
    _C.ENSEMBLE.WEIGHTS = [2, 1, 1, 2, 1]

    return _C
