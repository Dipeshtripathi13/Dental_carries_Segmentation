class Config:
    # Data
    IMG_HEIGHT = 256
    IMG_WIDTH = 512
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    
    # Model
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['background', 'teeth', 'caries']
    ACTIVATION = 'softmax'
    
    # Training
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda'
    MIXED_PRECISION = True
    
    # Augmentation
    AUG_PROB = 0.5
    ROTATE_LIMIT = 15
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    
    # Paths
    TRAIN_DIR = 'dataset/train'
    VAL_DIR = 'dataset/val'
    TEST_DIR = 'dataset/test'
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
