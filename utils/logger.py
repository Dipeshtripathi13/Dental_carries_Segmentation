from torch.utils.tensorboard import SummaryWriter
import logging
import os

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.setup_logging(log_dir)
    
    def setup_logging(self, log_dir):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def log_metrics(self, metrics, step, phase='train'):
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{metric_name}', value, step)
            logging.info(f'{phase.capitalize()} - Step {step} - {metric_name}: {value:.4f}')

