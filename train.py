from config import read_config 
from models.trainer import CDTrainer


cfg = read_config(train=True) # get all variables from config file

Trainer = CDTrainer(args=cfg)
Trainer.train_model()