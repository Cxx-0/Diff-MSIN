import numpy as np
import pandas as pd
import torch
from models import Diff_MSIN
from trainers import CTRTrainer
from basic.features import DenseFeature, SparseFeature, SequenceFeature
from utils.data import DataGenerator
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
from importlib import import_module

def df_to_dict(data):

    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict

def get_data_dict(data_path):

    train = pd.read_pickle(data_path)
    print("data load finished")

    user_id = 'reviewerID'

    n_users, n_items = train[user_id].max(), train["target_item_id"].max()

    
    features = [SparseFeature("target_item_id", vocab_size=n_items + 2, embed_dim=512), 
                ]
    target_features = features  

    history_features = [
        SequenceFeature("hist_item_id", vocab_size=n_items + 2, embed_dim=512, pooling="concat", shared_with="target_item_id"),
        ]
    
    self_embedding_features = [SparseFeature("target_image", vocab_size=-1, embed_dim=512), 
                            SparseFeature("target_text", vocab_size=-1, embed_dim=512)]
    target_embedding_features = self_embedding_features
    self_embedding_history_features = [SparseFeature("hist_item_id_text", vocab_size=-1, embed_dim=512), 
                            SparseFeature("hist_item_id_image", vocab_size=-1, embed_dim=512)]

    
    train, test = train_test_split(train, test_size=0.2, random_state=42)

    train = df_to_dict(train)
    test = df_to_dict(test)

    train_y, test_y = train["label"], test["label"]

    del train["label"]
    
    del test["label"]
    
    train_x, test_x = train, test

    return features, target_features, history_features, self_embedding_features, target_embedding_features, self_embedding_history_features, train_x, train_y, test_x, test_y


def main(config, dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, continue_train):
    torch.manual_seed(seed)
    features, target_features, history_features, self_embedding_features, target_embedding_features, self_embedding_history_features, train_x, train_y, test_x, test_y = get_data_dict(dataset_path)
    dg = DataGenerator(train_x, train_y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_test=test_x, y_test=test_y, batch_size=batch_size)
    if model_name == "Diff-MSIN":
        model = Diff_MSIN(config, features=features, history_features=history_features, target_features=target_features, self_embedding_features=self_embedding_features, self_embedding_history_features=self_embedding_history_features,
                     input_seq_len=50,input_dim=512,output_dim=512,iteration=1,output_seq_len=3, mlp_params={"dims": [256, 128]},attention_mlp_params={"dims": [256, 128]},
                     )
        if continue_train:
            checkpoint = torch.load(save_dir + '/model.pth')
            model.load_state_dict(checkpoint)
    
    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=10, device=device, model_path=save_dir)
    
    ctr_trainer.fit(train_dataloader, test_dataloader) 
    
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="../data/Ama/arts/") 
    parser.add_argument('--model_name', default='Diff-MSIN') 
    parser.add_argument('--epoch', type=int, default=15)  
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)  
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')  
    parser.add_argument('--save_dir', default='./model_result')
    parser.add_argument('--continue_train', default=False) 
    parser.add_argument('--seed', type=int, default=42) 

    args = parser.parse_args()
    config = Config(args.dataset_path)
    main(config, config.train_data, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed, args.continue_train)
