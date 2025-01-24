
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    
    def __init__(self, dataset):

        self.train_data = dataset + 'train_.pkl'
        self.embedding_pretrained_text = torch.load(dataset+'text_embedding.pt',map_location="cuda:0").squeeze(1)
        self.embedding_pretrained_image = torch.load(dataset + 'image_embedding.pt', map_location="cuda:0").squeeze(1)
        '''
        self.embedding_pretrained_text = torch.tensor(
            np.load(dataset + 'text_embedding.pt').astype('float32'))              
        self.embedding_pretrained_image = torch.tensor(
            np.load(dataset + 'image_embedding.pt').astype('float32'))             
        '''
        self.embedding_dims = 512                                                   

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   