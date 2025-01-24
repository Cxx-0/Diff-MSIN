'''
 0.7649436308929903
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.layers import EmbeddingLayer, MLP, Expert_net

class DIN_cosSim(nn.Module):
    """Deep Interest Network with cosine_similarity of text/image embedding
    Args:
        features (list): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
        history_features (list): the list of `Feature Class`,training by ActivationUnit. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
        target_features (list): the list of `Feature Class`, training by ActivationUnit. It means the target feature which will execute target-attention with history feature.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        attention_mlp_params (dict): the params of the ActivationUnit module, keys include:`{"dims":list, "activation":str, "dropout":float, "use_softmax":bool`}
        self_embedding_features (list): training by MLP. 用户自己embedding的数据所在列
        self_embedding_history_features (list): training by ActivationUnit. 用户自己embedding数据的历史行为序列，将会计算与self_embedding_features的attention
    """

    def __init__(self, config, features, history_features, target_features, 
                 mlp_params, attention_mlp_params, self_embedding_features = [], self_embedding_history_features=[]):
        super().__init__()
        self.features = features 
        self.history_features = history_features 
        self.target_features = target_features 
        self.self_embedding_features = self_embedding_features 
        self.self_embedding_history_features = self_embedding_history_features 
        self.num_history_features = len(history_features)
        self.num_self_embedding_history_features = len(self_embedding_history_features)

        self.mlp_embedding_dim = 64 
        self.expert_dim = 64
        
        
        self.features_dim = sum([fea.embed_dim for fea in features + history_features + target_features]) 
        self.self_embedding_dim = self.mlp_embedding_dim*len(self_embedding_features + self_embedding_history_features)
        self.cos_dims = 1*(self.num_self_embedding_history_features+self.num_history_features) 
        
        self.all_dims = self.features_dim + 3*self.expert_dim + self.self_embedding_dim 
                                       
        self.embedding = EmbeddingLayer(features + history_features + target_features)

        
        self.text_embedding = nn.Embedding.from_pretrained(config.embedding_pretrained_text, freeze=True)
        self.image_embedding = nn.Embedding.from_pretrained(config.embedding_pretrained_image, freeze=True)

        self.num_heads = 1
        
        self.attention_layers = nn.ModuleList(
            [ActivationUnit(fea.embed_dim, **attention_mlp_params) for fea in self.history_features]
            +[MultiheadAttention(self.mlp_embedding_dim, self.num_heads) for fea in self.self_embedding_history_features])

        
        self.mlp = MLP(self.features_dim + 3*self.expert_dim, activation="dice", **mlp_params)

        output_dim = self.mlp.mlp[-1].out_features 

        self.bn = nn.BatchNorm1d(self.num_self_embedding_history_features)
        self.bn_ID = nn.BatchNorm1d(self.num_history_features)
        '''
        self.cos_mlp = nn.Sequential(
            nn.Linear(self.cos_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        '''
                 
        self.text_embedding_mlp = nn.Sequential(
            nn.Linear(512, 2*self.mlp_embedding_dim),
            nn.ReLU(),
            nn.Linear(2*self.mlp_embedding_dim, self.mlp_embedding_dim)
        )
        self.image_embedding_mlp = nn.Sequential(
            nn.Linear(512, 2*self.mlp_embedding_dim),
            nn.ReLU(),
            nn.Linear(2*self.mlp_embedding_dim, self.mlp_embedding_dim)
        )

        
        self.mlp_share = Expert_net(self.features_dim, activation="dice", **mlp_params)
        self.text_mlp = Expert_net(int(self.self_embedding_dim/2), activation="dice", **mlp_params)
        self.image_mlp = Expert_net(int(self.self_embedding_dim/2), activation="dice", **mlp_params)

        self.image_weight = Expert_net(self.expert_dim, activation="dice", **mlp_params)
        self.text_weight = Expert_net(self.expert_dim, activation="dice", **mlp_params)
        self.share_weight = Expert_net(self.expert_dim, activation="dice", **mlp_params)

        
        

        
        self.n_share = 1 
        self.n_task = 2
        device = torch.device("cuda:0")
    
        for i in range(self.n_share):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(nn.Linear(self.expert_dim, 2+1), 
                                        					   nn.Softmax(dim=1)).to(device)) 
        self.Shared_Gates = [getattr(self,"gate_layer"+str(i+1)) for i in range(self.n_share)]

        
        for i in range(self.n_task):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(nn.Linear(self.expert_dim, 1+1),
                                        					   nn.Softmax(dim=1)))
        self.Task_Gates = [getattr(self,"gate_layer"+str(i+1)) for i in range(self.n_task)]

        
        '''ID_Gate网络结构
        for i in range(1):
            setattr(self, "weight_layer"+str(i+1), nn.Sequential(nn.Linear(self.fet_dim, 14), 
                                        					   nn.Softmax(dim=1)))
        self.ID_Gates = [getattr(self,"weight_layer"+str(i+1)) for i in range(1)]
        '''

        self.fet_dim = sum([fea.embed_dim for fea in features+target_features])
        self.n_mode = 3
        
        for i in range(self.n_mode):
            setattr(self, "weight_layer"+str(i+1), nn.Sequential(nn.Linear(self.fet_dim, 1), 
                                                            nn.Softmax(dim=1)).to(device))
        self.Mode_Gates = [getattr(self,"weight_layer"+str(i+1)) for i in range(self.n_mode)]
        
        hidden_layer2 = [64,16]
        self.cosSim_fusion_func = nn.Sequential(
            nn.Linear(self.num_history_features, hidden_layer2[0]), 
            nn.ReLU(),
            nn.Linear(hidden_layer2[0], self.num_history_features)) 
        
        
        
        self.ID_cosSim_fusion_func = nn.Sequential(
            nn.Linear(self.num_history_features, hidden_layer2[0]), 
            nn.ReLU(),
            nn.Linear(hidden_layer2[0], self.num_history_features)) 
        
        self.IT_cosSim_fusion_func = nn.Sequential(
            nn.Linear(self.num_self_embedding_history_features, hidden_layer2[0]), 
            nn.ReLU(),
            nn.Linear(hidden_layer2[0], 1)) 

        self.ID_Sim_attention = MultiheadAttention(self.num_history_features, self.num_heads)
        self.IT_Sim_attention = MultiheadAttention(1, self.num_heads)

        self.ID_cos_mlp = nn.Sequential(
            nn.Linear(self.num_history_features, hidden_layer2[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer2[0], 1)
        )

        self.IT_cos_mlp = nn.Sequential(
            nn.Linear(self.num_self_embedding_history_features, hidden_layer2[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer2[0], 1)
        )  

        self.tower_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
    def forward(self, x):

        
        embed_x_features = self.embedding(x, self.features)  
        embed_x_history = self.embedding(x, self.history_features)  
        embed_x_target = self.embedding(x, self.target_features)  

        
        sparse_emb = [] 
        sparse_emb_mlp = [] 
        for col in self.self_embedding_features:
            if 'text' in col.name: 
                sparse_emb.append(x[col.name])
                sparse_emb_mlp.append(self.text_embedding_mlp(x[col.name]))
            elif 'image' in col.name:
                sparse_emb.append(x[col.name])
                sparse_emb_mlp.append(self.image_embedding_mlp(x[col.name]))
        concat_self_embedding_features_without_mlp = torch.cat(sparse_emb, dim=1) 
        concat_self_embedding_features = torch.cat(sparse_emb_mlp, dim=1)

        
        sequence_emb = []
        sequence_emb_mlp = []
        for col in self.self_embedding_history_features:
            if 'text' in col.name:
                batch_size = x[col.name].shape[0]  
                text_seq_len = x[col.name].shape[1]  
                text_emb_dim = col.embed_dim  
                self_text_embedding = self.text_embedding(x[col.name]) 
                sequence_emb.append(self_text_embedding.unsqueeze(1))
                self_text_embedding = self.text_embedding_mlp(self_text_embedding.view(-1, text_emb_dim))
                self_text_embedding = self_text_embedding.view(batch_size, text_seq_len, -1)
                sequence_emb_mlp.append(self_text_embedding.unsqueeze(1))
            elif 'image' in col.name:
                batch_size = x[col.name].shape[0]  
                image_seq_len = x[col.name].shape[1]  
                image_emb_dim = col.embed_dim  
                self_image_embedding = self.image_embedding(x[col.name]) 
                sequence_emb.append(self_image_embedding.unsqueeze(1))
                self_image_embedding = self.image_embedding_mlp(self_image_embedding.view(-1, image_emb_dim))
                self_image_embedding = self_image_embedding.view(batch_size, image_seq_len, -1)
                sequence_emb_mlp.append(self_image_embedding.unsqueeze(1))  
        
        
        concat_self_embedding_hist_features_without_mlp = torch.cat(sequence_emb, dim=1)
        concat_self_embedding_hist_features = torch.cat(sequence_emb_mlp, dim=1)

        
        tensor1 = concat_self_embedding_features_without_mlp.unsqueeze(2)
        tensor2 = concat_self_embedding_hist_features_without_mlp
        
        cosSim = F.cosine_similarity(tensor1, tensor2, dim=-1) 
        

        cosSim_ID = F.cosine_similarity(embed_x_target[:,:-1].unsqueeze(2), embed_x_history, dim=-1) 
        

        
        attention_pooling = []
        
        for i in range(self.num_history_features):
            attention_seq = self.attention_layers[i](embed_x_history[:, i, :, :], embed_x_target[:, i, :])
            attention_pooling.append(attention_seq.unsqueeze(1))  
        attention_pooling = torch.cat(attention_pooling, dim=1)  
        
        
        
        for i in range(self.num_history_features):
            
            attention_pooling[:,i,:,:] = ((cosSim[:,1,:]+cosSim[:,0,:])/2).unsqueeze(2) * attention_pooling[:,i,:,:].clone() 
        attention_pooling = attention_pooling.sum(dim=2) 
        
        
        attention_pooling_self = []
        
        for i in range(self.num_history_features, self.num_history_features+self.num_self_embedding_history_features):
            attention_seq = self.attention_layers[i](concat_self_embedding_features[:, i-self.num_history_features, :].unsqueeze(1).repeat(1, concat_self_embedding_hist_features.size(2), 1),
                                                     concat_self_embedding_hist_features[:, i-self.num_history_features, :, :],
                                                     concat_self_embedding_hist_features[:, i-self.num_history_features, :, :]) 
            attention_pooling_self.append(attention_seq.unsqueeze(1))  
        attention_pooling_self = torch.cat(attention_pooling_self, dim=1)  
        attention_pooling_self = attention_pooling_self.sum(dim=2)

        mlp_share_in = torch.cat([
            attention_pooling.flatten(start_dim=1),
            embed_x_target.flatten(start_dim=1),
            embed_x_features.flatten(start_dim=1),
            
            
            
        ],
                            dim=1)  
        
        mlp_in_text = torch.cat([
            concat_self_embedding_features[:,0,:].flatten(start_dim=1), 
            attention_pooling_self[:,0,:].flatten(start_dim=1)
            
        ],
                            dim=1)  
        
        mlp_in_image = torch.cat([
            concat_self_embedding_features[:,1,:].flatten(start_dim=1), 
            attention_pooling_self[:,1,:].flatten(start_dim=1)
            
        ],
                            dim=1)  

        Share_Out = self.mlp_share(mlp_share_in)
        Expert_A_Out = self.text_mlp(mlp_in_text)
        Expert_B_Out = self.image_mlp(mlp_in_image)
        
        
        Gate_A = self.Task_Gates[0](Expert_A_Out)     
        
        Gate_Shared = self.Shared_Gates[0](Share_Out)     
        
        Gate_B = self.Task_Gates[1](Expert_B_Out)     

        Share_Out = Share_Out.unsqueeze(1)
        Expert_A_Out = Expert_A_Out.unsqueeze(1)
        Expert_B_Out = Expert_B_Out.unsqueeze(1)

        
        g = Gate_A.unsqueeze(2)  
        experts = torch.cat([Expert_A_Out,Share_Out],dim=1) 
        
        Gate_A_Out = torch.matmul(experts.transpose(1,2),g)
        Gate_A_Out = Gate_A_Out.squeeze(2)
        
        g = Gate_Shared.unsqueeze(2)  
        experts = torch.cat([Expert_A_Out,Share_Out,Expert_B_Out],dim=1) 
        
        Gate_Shared_Out = torch.matmul(experts.transpose(1,2),g)
        Gate_Shared_Out = Gate_Shared_Out.squeeze(2)
        
        g = Gate_B.unsqueeze(2)  
        experts = torch.cat([Expert_B_Out,Share_Out],dim=1) 
        
        Gate_B_Out = torch.matmul(experts.transpose(1,2),g)
        Gate_B_Out = Gate_B_Out.squeeze(2)

        '''
        Gate_ID = self.ID_Gates[0](embed_x_features.flatten(start_dim=1))    

        g = Gate_ID.unsqueeze(2)  
        experts = torch.cat([attention_pooling,embed_x_target,embed_x_features],dim=1) 
        
        Gate_ID_Out = torch.matmul(experts.transpose(1,2),g)
        Gate_ID_Out = Gate_ID_Out.squeeze(2)
        '''
        
        putin = torch.cat([embed_x_features,embed_x_target], dim=1)
        Gate_Text = self.Mode_Gates[0](putin.flatten(start_dim=1))
        Gate_Image = self.Mode_Gates[1](putin.flatten(start_dim=1))

        
        g = Gate_Text.unsqueeze(2)  
        experts = Gate_A_Out.unsqueeze(1) 
        Gate_Text_Out = torch.matmul(experts.transpose(1,2),g)
        Gate_Text_Out = Gate_Text_Out.squeeze(2)

        
        g = Gate_Image.unsqueeze(2)  
        experts = Gate_B_Out.unsqueeze(1) 
        Gate_Image_Out = torch.matmul(experts.transpose(1,2),g)
        Gate_Image_Out = Gate_Image_Out.squeeze(2)

        
        mlp_in = torch.cat([
            attention_pooling.flatten(start_dim=1),
            embed_x_target.flatten(start_dim=1),
            embed_x_features.flatten(start_dim=1),
            Gate_Image_Out+Gate_B_Out+concat_self_embedding_features[:,1,:]+attention_pooling_self[:,1,:].flatten(start_dim=1),
            Gate_Text_Out+Gate_A_Out+concat_self_embedding_features[:,0,:]+attention_pooling_self[:,0,:].flatten(start_dim=1),
            
            
            Gate_Shared_Out.flatten(start_dim=1),
            
            
            
        ],
                            dim=1)  

        Out = self.mlp(mlp_in)

        
        cosSim = self.bn(cosSim)
        cosSim = cosSim.sum(dim=-1)
        cosSim_ID = self.bn_ID(cosSim_ID)
        cosSim_ID = cosSim_ID.sum(dim=-1)

        '''
        cosSim_fusion = torch.cat([cosSim, cosSim_ID],dim=1)
        cosSim_fusion = self.cosSim_fusion_func(cosSim_ID)
        cosSim_fusion = cosSim_fusion.unsqueeze(1)
        cosSim_ID_att = self.Sim_attention(cosSim_ID.unsqueeze(1), cosSim_fusion, cosSim_fusion).squeeze(1)
        final_sim = torch.cat([cosSim_ID_att, cosSim],dim=1)
        y2 = self.cos_mlp(final_sim.flatten(start_dim=1))
        '''
        ID_cosSim_fusion = self.ID_cosSim_fusion_func(cosSim_ID)
        ID_cosSim_fusion = ID_cosSim_fusion.unsqueeze(1)
        ID_cosSim_att = self.ID_Sim_attention(cosSim_ID[:,0].unsqueeze(-1).unsqueeze(-1).expand(ID_cosSim_fusion.shape), ID_cosSim_fusion, ID_cosSim_fusion).squeeze(1)
        y_ID_Sim = self.ID_cos_mlp(ID_cosSim_att.flatten(start_dim=1))

        IT_cosSim_fusion = self.IT_cosSim_fusion_func(cosSim)
        IT_cosSim_fusion = IT_cosSim_fusion.unsqueeze(1)
        I_cosSim_att = self.IT_Sim_attention(cosSim[:,1].unsqueeze(-1).unsqueeze(-1).expand(IT_cosSim_fusion.shape), IT_cosSim_fusion, IT_cosSim_fusion).squeeze(1)
        T_cosSim_att = self.IT_Sim_attention(cosSim[:,0].unsqueeze(-1).unsqueeze(-1).expand(IT_cosSim_fusion.shape), IT_cosSim_fusion, IT_cosSim_fusion).squeeze(1)
        IT_cosSim_att = torch.cat([I_cosSim_att+cosSim[:,1].unsqueeze(-1), T_cosSim_att+cosSim[:,0].unsqueeze(-1)], dim=1)
        y_IT_Sim = self.IT_cos_mlp(IT_cosSim_att.flatten(start_dim=1))
        

        Out_final = torch.cat([Out, y_IT_Sim], dim=1)
        y = self.tower_net(Out_final)

        
        
        return torch.sigmoid(y.squeeze(1))

class ActivationUnit(nn.Module):
    """Activation Unit Layer mentioned in DIN paper, it is a Target Attention method.

    Args:
        embed_dim (int): the length of embedding vector.
        history (tensor):
    Shape:
        - Input: `(batch_size, seq_length, emb_dim)`
        - Output: `(batch_size, emb_dim)`
    """

    def __init__(self, emb_dim, dims=None, activation="dice", use_softmax=False):
        super(ActivationUnit, self).__init__()
        if dims is None:
            dims = [36]
        self.emb_dim = emb_dim
        self.use_softmax = use_softmax
        self.attention = MLP(4 * self.emb_dim, dims=dims, activation=activation)

    def forward(self, history, target, cosSim=None):
        seq_length = history.size(1)
        target = target.unsqueeze(1).expand(-1, seq_length, -1)  
        att_input = torch.cat([target, history, target - history, target * history],
                              dim=-1)  
        att_weight = self.attention(att_input.view(-1, 4 * self.emb_dim))  
        att_weight = att_weight.view(-1, seq_length)  
        if self.use_softmax:
            att_weight = att_weight.softmax(dim=-1)
        
        
        if cosSim == None:
            output = (att_weight.unsqueeze(-1) * history)
            
        else:
            
            cosSim = cosSim.softmax(dim=-1)
            output = ((cosSim*att_weight).unsqueeze(-1) * history)
        return output

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)

        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(attended_values)

        return output