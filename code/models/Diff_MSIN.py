import torch
import torch.nn as nn
import torch.nn.functional as F
from basic.layers import EmbeddingLayer, MLP, Expert_net, CrossNetV2
from torch.nn.functional import cosine_similarity

import time

def squash(caps):
    return caps/(caps.norm(dim=-1, keepdim=True)+ 1e-8)

def Max_Min(tensor):
    
    min_scores = tensor.min(dim=0, keepdim=True)[0]
    max_scores = tensor.max(dim=0, keepdim=True)[0]
    normalized_scores = (tensor - min_scores) / (max_scores - min_scores)
    return normalized_scores


class BeToNet(nn.Module):
    def __init__(self, vec_size):
        super(BeToNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(vec_size, 2*vec_size),
            nn.LeakyReLU(),
            nn.Linear(2*vec_size, vec_size),
            nn.LeakyReLU(),
            nn.Linear(vec_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class Diff_MSIN(nn.Module):
    def __init__(self, config, features, history_features, target_features, self_embedding_features, self_embedding_history_features,
                 input_seq_len, input_dim, output_dim, iteration, output_seq_len, 
                 mlp_params, attention_mlp_params):
        super(Diff_MSIN, self).__init__()
        self.iteration = iteration
        self.output_seq_len = output_seq_len
        self.input_seq_len = input_seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        
        self.B_matrix_id = nn.init.normal_(torch.empty(1, output_seq_len, input_seq_len), mean=0, std=1) 
        self.B_matrix_id.requires_grad = False
        self.S_matrix_id = nn.init.normal_(torch.empty(self.input_dim, self.output_dim), mean=0, std=1) 
        self.S_matrix_id = nn.Parameter(self.S_matrix_id)

        self.B_matrix_im = nn.init.normal_(torch.empty(1, output_seq_len, input_seq_len), mean=0, std=1) 
        self.B_matrix_im.requires_grad = False

        self.S_matrix_im = nn.Linear(self.input_dim, self.output_dim)
        
        nn.init.eye_(self.S_matrix_im.weight)

        self.B_matrix_te = nn.init.normal_(torch.empty(1, output_seq_len, input_seq_len), mean=0, std=1) 
        self.B_matrix_te.requires_grad = False

        self.S_matrix_te = nn.Linear(self.input_dim, self.output_dim)

        self.nNeg = 1 
        self.mlp_embedding_dim = output_dim 

        self.rule_n = 1 
        self.rule_len = 3
        self.public_expert_num = 2
        self.multihot_f_num = 1
        self.operator_n = 1 

        self.features = features 
        self.history_features = history_features 
        self.target_features = target_features 
        self.self_embedding_features = self_embedding_features 
        self.self_embedding_history_features = self_embedding_history_features 
        self.num_history_features = len(history_features)
        self.num_self_embedding_history_features = len(self_embedding_history_features)

        self.mlp = MLP(2*output_dim + 100*(self.rule_len+self.public_expert_num)-250, activation="dice", **mlp_params) # 
        self.embedding = EmbeddingLayer(features + history_features + target_features)

        self.text_embedding = nn.Embedding.from_pretrained(config.embedding_pretrained_text, freeze=False).float()
        self.image_embedding = nn.Embedding.from_pretrained(config.embedding_pretrained_image, freeze=False).float()

        self.bn = nn.BatchNorm1d(self.num_self_embedding_history_features)

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

        
        self.attention_layers = nn.ModuleList(
            [ActivationUnit(fea.embed_dim, **attention_mlp_params) for fea in self.history_features] 
            +[ActivationUnit(self.input_dim, **attention_mlp_params) for fea in self.self_embedding_history_features])


        self.tree_layers = [1]
        for l in range(1, self.rule_len):
            self.tree_layers.append(self.tree_layers[-1] * 2)
        self.node_n = sum(self.tree_layers) 

        self.blto_layers = nn.ModuleList([BeToNet(2*output_dim) for i in range(self.rule_len)])

        self.set_presentaions = self.sample_set_presentaions()

        self.Fusion = TensorFusion()

        self._ = [
        nn.Sequential(
            nn.Linear(512, 2*512),
            nn.ReLU(),
            nn.Linear(2*512, 512)
        ).to("cuda:0") for _ in range(1)
        ]

        self.cal_score = [
        nn.Sequential(
            nn.Linear(512, 2*512),
            nn.ReLU(),
            nn.Linear(2*512, 512)
        ).to("cuda:0") for _ in range(1)
        ]

        self.interest_extract_list = [
        nn.Sequential(
            nn.Linear(512, 2*512),
            nn.ReLU(),
            nn.Linear(2*512, 512)
        ).to("cuda:0") for _ in range(self.rule_len)
        ]

        self.public_expert_net = [
        nn.Sequential(
            nn.Linear(512, 2*512),
            nn.ReLU(),
            nn.Linear(2*512, 512)
        ).to("cuda:0") for _ in range(self.public_expert_num)
        ]

        self.modality_fusion = nn.Sequential(
            nn.Linear(2*512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.M = [] 
        self.preference_embeddings = {}
        
    def forward(self, x):
        embed_x_history = self.embedding(x, self.history_features)   
        embed_x_target = self.embedding(x, self.target_features)  

        attention_pooling = []
        for i in range(self.num_history_features):
            attention_seq = self.attention_layers[i](embed_x_history[:, i, :, :], embed_x_target[:, i, :])
            attention_pooling.append(attention_seq.unsqueeze(1))  
        attention_pooling = torch.cat(attention_pooling, dim=1)  

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

        sparse_emb = [] 
        sparse_emb_mlp = [] 
        for col in self.self_embedding_features:
            if 'text' in col.name: 
                text_emb = self.text_embedding(x[col.name]).unsqueeze(1)
                
                sparse_emb.append(text_emb)
                sparse_emb_mlp.append(self.text_embedding_mlp(text_emb.float()))
            elif 'image' in col.name:
                
                image_emb = self.text_embedding(x[col.name]).unsqueeze(1)
                
                sparse_emb.append(image_emb)
                sparse_emb_mlp.append(self.image_embedding_mlp(image_emb))
        concat_self_embedding_features_without_mlp = torch.cat(sparse_emb, dim=1) 

        low_capsule = concat_self_embedding_hist_features_without_mlp[:,0,:,:]  
        embed_x_history = low_capsule  

        B, _, _ = low_capsule.size() 
        assert self.iteration >= 1
        for i in range(self.iteration):
            low_capsule_new = squash(low_capsule)

        low_capsule_image = low_capsule_new

        low_capsule = concat_self_embedding_hist_features_without_mlp[:,1,:,:]  
        embed_x_history = low_capsule  
        B, _, _ = low_capsule.size() 
        assert self.iteration >= 1
        for i in range(self.iteration):
            low_capsule_new = squash(low_capsule)
        
        low_capsule_text = low_capsule_new
        low_capsule = torch.cat([low_capsule_image.unsqueeze(1), low_capsule_text.unsqueeze(1), embed_x_history.unsqueeze(1)],dim=1)
        concat_self_embedding_features_without_mlp = torch.cat([concat_self_embedding_features_without_mlp, embed_x_target], dim=1)
        target_tensor = self.Norm(concat_self_embedding_features_without_mlp.unsqueeze(2))
        '''
        prefer_presentation = target_tensor
        prefer_low_capsule = self.interest_extract_list[0](low_capsule)+low_capsule
        score = self.fuhao[0](prefer_presentation*prefer_low_capsule).sum(dim=-1)
        '''
        scores = []
        prefer_presentations = []
        target_presentations = []
        
        for m in range(self.rule_len): 
            prefer_presentation = target_tensor[:,m]
            prefer_low_capsule = self.interest_extract_list[m](low_capsule[:,m])+low_capsule[:,m]
            prefer_presentations.append(prefer_low_capsule.unsqueeze(1))
            target_presentations.append(prefer_presentation.unsqueeze(1))
            #score = self.fuhao[0](prefer_low_capsule*prefer_presentation).sum(dim=-1)
            #scores.append(score.unsqueeze(1))

        low_capsule_fusion = torch.mean(low_capsule,dim=1).unsqueeze(1)
        target_tensor_fusion = torch.mean(target_tensor,dim=1).unsqueeze(1)

        for m in range(self.public_expert_num): 
            prefer_presentation = target_tensor_fusion#self.public_expert_net[m](target_tensor)+target_tensor
            prefer_low_capsule = self.public_expert_net[m](low_capsule_fusion)+low_capsule_fusion
            prefer_presentations.append(prefer_low_capsule)
            target_presentations.append(prefer_presentation)
            #score = self.fuhao[0](prefer_low_capsule*prefer_presentation).sum(dim=-1)
            #scores.append(score.unsqueeze(1))
        
        prefer_presentations = torch.cat(prefer_presentations,dim=1)
        prefer_presentations = self.cross_net(prefer_presentations)
        target_presentations = torch.cat(target_presentations,dim=1)
        attention_score = self.attention(prefer_presentations, target_presentations)
        score = self.cal_score[0](attention_score*prefer_presentations*target_presentations).sum(dim=-1)

        loss_con = self.calculate_mean_cosine_distance(prefer_presentations)

        mlp_in = torch.cat([
            embed_x_target.flatten(start_dim=1),
            attention_pooling.flatten(start_dim=1),
            score.flatten(start_dim=1),
        ],
                            dim=1)  
        
        y = self.mlp(mlp_in.flatten(start_dim=1))

        return torch.sigmoid(y.squeeze(1)),loss_con,0,0,0

    def attention(self, prefer_presentations, target_presentations):
        target_presentations = target_presentations.expand(-1, -1, prefer_presentations.size(2), -1)

        # 计算点积注意力分数
        attention_scores = torch.matmul(prefer_presentations, target_presentations.transpose(-1, -2))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权的prefer_presentations
        weighted_prefer = torch.matmul(attention_weights, prefer_presentations)
        return weighted_prefer

    def cross_net(self, history_tensor):
        """
        对历史信息张量的第二维进行两两交互。
        
        参数:
            history_tensor (torch.Tensor): 历史信息张量，形状为 [128, 5, 50, 512]。
        
        返回:
            torch.Tensor: 两两交互的结果，形状为 [128, 50, 512]。
        """
        batch_size, num_fields, sequence_length, embedding_size = history_tensor.size()
        
        # 扩展张量以进行两两交互
        expanded_tensor = history_tensor.view(batch_size, num_fields, 1, sequence_length, embedding_size) * history_tensor.view(batch_size, 1, num_fields, sequence_length, embedding_size)
        
        # 求和所有交互的结果
        interaction_sum = 0.1*expanded_tensor.mean(dim=2) + history_tensor
        
        return interaction_sum

    def calculate_mean_cosine_distance(self, tensor):

        mean_tensor = tensor.mean(dim=2)
        num_tensors = mean_tensor.shape[1]

        similarity_matrix = []
        for i in range(num_tensors):
            for j in range(i + 1, num_tensors):
                cosine_sim = F.cosine_similarity(mean_tensor[:, i, :], mean_tensor[:, j, :], dim=1)
                similarity_matrix.append(cosine_sim)

        similarity_matrix = torch.stack(similarity_matrix, dim=1)
        similarity_mean = similarity_matrix.mean(dim=1, keepdim=True)
        return similarity_mean.squeeze(-1).mean()

    def Norm(self,tensor):
        return tensor/(tensor.norm(dim=-1, keepdim=True)+ 1e-8)

    def labelAwareAttation(self, caps, tar, p=2):
        """label-aware attention, input caps and targets, output logits
            caps: (bs, K, D)
            tar: (bs, cnt, D)
            for postive tar, cnt = 1
            for negative tar, cnt = self.nNeg
        """
        tar = tar.transpose(1, 2) 
        w = torch.softmax(
                
                torch.pow(torch.transpose(torch.matmul(caps, tar), 1, 2), p),
                dim=2
            )
        w = w.unsqueeze(2) 

        caps = torch.matmul(w, caps.unsqueeze(1)).squeeze(2)

        return caps

    def attantion(self, ):
        attention_scores = torch.matmul(prefer_presentations, target_presentations.transpose(-1, -2))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 计算加权的target_presentations
        weighted_target = torch.matmul(attention_weights, target_presentations)
        weighted_target

        return 

    def sampledSoftmax(self, caps, tar, bs, tmp=0.01):
        
        tarPos = tar 
        capsPos = self.labelAwareAttation(caps, tarPos).squeeze(1) 

        posLogits = torch.sigmoid(torch.sum(capsPos * tarPos.squeeze(), dim=1) / tmp)

        tarNeg = tarPos[torch.multinomial(torch.ones(bs), self.nNeg * bs, replacement=True)].view(bs, self.nNeg, self.output_dim) 
        capsNeg = self.labelAwareAttation(caps, tarNeg)
        
        negLogits = torch.sigmoid(torch.sum(capsNeg * tarNeg, dim=2).view(bs * self.nNeg) / tmp)

        logits = torch.concat([posLogits, negLogits])
        labels = torch.concat([torch.ones(bs, ), torch.zeros(bs * self.nNeg)])

        return logits, torch.zeros(bs * self.nNeg) 
    
    def sample_set_presentaions(self):

        set_presentaions = torch.empty(self.rule_len, self.output_dim)
 
        vectors = torch.randn(self.rule_len, self.output_dim)
        
        orthogonal_vectors, _ = torch.linalg.qr(vectors.T)
        
        orthogonal_vectors = orthogonal_vectors.T
        
        finished_set = []
        while len(finished_set) != self.rule_len:
            sample_set_presentaion = torch.randn(self.output_dim)
            r_list = []
            for rl in range(self.rule_len):
                r = self.blto_layers[rl](torch.cat([sample_set_presentaion, orthogonal_vectors[rl]]))
                r_list.append(r)
            
            count = sum(1 for r in r_list if r > 0.5)
            
            if count == 1:
                index = r_list.index(next(r for r in r_list if r > 0.5))
                set_presentaions[index] = sample_set_presentaion
                if index not in finished_set:
                    finished_set.append(index)

        return set_presentaions.to("cuda")
        
    def calculate_preferences_embeddings(self, embeddings, scores):
        
        num_preferences = scores.size(1)
        emb_dim = embeddings.size(-1)
        
        
        num_combinations = 2 ** num_preferences
        combinations = [format(i, f'0{num_preferences}b') for i in range(1, num_combinations)]
        
        
        preference_embeddings = {combination: torch.ones(1, emb_dim, device=embeddings.device) for combination in combinations}
        
        for combination in combinations:
            
            mask = torch.tensor([float(bit) for bit in combination], device=embeddings.device)
            
            matching = (scores == mask).all(dim=-1, keepdim=True)
            
            weighted_embeddings = matching.float() * embeddings
            sum_embeddings = weighted_embeddings.sum(dim=0, keepdim=True)
            count = matching.sum()
            
            if count > 0:
                preference_embeddings[combination] = sum_embeddings / count
        preference_embeddings = {int(key, 2): value for key, value in preference_embeddings.items()}

        return preference_embeddings
    
    def union_theorem_loss(self, preference_embeddings, target_embedding):
        loss = 1.0
        count = 0
        m = 0
        for parent_key in preference_embeddings.keys():
            loss_tmp = 1
            parent_embedding = preference_embeddings[parent_key].unsqueeze(0)
            a = self.fuhao[m](parent_embedding*target_embedding)
            a = Max_Min(a).mean()
            for child_key in preference_embeddings.keys():
                if parent_key != child_key and child_key & parent_key == child_key:
                    
                    child_embedding = preference_embeddings[child_key].unsqueeze(0)
                    child = self.fuhao[m](child_embedding*target_embedding)
                    child = Max_Min(child).mean()
                    loss_tmp *= (1-child)
                    count += 1
            m+=1
            if loss_tmp != 1:
                loss += a / ((1-loss_tmp))
                
        loss = torch.log(loss)
        return loss / count if count > 0 else loss

    def complement_theorem_loss(self, preference_embeddings, target_embedding):
        loss = 0.0
        count = 0
        m = 0
        for key1 in preference_embeddings.keys():
            embedding1 = preference_embeddings[key1].unsqueeze(0)
            a = self.fuhao[m](embedding1*target_embedding)
            a = Max_Min(a).mean()
            for key2 in preference_embeddings.keys():
                if key1 != key2 and not (key1 & key2):
                    embedding2 = preference_embeddings[key2].unsqueeze(0)
                    
                    score = self.fuhao[m](embedding2*target_embedding)
                    score = Max_Min(score).mean()
                    
                    loss += a/((1 - score))  
                    count += 1
            m+=1
            
        loss = torch.log(loss)
        return loss / count if count > 0 else loss
    
class TensorFusion(nn.Module):
    def __init__(self, threshold=0.5):
        super(TensorFusion, self).__init__()
        self.threshold = threshold

    def classify_and_fuse_for_same_modality(self, r_tensor, v_tensor):
        
        mask_belong = r_tensor >= 0.5

        
        interest_by_category = v_tensor.unsqueeze(2) * mask_belong.to("cuda").unsqueeze(-1)
        interest_category_mean = interest_by_category.mean(dim=1)

        
        mask_all_below_threshold = torch.logical_not(mask_belong).all(dim=2)
        interest_special_mean = v_tensor*mask_all_below_threshold.to("cuda").unsqueeze(-1)
        interest_special_mean = interest_special_mean.mean(dim=1)

        v_fusion = torch.cat([interest_category_mean.to("cuda"), interest_special_mean.unsqueeze(1).to("cuda")],dim=1)

        
        v_fusion = torch.where(torch.isnan(v_fusion), torch.tensor(0.0).to("cuda"), v_fusion)

        return v_fusion

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
            output = (att_weight.unsqueeze(-1) * history).sum(dim=1)
        else:
            cosSim = cosSim.softmax(dim=-1)
            output = ((cosSim*att_weight).unsqueeze(-1) * history).sum(dim=1)
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