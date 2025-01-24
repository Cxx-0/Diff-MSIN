import random
import torch
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split


class TorchDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class PredictDataset(Dataset):

    def __init__(self, x):
        super().__init__()
        self.x = x

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])


class MatchDataGenerator(object):

    def __init__(self, x, y=[]):
        super().__init__()
        if len(y) != 0:
            self.dataset = TorchDataset(x, y)
        else:  
            self.dataset = PredictDataset(x)

    def generate_dataloader(self, x_test_user, x_all_item, batch_size, num_workers=8):
        train_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = PredictDataset(x_test_user)

        
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        item_dataset = PredictDataset(x_all_item)
        item_dataloader = DataLoader(item_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, test_dataloader, item_dataloader


class DataGenerator(object):

    def __init__(self, x, y):
        super().__init__()
        self.dataset = TorchDataset(x, y)
        self.length = len(self.dataset)

    def generate_dataloader(self, x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16,
                            num_workers=8):
        if split_ratio != None:
            train_length = int(self.length * split_ratio[0])
            val_length = int(self.length * split_ratio[1])
            test_length = self.length - train_length - val_length
            print("the samples of train : val : test are  %d : %d : %d" % (train_length, val_length, test_length))
            train_dataset, val_dataset, test_dataset = random_split(self.dataset,
                                                                    (train_length, val_length, test_length))
        else:
            train_dataset = self.dataset
            val_dataset = TorchDataset(x_val, y_val)
            test_dataset = TorchDataset(x_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader


def get_auto_embedding_dim(num_classes):
    """ Calculate the dim of embedding vector according to number of classes in the category
    emb_dim = [6 * (num_classes)^(1/4)]
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    Args:
        num_classes: number of classes in the category
    
    Returns:
        the dim of embedding vector
    """
    return np.floor(6 * np.pow(num_classes, 0.26))


def get_loss_func(task_type="classification"):
    if task_type == "classification":
        return torch.nn.BCELoss()
    elif task_type == "regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError("task_type must be classification or regression")


def get_metric_func(task_type="classification"):
    if task_type == "classification":
        return roc_auc_score
    elif task_type == "regression":
        return mean_squared_error
    else:
        raise ValueError("task_type must be classification or regression")

def generate_seq_feature(data,
                         user_col,
                         item_col,
                         time_col,
                         label_col,
                         label,
                         item_attribute_cols=[],
                         not_sequence_cols=[],
                         min_item=0,
                         shuffle=True,
                         max_len=50):
    """generate sequence feature and negative sample for ranking.

    Args:
        data (pd.DataFrame): the raw data.
        user_col (str): the col name of user_id
        item_col (str): the col name of item_id
        time_col (str): the col name of timestamp
        label_col (str): the col name of label
        label (list[str]): 
        item_attribute_cols (list[str], optional): the other attribute cols of item which you want to generate sequence feature. Defaults to `[]`.
        not_sequence_cols (list[str], optional): 
        sample_method (int, optional): the negative sample method `{
            0: "random sampling",
            1: "popularity sampling method used in word2vec",
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`.
            Defaults to 0.
        min_item (int, optional): the min item each user must have. Defaults to 0.
        shuffle (bool, optional): shulle if True
        max_len (int, optional): the max length of a user history sequence.

    Returns:
        pd.DataFrame: split train with sequence features by time and decode of label.
    """

    action_to_index = {string: index for index, string in enumerate(set(label))}
    label = [action_to_index[string] for string in label]

    for feat in data.loc[:, data.columns != label_col]:
        if feat != time_col and type(data[feat].iloc[0]) == str:
            le = LabelEncoder()
            data[feat] = le.fit_transform(data[feat])
            data[feat] = data[feat].apply(lambda x: x + 1)  
    data[label_col] = data[label_col].apply(lambda x: action_to_index[x] if x in action_to_index else len(label))

    
    n_items = data[item_col].max()
    item2attr = {}
    if len(item_attribute_cols) > 0:
        for col in item_attribute_cols:
            map = data[[item_col, col]]
            item2attr[col] = map.set_index([item_col])[col].to_dict()
    
    train_data = []
    data.sort_values(time_col, inplace=True)
    for uid, hist in tqdm.tqdm(data.groupby(user_col), desc='generate sequence features'): 

        

        pos_list = hist[item_col].tolist() 
        label_list = hist[label_col].tolist() 
        time_list = hist[time_col].tolist() 
        len_pos_list = len(pos_list)
        if len_pos_list < min_item:  
            continue
        
        episode = len_pos_list
        start_episode = 0 
        for i in range(1, episode):
            
            if (i-1) % max_len == 0:
                start_episode = i-1

            pos_label = label_list[i] 
            pos_time = time_list[i] 

            hist_item = pos_list[start_episode:i] 
            hist_item = hist_item + [0] * (max_len - len(hist_item)) 
            pos_item = pos_list[i] 

            
            if pos_label == len(label): 
                pos_seq = [0, pos_item, uid, hist_item, pos_time] 
            else:
                pos_seq = [1, pos_item, uid, hist_item, pos_time] 

            if len(item_attribute_cols) > 0: 
                for attr_col in item_attribute_cols:  
                    if attr_col in not_sequence_cols: 
                        pos2attr = [item2attr[attr_col][pos_item]]
                    else:
                        hist_attr = hist[attr_col].tolist()[start_episode:i]
                        
                        if torch.is_tensor(hist_attr[0]):
                            
                            zero_tensor = torch.zeros((max_len - len(hist_attr), hist_attr[0].shape[1]))
                            
                            hist_attr = torch.cat(hist_attr, dim=0)
                            
                            hist_attr = torch.cat([hist_attr, zero_tensor], dim=0)
                        
                        else:
                            hist_attr = hist_attr + [0] * (max_len - len(hist_attr))
                        pos2attr = [hist_attr, item2attr[attr_col][pos_item]]
                    pos_seq += pos2attr

            train_data.append(pos_seq) 
        '''
        
        col_name = ['label', 'target_item_id', user_col, 'hist_item_id', 'time']
        if len(item_attribute_cols) > 0:
            for attr_col in item_attribute_cols:  
                name = ['hist_'+attr_col, 'target_'+attr_col]
                col_name += name

        
        if shuffle:
            random.shuffle(train_data)

        
        train_data = pd.DataFrame(train_data, columns=col_name).sort_values('time')
        train_data = train_data.drop('time', axis=1).copy()
        
        with open(f'train_data/{uid}.pkl','wb') as f:
            pickle.dump(train_data, f)

        f.close()
        '''
    
    col_name = ['label', 'target_item_id', user_col, 'hist_item_id', 'time']
    if len(item_attribute_cols) > 0:
        for attr_col in item_attribute_cols:  
            if attr_col in not_sequence_cols:
                name = ['target_'+attr_col]
            else:
                name = ['hist_'+attr_col, 'target_'+attr_col]
            col_name += name

    
    if shuffle:
        random.shuffle(train_data)

    
    train = pd.DataFrame(train_data, columns=col_name)

    
    return train, action_to_index


def generate_seq_feature_old(data,
                         user_col,
                         item_col,
                         time_col,
                         item_attribute_cols=[],
                         min_item=0,
                         shuffle=True,
                         max_len=50):
    """generate sequence feature and negative sample for ranking.

    Args:
        data (pd.DataFrame): the raw data.
        user_col (str): the col name of user_id
        item_col (str): the col name of item_id
        time_col (str): the col name of timestamp
        item_attribute_cols (list[str], optional): the other attribute cols of item which you want to generate sequence feature. Defaults to `[]`.
        sample_method (int, optional): the negative sample method `{
            0: "random sampling",
            1: "popularity sampling method used in word2vec",
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`.
            Defaults to 0.
        min_item (int, optional): the min item each user must have. Defaults to 0.
        shuffle (bool, optional): shulle if True
        max_len (int, optional): the max length of a user history sequence.

    Returns:
        pd.DataFrame: split train, val and test data with sequence features by time.
    """
    for feat in data:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])
        data[feat] = data[feat].apply(lambda x: x + 1)  
    data = data.astype('int32')

    
    n_items = data[item_col].max()
    item2attr = {}
    if len(item_attribute_cols) > 0:
        for col in item_attribute_cols:
            map = data[[item_col, col]]
            item2attr[col] = map.set_index([item_col])[col].to_dict()

    train_data, val_data, test_data = [], [], []
    data.sort_values(time_col, inplace=True)
    
    for uid, hist in tqdm.tqdm(data.groupby(user_col), desc='generate sequence features'):
        pos_list = hist[item_col].tolist()
        len_pos_list = len(pos_list)
        if len_pos_list < min_item:  
            continue

        neg_list = [neg_sample(pos_list, n_items) for _ in range(len_pos_list)]
        for i in range(1, min(len_pos_list, max_len)):
            hist_item = pos_list[:i]
            hist_item = hist_item + [0] * (max_len - len(hist_item))
            pos_item = pos_list[i]
            neg_item = neg_list[i]
            pos_seq = [1, pos_item, uid, hist_item]
            neg_seq = [0, neg_item, uid, hist_item]
            if len(item_attribute_cols) > 0:
                for attr_col in item_attribute_cols:  
                    hist_attr = hist[attr_col].tolist()[:i]
                    hist_attr = hist_attr + [0] * (max_len - len(hist_attr))
                    pos2attr = [hist_attr, item2attr[attr_col][pos_item]]
                    neg2attr = [hist_attr, item2attr[attr_col][neg_item]]
                    pos_seq += pos2attr
                    neg_seq += neg2attr
            if i == len_pos_list - 1:
                test_data.append(pos_seq)
                test_data.append(neg_seq)
            elif i == len_pos_list - 2:
                val_data.append(pos_seq)
                val_data.append(neg_seq)
            else:
                train_data.append(pos_seq)
                train_data.append(neg_seq)

    col_name = ['label', 'target_item_id', user_col, 'hist_item_id']
    if len(item_attribute_cols) > 0:
        for attr_col in item_attribute_cols:  
            name = ['hist_'+attr_col, 'target_'+attr_col]
            col_name += name

    
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    train = pd.DataFrame(train_data, columns=col_name)
    val = pd.DataFrame(val_data, columns=col_name)
    test = pd.DataFrame(test_data, columns=col_name)

    return train, val, test

def df_to_dict(data):
    """
    Convert the DataFrame to a dict type input that the network can accept
    Args:
        data (pd.DataFrame): datasets of type DataFrame
    Returns:
        The converted dict, which can be used directly into the input network
    """
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict


def neg_sample(click_hist, item_size):
    neg = random.randint(1, item_size)
    while neg in click_hist:
        neg = random.randint(1, item_size)
    return neg


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length.
        This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences
        reference: https://github.com/huawei-noah/benchmark/tree/main/FuxiCTR/fuxictr
    """
    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncating in ["pre", "post"], "Invalid truncating={}.".format(truncating)

    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr


def array_replace_with_dict(array, dic):
    """Replace values in NumPy array based on dictionary.
    Args:
        array (np.array): a numpy array
        dic (dict): a map dict

    Returns:
        np.array: array with replace
    """
    
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    
    idx = k.argsort()
    return v[idx[np.searchsorted(k, array, sorter=idx)]]
