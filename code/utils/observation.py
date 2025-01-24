import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

def tsne(tensor1, tensor2, tensor3, tensor4 = None):

    
    
    
    
    plt.rcParams['font.size'] = 18   
    plt.figure(figsize=(8, 6))

    if tensor4 != None:

        
        combined_tensor = torch.cat((tensor1, tensor2, tensor3, tensor4), dim=0)

        
        tsne = TSNE(n_components=2)
        embedded_tensor = tsne.fit_transform(combined_tensor.cpu().detach().numpy())

        
        labels = ['IDs Features'] * tensor1.shape[0] + ['Text Features'] * tensor2.shape[0] + ['Image Features'] * tensor3.shape[0] + ['Fusion Features'] * tensor4.shape[0]

        legend_labels = ['IDs Features', 'Text Features', 'Image Features', 'Fusion Features']

        
        for label in legend_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embedded_tensor[indices, 0], embedded_tensor[indices, 1], color=legend_colors[legend_labels.index(label)], label=label)
    
    else:
        
        combined_tensor = torch.cat((tensor1, tensor2, tensor3), dim=0)

        
        tsne = TSNE(n_components=2)
        embedded_tensor = tsne.fit_transform(combined_tensor.cpu().detach().numpy())

        
        labels = ['IDs Features'] * tensor1.shape[0] + ['Text Features'] * tensor2.shape[0] + ['Image Features'] * tensor3.shape[0]


        
        legend_labels = ['IDs Features', 'Text Features', 'Image Features']

        
        for label in legend_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embedded_tensor[indices, 0], embedded_tensor[indices, 1], color=legend_colors[legend_labels.index(label)], label=label)
    
    
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    plt.legend(loc = 'lower right')
    plt.savefig('scatter_plot.pdf')
    
    
    