import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

'''
This script creates training and validation curves from the log output of RfDNet.
The filter functions can be used to filter out different sub-losses. All sub-losses of GroupFree3D are visualized by default.
'''

filter_function = lambda x: x[0] not in '0123456789'
#filter_function = lambda x: 'last' in x
#filter_function = lambda x: x[0] in '0123456789'
#filter_function = lambda x: 'objectness' in x
#filter_function = lambda x: 'heading' in x
#filter_function = lambda x: 'size' in x # drops well
#filter_function = lambda x: 'box_loss' in x # drops decently
#filter_function = lambda x: 'sem_cls' in x # drops decently
#filter_function = lambda x: 'center' in x # drops so well
#filter_function = lambda x: 'total' == x # drops so well

loss_names = [
    'total',
    'query_points_generation_loss',
    'proposal_objectness_loss',
    'last_objectness_loss',
    '0head_objectness_loss',
    '1head_objectness_loss',
    '2head_objectness_loss',
    '3head_objectness_loss',
    '4head_objectness_loss',
    'sum_heads_objectness_loss',
    'proposal_center_loss',
    'proposal_heading_cls_loss',
    'proposal_heading_reg_loss',
    'proposal_size_cls_loss',
    'proposal_size_reg_loss',
    'proposal_box_loss',
    'proposal_sem_cls_loss',
    'last_center_loss',
    'last_heading_cls_loss',
    'last_heading_reg_loss',
    'last_size_cls_loss',
    'last_size_reg_loss',
    'last_box_loss',
    'last_sem_cls_loss',
    '0head_center_loss',
    '0head_heading_cls_loss',
    '0head_heading_reg_loss',
    '0head_size_cls_loss',
    '0head_size_reg_loss',
    '0head_box_loss',
    '0head_sem_cls_loss',
    '1head_center_loss',
    '1head_heading_cls_loss',
    '1head_heading_reg_loss',
    '1head_size_cls_loss',
    '1head_size_reg_loss',
    '1head_box_loss',
    '1head_sem_cls_loss',
    '2head_center_loss',
    '2head_heading_cls_loss',
    '2head_heading_reg_loss',
    '2head_size_cls_loss',
    '2head_size_reg_loss',
    '2head_box_loss',
    '2head_sem_cls_loss',
    '3head_center_loss',
    '3head_heading_cls_loss',
    '3head_heading_reg_loss',
    '3head_size_cls_loss',
    '3head_size_reg_loss',
    '3head_box_loss',
    '3head_sem_cls_loss',
    '4head_center_loss',
    '4head_heading_cls_loss',
    '4head_heading_reg_loss',
    '4head_size_cls_loss',
    '4head_size_reg_loss',
    '4head_box_loss',
    '4head_sem_cls_loss',
    'sum_heads_box_loss',
    'sum_heads_sem_cls_loss',
    'loss',
	# custom
	'custom_heading_loss'
]

loss_names = list(filter(filter_function, loss_names))
loss_dict = defaultdict(list)

def get_losses(mode='train'):
    for loss_name in loss_names:
        pattern = fr'Currently the last {mode} loss \({loss_name}\) is: (\d+\.\d+)'
        print(pattern)
        with open('./trainloss.txt') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    loss = float(match.group(1))
                    loss_dict[loss_name].append(loss)

def plot_losses():
    for name, loss in loss_dict.items():
        #print(loss)
        plt.plot(loss, label=name)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

get_losses()
plot_losses()
