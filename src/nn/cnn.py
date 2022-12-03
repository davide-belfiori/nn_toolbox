from keras.layers import Input
from keras.models import Model
from nn_modules.mlp import MLP
from nn_modules.cnn import *

# ----------------
# --- NETWORKS ---
# ----------------

class Conv2DResnet():

    def __init__(self, input_shape: tuple, 
                       filter_list: 'list[int]', 
                       pool_size: int = None, 
                       kernel_size: int = 3, 
                       activ_type: int = ACT_RELU, 
                       group_size: int = 4, 
                       res_block_type: int = RES_TYPE_ORIGIN, 
                       block_drop_rate: float = 0.0, 
                       mlp: MLP = None, 
                       preprocessing = None) -> None:
        self.input_shape = input_shape
        self.filter_list = filter_list
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.activ_type = activ_type
        self.group_size = group_size
        self.res_block_type = res_block_type
        self.block_drop_rate = block_drop_rate
        self.mlp = mlp
        self.preprocessing = preprocessing
        # >>> Build the model
        self.input = Input(shape=self.input_shape)
        x = self.input
        if self.preprocessing != None:
            x = self.preprocessing(x)
        if self.pool_size != None:
            in_filters = self.input_shape[-1] if self.preprocessing == None else self.preprocessing.filters
            for filters in self.filter_list:
                x = Conv2DResDownBlock(filters = filters, 
                                       pool_size = self.pool_size, 
                                       kernel_size = self.kernel_size, 
                                       activ_type = self.activ_type, 
                                       group_size = self.group_size, 
                                       block_type = self.res_block_type, 
                                       input_filters = in_filters,
                                       drop_rate = self.block_drop_rate)(x)
                in_filters = filters
        else:
            in_filters = self.input_shape[-1] if self.preprocessing == None else self.preprocessing.filters
            for filters in self.filter_list:
                x = Conv2DResBlock(filters = filters,
                                   kernel_size = self.kernel_size,
                                   activ_type = self.activ_type,
                                   group_size = self.group_size,
                                   block_type = self.res_block_type,
                                   input_filters = in_filters,
                                   drop_rate = self.block_drop_rate)(x)
                in_filters = filters
        if self.mlp != None:
            self.output = mlp(x)
        else:
            self.output = x
        
    def as_model(self):
        return Model(inputs = [self.input], outputs = [self.output])
