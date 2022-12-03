from tensorflow_addons.layers import GroupNormalization
from keras.layers import Layer, Conv2D, Add, Dropout, MaxPooling2D, UpSampling2D
from nn_modules.activation import *

# ---------------
# --- GLOBALS ---
# ---------------

# >>> RESNET BLOCK TYPES
#
RES_TYPE_ORIGIN = 0
RES_TYPE_NORM_AFTER_ADD = 1
RES_TYPE_ACTIV_AFTER_ADD = 2
RES_TYPE_PRE_ACTIV = 3
RES_TYPE_FULL_PRE_ACTIV = 4

# ---------------
# --- MODULES ---
# ---------------

class Conv2DBlock(Layer):

    def __init__(self, filters: int, kernel_size: int = 3):
        super(Conv2DBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = Conv2D(filters=filters, kernel_size=self.kernel_size, padding="same")
        self.conv2 = Conv2D(filters=filters, kernel_size=self.kernel_size, padding="same")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size
        })
        return config

class Conv2DResBlock(Conv2DBlock):

    def __init__(self, filters: int, kernel_size: int = 3, activ_type = ACT_RELU, group_size: int = 4, block_type = RES_TYPE_ORIGIN, input_filters: int = None, drop_rate: float = 0.0):
        super(Conv2DResBlock, self).__init__(filters=filters, kernel_size=kernel_size)
        self.activ_type = activ_type
        self.activ = activation(activ_type)
        if self.filters % group_size != 0:
            self.group_size = filters
        else:
            self.group_size = group_size
        if input_filters != None:
            self.input_filters = input_filters
        else:
            self.input_filters = filters
        self.norm1 = GroupNormalization(groups=self.filters // self.group_size, axis=3)
        self.norm2 = GroupNormalization(groups=self.filters // self.group_size, axis=3)
        self.res_conv = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")
        self.res_add = Add()
        self.drop_rate = drop_rate
        self.drop = Dropout(rate=self.drop_rate)
        self.block_type = block_type

        if self.block_type == RES_TYPE_FULL_PRE_ACTIV:
            if self.input_filters % self.group_size != 0:
                self.norm1 = GroupNormalization(groups=1, axis=3)
            else:
                self.norm1 = GroupNormalization(groups=self.input_filters // self.group_size)
        
    def call(self, inputs):
        if self.block_type == RES_TYPE_ORIGIN:
            x = self.conv1(inputs)
            x = self.norm1(x)
            x = self.activ(x)
            x = self.conv2(x)
            x = self.norm2(x)
            res = self.res_conv(inputs)
            x = self.res_add([x, res])
            x = self.activ(x)
            return self.drop(x)
        if self.block_type == RES_TYPE_NORM_AFTER_ADD:
            x = self.conv1(inputs)
            x = self.norm1(x)
            x = self.activ(x)
            x = self.conv2(x)
            res = self.res_conv(inputs)
            x = self.res_add([x, res])
            x = self.norm2(x)
            x = self.activ(x)
            return self.drop(x)
        if self.block_type == RES_TYPE_ACTIV_AFTER_ADD:
            x = self.conv1(inputs)
            x = self.norm1(x)
            x = self.activ(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.activ(x)
            res = self.res_conv(inputs)
            x = self.res_add([x, res])
            return self.drop(x)
        if self.block_type == RES_TYPE_PRE_ACTIV:
            x = self.activ(inputs)
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.activ(x)
            x = self.conv2(x)
            x = self.norm2(x)
            res = self.res_conv(inputs)
            x = self.res_add([x, res])
            return self.drop(x)
        if self.block_type == RES_TYPE_FULL_PRE_ACTIV:
            x = self.norm1(inputs)
            x = self.activ(x)
            x = self.conv1(x)
            x = self.norm2(x)
            x = self.activ(x)
            x = self.conv2(x)
            res = self.res_conv(inputs)
            x = self.res_add([x, res])
            return self.drop(x)

        raise ValueError("Invalid Resnet block type.")

    def get_config(self):
        config = super().get_config()
        config.update({
            "activ_type": self.activ_type,
            "group_size": self.group_size,
            "block_type": self.block_type,
            "drop_rate": self.drop_rate
        })
        return config

class Conv2DResDownBlock(Conv2DResBlock):

    def __init__(self, filters: int, pool_size: int = 2, kernel_size: int = 3, activ_type=ACT_RELU, group_size: int = 4, block_type=RES_TYPE_ORIGIN, input_filters: int = None, drop_rate: float = 0):
        super(Conv2DResDownBlock, self).__init__(filters, kernel_size, activ_type, group_size, block_type, input_filters, drop_rate)
        self.pool_size = pool_size
        self.down = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))

    def call(self, inputs):
        x = super().call(inputs)
        return self.down(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size
        })
        return config

class Conv2DResUpBlock(Conv2DResBlock):

    def __init__(self, filters: int, sample_size: int = 2, kernel_size: int = 3, activ_type=ACT_RELU, group_size: int = 4, block_type=RES_TYPE_ORIGIN, input_filters: int = None, drop_rate: float = 0):
        super(Conv2DResUpBlock, self).__init__(filters, kernel_size, activ_type, group_size, block_type, input_filters, drop_rate)
        self.sample_size = sample_size
        self.up = UpSampling2D(size=(self.sample_size, self.sample_size))

    def call(self, inputs):
        x = super().call(inputs)
        return self.up(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_size": self.sample_size
        })
        return config
