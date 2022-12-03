from keras.layers import Layer, Dense, Dropout, Flatten
from nn_modules.activation import *

class MLP(Layer):

    def __init__(self, unit_list: 'list[int]', flatten_input: bool = False, input_drop_rate: float = 0.0, layer_activ_type: int = ACT_RELU, output_activ_type: int = ACT_RELU, out_drop_rate: float = 0.0) -> None:
        super(MLP, self).__init__()
        self.unit_list = unit_list
        self.flatten_input = flatten_input
        self.input_drop_rate = input_drop_rate
        self.layer_activ_type = layer_activ_type
        self.output_activ_type = output_activ_type
        self.out_drop_rate = out_drop_rate
        self.in_drop = Dropout(rate=self.input_drop_rate)
        self.flat = Flatten()
        self.mlp = []
        for units in self.unit_list:
            self.mlp.append(Dense(units=units))
        self.layer_activ = activation(self.layer_activ_type) if self.layer_activ_type != None else Identity()
        self.out_activ = activation(self.output_activ_type) if self.output_activ_type != None else Identity()
        self.out_drop = Dropout(rate = self.out_drop_rate)

    def call(self, inputs):
        x = self.in_drop(inputs)
        if self.flatten_input:
            x = self.flat(x)
        for layer in self.mlp:
            x = layer(x)
            x = self.layer_activ(x)
        x = self.out_activ(x)
        return self.out_drop(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "unit_list": self.unit_list,
            "input_drop_rate": self.input_drop_rate,
            "layer_activ_type": self.layer_activ_type,
            "output_activ_type": self.output_activ_type,
            "out_drop_rate": self.out_drop_rate
        })
        return config 