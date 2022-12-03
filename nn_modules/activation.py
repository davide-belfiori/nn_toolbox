from keras.layers import Activation, Layer

# ---------------
# --- GLOBALS ---
# ---------------

# >>> Activation types
#
ACT_RELU = 0
ACT_SIGM = 1
ACT_TANH = 2
ACT_GELU = 3

# -----------------
# --- FUNCTIONS ---
# -----------------

def activation(activ_type: int = ACT_RELU) -> Activation:
    if activ_type == ACT_RELU:
        return Activation("relu")
    if activ_type == ACT_SIGM:
        return Activation("sigmoid")
    if activ_type == ACT_TANH:
        return Activation("tanh")
    if activ_type == ACT_GELU:
        return Activation("gelu")
    
    raise ValueError("Invalid activation type.")

# ---------------
# --- MODULES ---
# ---------------

class Identity(Layer):
    '''
        Identity activation function.
    '''
    def __init__(self):
        super(Identity, self).__init__()
    
    def call(self, inputs):
        return inputs