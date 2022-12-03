from nn_modules.activation import Identity
import numpy as np

def test_identity() -> None:
    input_tensor = np.random.random(size = (4, 16, 16, 3))
    identity = Identity()
    output_tensor = identity(input_tensor)
    assert np.abs(input_tensor - output_tensor).sum() == 0
