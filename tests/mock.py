from tensorflow import ones, zeros, random

class MockInput():

    def __init__(self, batch_size: int = 5, image_height: int = 8, image_width: int = 8, channels: int = 3) -> None:
        self.batch_size = max(1, batch_size)
        self.image_height = max(1,image_height)
        self.image_width = max(1,image_width)
        self.channels = max(1,channels)

        self.shape = (self.batch_size, self.image_width, self.image_height, self.channels)

    def ones(self):
        return ones(shape = self.shape)

    def zeros(self):
        return zeros(shape = self.shape)
    
    def random(self):
        return random.normal(shape = self.shape)
