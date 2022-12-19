import numpy as np
from PIL import Image

filename = "image.png"
image = Image.open(filename)

np_image = np.asarray(image)
print(np_image.shape)
np_image = np.expand_dims(np_image, axis=0)
print(np_image.shape)

outfilename = "image.npz"
np.savez(outfilename, np_image, np.asarray([551]))
