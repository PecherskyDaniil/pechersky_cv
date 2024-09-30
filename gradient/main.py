import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

size = 100
image = np.zeros((size, size, 3), dtype="uint8")
assert image.shape[0] == image.shape[1]

color1 = [255, 128, 0]
color2 = [0, 128, 255]

for i, v1 in enumerate(np.linspace(0, 0.5, image.shape[0])):
  for j, v2 in enumerate(np.linspace(0, 0.5, image.shape[0])):
    r = lerp(color2[0], color1[0], v1+v2)
    g = lerp(color2[1], color1[1], v1+v2)
    b = lerp(color2[2], color1[2], v1+v2)
    image[i, j, :] = [r, g, b]

plt.figure(1)
plt.imshow(image)
plt.show()
