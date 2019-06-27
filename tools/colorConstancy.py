"""
This script is the collection of color constancy methods.
"""
import numpy as np
import cv2

def Grey_world(Img):
  nimg = np.asarray(Img)
  nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
  avgB = np.average(nimg[0])
  avgG = np.average(nimg[1])
  avgR = np.average(nimg[2])

  avg = (avgB + avgG + avgR) / 3

  nimg[0] = np.minimum(nimg[0] * (avg / avgB), 255)
  nimg[1] = np.minimum(nimg[1] * (avg / avgG), 255)
  nimg[2] = np.minimum(nimg[2] * (avg / avgR), 255)

  return nimg.transpose(1,2,0).astype(np.uint8)

def His_equ(Img):
  import cv2
  nimg = np.asarray(Img)
  ycrcb = cv2.cvtColor(nimg, cv2.COLOR_BGR2YCR_CB)
  channels = cv2.split(ycrcb)
  cv2.equalizeHist(channels[0], channels[0])
  cv2.merge(channels, ycrcb)
  img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
  return img_eq

def White_balance(nimg):
  """
  Very very slow
  """
  import cv2
  rows = nimg.shape[0]
  cols = nimg.shape[1]

  final = cv2.cvtColor(nimg, cv2.COLOR_BGR2LAB)

  avg_a = np.average(final[:,:,1])
  avg_b = np.average(final[:,:,2])

  for x in range(final.shape[0]):
    for y in range(final.shape[1]):
      l, a, b = final[x, y, :]
      l *= 100/255.0
      final[x,y,1] = a-((avg_a-128)*(1/100.0)*1.1)
      final[x,y,2] = b-((avg_b-128)*(1/100.0)*1.1)

  final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

  return final

def shades_of_gray(img, power=6, gamma=None):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255*pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    
    return img.astype(img_dtype)
