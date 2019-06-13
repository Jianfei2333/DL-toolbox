"""
This script is the collection of color constancy methods.
"""
import numpy as np

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