from tensorflow import keras
import tensorflow as tf
import pickle
import numpy as np
import os
import re
from path import Path
from typing import List
import shutil
from glob import glob
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from collections import defaultdict
from dataclasses import dataclass
import cv2
from sklearn.cluster import DBSCAN
import difflib

class BBox:
    x: int
    y: int
    w: int
    h: int
    def __init__(self,xi,yi,wi,hi):
      self.x=xi
      self.y=yi
      self.w=wi
      self.h=hi


class DetectorRes:
    img: np.ndarray
    bbox: BBox
    def __init__(self,i,b):
      self.img=i
      self.bbox=b


def detect(img: np.ndarray,
           kernel_size: int,
           sigma: float,
           theta: float,
           min_area: int) -> List[DetectorRes]:
    assert img.ndim == 2
    assert img.dtype == np.uint8

    # apply filter kernel
    kernel = _compute_kernel(kernel_size, sigma, theta)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    img_thres = 255 - cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 97, 2)
    # plt.imshow(img_thres,cmap='gray')
    # plt.show()


    # append components to result
    height, width = img_thres.shape

    # append components to result
    res = []
    components = cv2.findContours(img_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in components:
        x, y, w, h = cv2.boundingRect(c)  # bounding box as tuple (x, y, w, h)
        if (cv2.contourArea(c) < min_area) or (w<width*0.04) or (h<height*0.025) or (w>width*0.8) or (h>height*0.5) or (x<width*0.01) or (x+w> width*0.95) or (y<height*0.01) or (y+h> height*0.95):        # skip small word candidates

            continue
        # append bounding box and image of word to result list

        crop = img[y:y + h, x:x + w]
        res.append(DetectorRes(crop, BBox(x, y, w, h)))

    return res


def _compute_kernel(kernel_size: int,
                    sigma: float,
                    theta: float) -> np.ndarray:
    """Compute anisotropic filter kernel."""

    assert kernel_size % 2  # must be odd size

    # create coordinate grid
    half_size = kernel_size // 2
    xs = ys = np.linspace(-half_size, half_size, kernel_size)
    x, y = np.meshgrid(xs, ys)

    # compute sigma values in x and y direction, where theta is roughly the average x/y ratio of words
    sigma_y = sigma
    sigma_x = sigma_y * theta

    # compute terms and combine them
    exp_term = np.exp(-x ** 2 / (2 * sigma_x) - y ** 2 / (2 * sigma_y))
    x_term = (x ** 2 - sigma_x ** 2) / (2 * np.math.pi * sigma_x ** 5 * sigma_y)
    y_term = (y ** 2 - sigma_y ** 2) / (2 * np.math.pi * sigma_y ** 5 * sigma_x)
    kernel = (x_term + y_term) * exp_term

    # normalize and return kernel
    kernel = kernel / np.sum(kernel)
    return kernel


def prepare_img(img: np.ndarray,
                height: int) -> np.ndarray:
    """Convert image to grayscale image (if needed) and resize to given height."""
    assert img.ndim in (2, 3)
    assert height > 0
    assert img.dtype == np.uint8
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def sort_multiline(detections: List[DetectorRes],
                   max_dist: float = 0.7,
                   min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    """Cluster detections into lines, then sort the lines according to x-coordinates of word centers.

    Args:
        detections: List of detections.
        max_dist: Maximum Jaccard distance (0..1) between two y-projected words to be considered as neighbors.
        min_words_per_line: If a line contains less words than specified, it is ignored.

    Returns:
        List of lines, each line itself a list of detections.
    """
    lines = _cluster_lines(detections, max_dist, min_words_per_line)
    res = []
    for line in lines:
        res += sort_line(line)
    return res


def _cluster_lines(detections: List[DetectorRes],
                   max_dist: float = 0.7,
                   min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    # print(len(detections))
    num_bboxes = len(detections)
    dist_mat = np.ones((num_bboxes, num_bboxes))
    for i in range(num_bboxes):
        for j in range(i, num_bboxes):
            a = detections[i].bbox
            b = detections[j].bbox
            if a.y > b.y + b.h or b.y > a.y + a.h:
                continue
            intersection = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            union = a.h + b.h - intersection
            iou = np.clip(intersection / union if union > 0 else 0, 0, 1)
            dist_mat[i, j] = dist_mat[j, i] = 1 - iou  # Jaccard distance is defined as 1-iou

    dbscan = DBSCAN(eps=max_dist, min_samples=min_words_per_line, metric='precomputed').fit(dist_mat)

    clustered = defaultdict(list)
    for i, cluster_id in enumerate(dbscan.labels_):
        if cluster_id == -1:
            continue
        clustered[cluster_id].append(detections[i])

    res = sorted(clustered.values(), key=lambda line: [det.bbox.y + det.bbox.h / 2 for det in line])
    return res


def sort_line(detections: List[DetectorRes]) -> List[List[DetectorRes]]:
    """Sort the list of detections according to x-coordinates of word centers."""
    return [sorted(detections, key=lambda det: det.bbox.x + det.bbox.w / 2)]
 
 
def ocr_image(src_img,model,processor):
  img = Image.open(src_img).convert("RGB")
  pixel_values = processor(images=img, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def match_medicines(sentence, medicines):
  pairs=[]
  for word in sentence.split():
    for medicine in medicines:
      theta=difflib.SequenceMatcher(None, word.lower(), medicine.lower()).ratio()
      if theta>0.7:
        # print('appending '+ word + medicine + str(theta))
        pairs.append((medicine, theta))

  if(len(pairs)>0):
    max_pair = max(pairs, key=lambda x: x[1])
    return max_pair[0]
  else:
    return (None)


def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        res += Path(data_dir).files(ext)
    return res


def preprocess(fn_img):

  list_img_names_serial = []


  data= "/content"   # Put path here
  kernel_size=51
  sigma=35
  theta=50
  min_area=100
  img_height=1000

  img = prepare_img(cv2.imread(fn_img), img_height)
  height, width = img.shape
  detections = detect(img,
                      kernel_size=kernel_size,
                      sigma=sigma,
                      theta=theta,
                      min_area=min_area)
  # if (len(detections)==0):
  #   continue
  lines = sort_multiline(detections)
  path = './test_images'
  isExist = os.path.exists(path)
  if isExist == False:
      os.mkdir(path)
  else:
      shutil.rmtree(path)
      os.mkdir(path)

  # plt.imshow(img, cmap='gray')
  # num_colors = 7
  # colors = plt.cm.get_cmap('rainbow', num_colors)
  for line_idx, line in enumerate(lines):
      for word_idx, det in enumerate(line):
          xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
          ys = [det.bbox.y, det.bbox.y + det.bbox.h,
                det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
          # plt.plot(xs, ys, c=colors(line_idx % num_colors))
          # plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
          # print(det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h)
          crop_img = img[det.bbox.y - 15 :det.bbox.y +
                          det.bbox.h + 15, det.bbox.x - 15 :det.bbox.x+det.bbox.w + 15]

          cv2.imwrite(f"{path}/line" + str(line_idx) + "word" +
                      str(word_idx) + ".jpg", crop_img)
          full_img_path = "line" + str(line_idx) + "word" + str(word_idx)+".jpg"
          list_img_names_serial.append(full_img_path)
          # print(list_img_names_serial)
          list_img_names_serial_set = set(list_img_names_serial)


  path = './test_images'
  files = sorted(glob(os.path.join(path, "*")), key=os.path.basename)
  return(files)


def inf_ocr(files):
  df = pd.read_csv('ocr\meds_db.csv')
  df = df.astype(str)
  medicines = df.values.flatten().tolist()
  out=""
  meds=[]
  name=files[0][:20]
  processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1") #update path here
  model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-stage1")


  for i in range(len(files)):
    # print('processing ' +files[i])
    if name==files[i][:20]:
      out= out + " " + ocr_image(files[i],model,processor)
    else:
      # print('going to next line')
      t=match_medicines(out, medicines)
      if(t):
        meds.append(t)
      out=str(ocr_image(files[i],model,processor))
      name=files[i][:20]
  t=match_medicines(out, medicines)
  if(t):
    meds.append(t)

  return(meds)


def infer(path):
  # path="/content/2024_04_04 23_00 Office Lens.jpg"
  files=preprocess(path)
  return(inf_ocr(files))


# meds=infer("ocr\pres.png")
# print("endigs")
# print(meds)
