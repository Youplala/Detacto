import functools
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import io
import os
from flask import Flask, jsonify, request
import requests
from PIL import Image
from flask import send_file
import sys
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

cred = credentials.Certificate('firebase_key.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'cloudcomputing-327312.appspot.com'
})
bucket = storage.bucket()




app = Flask(__name__)
 

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  #image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  image_path = image_url
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w  * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()


@app.route('/')
def classify():

  url='http://172.22.54.170:8080/run'

  multiple_files = [
      ('image', open('image.jpg', 'rb')),
      ('image', open('style.jpg', 'rb'))
  ]

  payload = {'id': '123', 'type': 'jpg'}

  r = requests.post(url, files=multiple_files, data=payload)

  # convert server response into JSON format.
  return r

@app.route('/run', methods = ['POST']) 
def run():

  MODEL_PATH = 'model'
  hub_module = hub.load(MODEL_PATH)

  files = request.files.to_dict(flat=False)
  images = files['image'] # list containing two images
  for i, file in enumerate(images):
    file.save(f'image-{i}.jpg') # Save images from POST request

  img = Image.open('image-1.jpg')

  content_image_url = "image-0.jpg"  # @param {type:"string"}
  style_image_url = "image-1.jpg"  # @param {type:"string"}
  output_image_size = 384  # @param {type:"integer"}

  # The content image size can be arbitrary.
  content_img_size = (output_image_size, output_image_size)
  # The style prediction model was trained with image size 256 and it's the 
  # recommended image size for the style image (though, other sizes work as 
  # well but will lead to different results).
  style_img_size = (256, 256)  # Recommended to keep it at 256.

  content_image = load_image(content_image_url, content_img_size)
  style_image = load_image(style_image_url, style_img_size)
  style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
  #show_n([content_image, style_image], ['Content image', 'Style image'])
  #hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
  #hub_handle = 'https://storage.googleapis.com/tfhub-modules/google/magenta/arbitrary-image-stylization-v1-256/2.tar.gz'
  outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
  stylized_image = outputs[0]
  img = stylized_image[0].numpy()
  tf.keras.utils.save_img("stylized.jpg", img)

  im = Image.open('stylized.jpg')
  buf = io.BytesIO()
  im.save(buf, format='JPEG')
  b = buf.getvalue()


  #image_data = requests.get("https://brosset.li/wp-content/uploads/2021/07/miroir.jpg").content
  #b = bytearray(img)
  blob = bucket.blob('stylized.jpg')
  blob.upload_from_string(
          b,
          content_type='image/jpg'
      )
  print(blob.public_url)
  #show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
  return jsonify(blob.public_url)
  #return jsonify({'msg': 'success'})

if __name__ == '__main__':

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))