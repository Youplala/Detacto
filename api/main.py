import os
from matplotlib import gridspec
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

cred = credentials.Certificate(os.path.basename('firebase_key.json'))
firebase_admin.initialize_app(cred, {
    'storageBucket': 'cloudcomputing-327312.appspot.com'
})
bucket = storage.bucket()

app = Flask(__name__)

MODEL_PATH = 'model'
hub_module = hub.load(MODEL_PATH)
 

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image


#@functools.lru_cache(maxsize=None)
def load_image(image_url, number, image_size=(256, 256), preserve_aspect_ratio=True):
  image_path = tf.keras.utils.get_file(os.path.basename(f"image-{number}.jpg"), image_url)
  print(image_path)
  img = tf.io.read_file(image_path)

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(img, channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img, image_path

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

def uploadImageToFirebase(image_url):
  blob = bucket.blob(os.path.basename(image_url))
  blob.upload_from_filename(os.path.basename(image_url))
  return blob.public_url


@app.route('/api', methods = ['POST']) 
def run():

  try:
    content = request.get_json()
    print(content)
    path1 = content['pic1']
    path2 = content['pic2']
    output_image_size = 384  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image, path1 = load_image(path1, 0, content_img_size)
    style_image, path2 = load_image(path2, 1, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    img = stylized_image[0].numpy()
    tf.keras.utils.save_img(os.path.basename("stylized.jpg"), img)

    url = uploadImageToFirebase(os.path.basename("stylized.jpg"))

    os.remove(os.path.basename("stylized.jpg")) 
    os.remove(path1)
    os.remove(path2)

  except:
    return jsonify({"status": "bad"})
  
  return jsonify({'status': 'ok', 'image_url': url})

if __name__ == '__main__':

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))