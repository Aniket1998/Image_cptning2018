import vgg
import scipy.misc
import tensorflow as tf

image_PATH='bkg.png'
Weight_PATH='imagenet-vgg-verydeep-19.mat'
layer='relu5_4'

img = scipy.misc.imread(image_PATH, mode='RGB')
shape = (1,) + img.shape
print(shape)
with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
    image = tf.placeholder(tf.float64, shape=shape, name='image')
    image_pre = vgg.preprocess(img)
    net= vgg.net(Weight_PATH,image_pre)
    features = net[layer].eval(feed_dict={image:image_pre})

print(features)
