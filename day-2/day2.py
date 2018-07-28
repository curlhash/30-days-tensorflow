import tensorflow as tf
import sys
from PIL import Image,ImageFilter
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

nHiddenLayer1 = 1000
nHiddenLayer2 = 1000
nHiddenLayer3 = 1000
nHiddenLayer4 = 1000

nClasses = 10
batchSize = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neuralNetworkModel(data):
	hiddenLayer_1 = {
		'weights': tf.Variable(tf.random_normal([784, nHiddenLayer1])),
		'biases': tf.Variable(tf.random_normal([nHiddenLayer1]))
	}

	hiddenLayer_2 = {
		'weights': tf.Variable(tf.random_normal([nHiddenLayer1, nHiddenLayer2])),
		'biases': tf.Variable(tf.random_normal([nHiddenLayer2]))
	}

	hiddenLayer_3 = {
		'weights': tf.Variable(tf.random_normal([nHiddenLayer2, nHiddenLayer3])),
		'biases': tf.Variable(tf.random_normal([nHiddenLayer3]))
	}

	hiddenLayer_4 = {
		'weights': tf.Variable(tf.random_normal([nHiddenLayer3, nHiddenLayer4])),
		'biases': tf.Variable(tf.random_normal([nHiddenLayer4]))
	}

	outputLayer = {
		'weights': tf.Variable(tf.random_normal([nHiddenLayer4, nClasses])),
		'biases': tf.Variable(tf.random_normal([nClasses]))
	}


	l_1 = tf.add(tf.matmul(data, hiddenLayer_1['weights']), hiddenLayer_1['biases'])
	l_1 = tf.nn.relu(l_1)

	l_2 = tf.add(tf.matmul(l_1, hiddenLayer_2['weights']), hiddenLayer_2['biases'])
	l_2 = tf.nn.relu(l_2)

	l_3 = tf.add(tf.matmul(l_2, hiddenLayer_3['weights']), hiddenLayer_3['biases'])
	l_3 = tf.nn.relu(l_3)

	l_4 = tf.add(tf.matmul(l_3, hiddenLayer_4['weights']), hiddenLayer_4['biases'])
	l_4 = tf.nn.relu(l_4)

	output = tf.add(tf.matmul(l_4, outputLayer['weights']), outputLayer['biases'])

	return output


def trainNN(x):
	prediction = neuralNetworkModel(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hmEpochs = 20

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# Train the model and save the model to disk as a model.ckpt file
		# file is stored in the same directory as this python script is started
		saver = tf.train.Saver()

		for epoch in range(hmEpochs):
			epochLoss = 0
			for _ in range(int(mnist.train.num_examples/batchSize)):
				epochX, epochY = mnist.train.next_batch(batchSize)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epochX, y: epochY})
				epochLoss += c
			print('Epoch ', epoch, 'completed out of ', hmEpochs, 'loss: ', epochLoss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
		save_path = saver.save(sess, "/tmp/model.ckpt")
		print ("Model saved in file: ", save_path)


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    #newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva
    #print(tva)

def useNeuralNet(data):
	prediction = neuralNetworkModel(x)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, '/tmp/model.ckpt')
		print(sess.run(tf.argmax(tf.nn.relu(sess.run(prediction, feed_dict={x: [data]})), 1)))

#trainNN(x)
image = imageprepare('image.png')
useNeuralNet(image)
	