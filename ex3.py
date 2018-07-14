# load MNIST data
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# start tensorflow interactiveSession
import tensorflow as tf

#卷积神经网络对MNSIT手写数字集进行识别
#两层卷积神经网络
#第一层卷积神经网络5*5+32
#第二层卷积神经网络5*5+64

sess = tf.InteractiveSession()  #相对于session更加灵活

with tf.name_scope("Input"):
    x = tf.placeholder("float",shape=[None,784])    #输入层神经元
    y_ = tf.placeholder("float",shape=[None,10])    #输出层神经元

def weight_variable(shape):
    #权重初始化函数
    initial = tf.truncated_normal(shape,stddev = 0.1)   #产生一个正态分布初始化，stddev为方差
    return tf.Variable(initial)
def bias_variable(shape):
    #偏置初始化函数
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)
def conv2d(x,W):
    #卷积函数
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')
def max_pool_2x2(x):
    #弛化层max_pooling
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
with tf.name_scope("Inference"):
    w_conv1 = weight_variable([5,5,1,32])   #前两个参数是卷积核的大小，第三个参数是图像通道数目，最后一个参数是特征数目？
    b_conv1 = bias_variable([32])   #卷积层的偏置
    x_image = tf.reshape(x,[-1,28,28,1])    #输入图像变成一个4d向量，2、3维对应图片的宽、高，最后一维代表图片的颜色通道数
    h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)   #第一层卷积层，对输入图像进行卷积并且进行非线性映射
    h_pool1 = max_pool_2x2(h_conv1)     #第一层弛化层，对应卷积神经网络中的弛化层，相当于特征选择
    tf.summary.histogram('b1',b_conv1)
    tf.summary.histogram('w1',w_conv1)
    tf.summary.image('w1',x_image)
    # tf.summary.distribution('w1',w_conv1)
    w_conv2 = weight_variable([5,5,32,64])  #第二层的卷积层，5*5的卷积核，特征数量64
    b_conv2 = bias_variable([64])   #卷积层的偏置64
    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)   #第二层卷积层，对原始图像进行卷及操作并且进行非线性映射
    h_pool2 = max_pool_2x2(h_conv2)     #第二层弛化层，弛化层操作
    tf.summary.histogram('b2',b_conv2)
    #密集连接层
    w_fc1 = weight_variable([7*7*64,1024])      #上一层弛化层输出的张量是7*7，不知道为什么就变成了7*7的图像
    b_fc1 = bias_variable([1024])       #一个1024个神经元的全连接层
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)     #全连接层非线性映射
    tf.summary.histogram('b3',b_fc1)
with tf.name_scope("Drop"):
    #加入dropout层
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
with tf.name_scope("Softmax"):
    #添加一个softmax层
    w_fc2 = weight_variable([1024,10])  #softmax层
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)   #获得最后的输出结果
    tf.summary.histogram('b4',b_fc2)
    tf.summary.histogram('w4',w_fc2)
#训练过程
with tf.name_scope("Loss"):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))   #计算交叉熵
    tf.summary.scalar('cross_entropy',cross_entropy)
with tf.name_scope("Train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy',accuracy)
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./ex3",sess.graph)

for i in range(2000):
    batch = mnist.train.next_batch(50)    #每次获得50幅图像
    test = mnist.test.next_batch(400)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i%10 == 0:
        # print(sess.run(accuracy, feed_dict={x: test[0], y_: test[1], keep_prob: 1.0}))
        sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        result = sess.run(merged,feed_dict={x:test[0],y_:test[1],keep_prob:1.0})
        writer.add_summary(result,i)


        # print(sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        # print("step %d, training accuracy %g"%(i,train_accuracy))
writer.close()




