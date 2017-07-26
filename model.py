import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
"""
    learn from https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
    first use class not function to manage our model
    second tempt to change the interface matched with supplement material in the paper
"""

class model():
    def __init__(self,stride_way):
        self.stride_Way = stride_way


    def upsample(self, input_layer, scale):
        """
        按一定的比例进行上采样
        :param input_layer:  上一层的输入
        :param scale:        放缩的比例大小
        :return:  放缩后的层
        """
        shape = tf.shape(input_layer)   #原层每张图像的星族昂
        height = shape[1]               #原层的高度
        width = shape[2]                #原层的宽度
        return tf.image.resize_nearest_neighbor(input_layer, [height * scale, width * scale])


    # TODO read dispnet
    """
        total copy the implement from source code  for there is no clear description
    """
    def get_disp(self, input_layer):
        """
        通过卷基层获取视差
        :param input_layer: 上一层输入
        :return: 视差图像
        """
        disp = 0.3 * self.encoder_conv_single(input_layer, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def encoder_conv_single(self,input_layer, kernel, strides, channel_out, activation_fn=tf.nn.elu):
        """
        七层encoder卷积层的最底层封装
        :param input_layer:  输入层
        :param kernel:  卷积核大小    {int}
        :param strides:   步长大小    {int}
        :param channel_out:  输出的通道数量
        :param activation_fn 论文指出 激活函数使用elu 的效果比relu效果好
        :return:  output
        """
        ext = (kernel-1)//2                               #计算扩展的边数
                                                          #在输入图像或前一层的长和宽分别添加的一定数量的0
        p_x = tf.pad(input_layer, [[0,0], [ext, ext], [ext, ext]], [0,0])    # 为的是使用valid,但是感觉和直接使用same没什么区别啊
        return slim.conv2d(p_x, num_outputs=channel_out, kernel_size=kernel, stride=strides, padding='VALID'
                           , activation_fn=activation_fn)

    def encoder_conv_twin(self, input_layer, channel_out, kernel_size):
        """
        将convx 和 convxb 进行合并 得到的合并卷积
        :param input_layer:   输入的图像或者卷基层
        :param channel_out:  输出的通道数(特征个数) {int}
        :param kernel_size:   卷积核的大小 {int}  这两层的卷积核大小保持一致
        :return: output
        """

        # check the param put in strides (paper and source code are different)
        if self.stride_Way == 'paper':
            first = 2
            second = 3 - first
        else :
            first = 1
            second = 3 - first

        # source code implement
        # conv1 = self.encoder_conv_single(input_layer, channel_out, kernel_size, first)
        # conv2 = self.encoder_conv_single(conv1, channel_out, kernel_size, second)
        #
        # return  conv2

        # implement with new feature of slim
        return slim.stack(input_layer, self.encoder_conv_single ,(channel_out, kernel_size, [first,second]))

    def decoder_upconv(self,input_layer, channel_out, kernel_size, scale):
        """
        上采样卷积层  采用最近邻居的上采样的方式进行卷积
        :param input_layer:  上一层的输入
        :param channel_out:  输出的通道数
        :param kernel_size:  卷积核的大小{int}
        :param scale :       上采样的范围和比例
        :return: 上采样后的layer
        """
        if self.stride_Way == 'paper':
            stride = 2
        else :
            stride = 1
        upsample = self.upsample(input_layer,scale)
        return self.encoder_conv_single(upsample, kernel_size, stride, channel_out)

    #TODO conv2d_transpose 源码
    def decoder_deconv(self,input_layer, chanel_out,  kernel_size, scale):
        """
        反采样卷积层  采用反卷积的方式对图像进行扩大
        :param input_layer:  输入层
        :param chanel_out:   输出通道的个数
        :param kernel_size:  每层卷积核的大小 {int}
        :param scale:        每层放缩的比例
        :return:    反卷积后的layer
        """

        p_x = tf.pad(input_layer, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, chanel_out, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]   #need to check after reading the source code of conv2d_transpose







