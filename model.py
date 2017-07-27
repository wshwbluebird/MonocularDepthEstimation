import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from bilinear_sample import bilinear_sampler_1d_h

"""
    learn from https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
    first use class not function to manage our model
    second tempt to change the interface matched with supplement material in the paper
"""


# TODO 损失函数
# TODO 输出函数
# TODO 统计函数


class Model():
    def __init__(self,  params, mode, left, right, reuse_variables=None, model_index=0):
        """
        :param params: 传进来参数的值
        :param mode:   train or test
        :param left:   Il
        :param right:  Ir
        :param reuse_variables:  哪些变量是重用的（这块还有一点没有看懂）
        :param model_index:      模型序号{int}
        """
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()

    def upsample(self, input_layer, scale):
        """
        按一定的比例进行上采样
        :param input_layer:  上一层的输入
        :param scale:        放缩的比例大小
        :return:  放缩后的层
        """
        shape = tf.shape(input_layer)  # 原层每张图像的星族昂
        height = shape[1]  # 原层的高度
        width = shape[2]  # 原层的宽度
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

    def encoder_conv_single(self, input_layer, kernel, strides, channel_out, activation_fn=tf.nn.elu):
        """
        七层encoder卷积层的最底层封装
        :param input_layer:  输入层
        :param kernel:  卷积核大小    {int}
        :param strides:   步长大小    {int}
        :param channel_out:  输出的通道数量
        :param activation_fn 论文指出 激活函数使用elu 的效果比relu效果好
        :return:  output
        """
        ext = (kernel - 1) // 2  # 计算扩展的边数
        # 在输入图像或前一层的长和宽分别添加的一定数量的0
        p_x = tf.pad(input_layer, [[0, 0], [ext, ext], [ext, ext]], [0, 0])  # 为的是使用valid,但是感觉和直接使用same没什么区别啊
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
        if self.params.stride_Way == 'paper':
            first = 2
            second = 3 - first
        else:
            first = 1
            second = 3 - first

        # source code implement
        # conv1 = self.encoder_conv_single(input_layer, channel_out, kernel_size, first)
        # conv2 = self.encoder_conv_single(conv1, channel_out, kernel_size, second)
        #
        # return  conv2

        # implement with new feature of slim
        return slim.stack(input_layer, self.encoder_conv_single, (channel_out, kernel_size, [first, second]))

    def decoder_upconv(self, input_layer, channel_out, kernel_size, scale):
        """
        上采样卷积层  采用最近邻居的上采样的方式进行卷积
        :param input_layer:  上一层的输入
        :param channel_out:  输出的通道数
        :param kernel_size:  卷积核的大小{int}
        :param scale :       上采样的范围和比例
        :return: 上采样后的layer
        """
        if self.params.stride_Way == 'paper':
            stride = 2
        else:
            stride = 1
        upsample = self.upsample(input_layer, scale)
        return self.encoder_conv_single(upsample, kernel_size, stride, channel_out)

    # TODO conv2d_transpose 源码
    def decoder_deconv(self, input_layer, channel_out, kernel_size, scale):
        """
        反采样卷积层  采用反卷积的方式对图像进行扩大
        :param input_layer:  输入层
        :param channel_out:   输出通道的个数
        :param kernel_size:  每层卷积核的大小 {int}
        :param scale:        每层放缩的比例
        :return:    反卷积后的layer
        """

        p_x = tf.pad(input_layer, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, channel_out, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]  # need to check after reading the source code of conv2d_transpose

    """
        参考 https://arxiv.org/abs/1512.03385
        原论文实现的ImageNet 和 shortcut    block的计算采用的是B
        实现 参考的还是 UMDE 的source code
    """

    def encoder_res_conv(self, input_layer, channel_out, strides, active_fn_final=tf.nn.elu):
        """
        本质是 ImageNet 一个包含shortcut的block（Residual learning: a building block）
        用于实现 1x1 3x3 1x1   + shortcut的模型
        :param input_layer:   上一输入层
        :param channel_out:   输出的参考的通道数(在这个函数中,输出的通道数是参考通道数的四倍)
        :param strides:       步长  如果使用 identity shortcuts (Eqn.(1)）步长为1  需要扩大 则步长为2
        :param active_fn_final  默认最后一步的激活函数
        :return:  经过缩小后的4*channel_out的卷基层
        """
        # judge whether use projection
        # TODO 原论文的实现是  tf.shape(input_layer)[3] !=  channel_out 但是个人认为应该*4  附原论文说明
        """
             The identity shortcuts (Eqn.(1)) can be directly used when the input and output are of the same dimensions
              (solid line shortcuts in Fig. 3).
        """
        is_project = tf.shape(input_layer)[3] != 4 * channel_out or strides == 2

        # 实现 1x1 3x3 1x1
        shortcut = []
        conv1 = self.encoder_conv_single(input_layer, channel_out, 1, 1)
        conv2 = self.encoder_conv_single(conv1, channel_out, 3, strides)
        conv3 = self.encoder_conv_single(conv2, 4 * channel_out, 1, 1, None)  # 最后一个none 制定不需要激活函数

        if is_project:
            # use projection
            shortcut = self.encoder_conv_single(input_layer, 4 * channel_out, 1, strides, None)
        else:
            # use identity
            shortcut = input_layer

        return active_fn_final(conv3 + shortcut)

    """
       实现参考为 https://arxiv.org/abs/1512.03385 的Table1
    """

    def encoder_res_block(self, input_layer, channel_out, num_blocks):
        """
        resnet 的下采样卷积函数
        :param input_layer: 上一输入层
        :param channel_out: 参考输出通道数 （在这个函数中,输出的通道数是参考通道数的四倍)
        :param num_blocks:  在这一个卷积中 有多少个block 数据来源为  https://arxiv.org/abs/1512.03385  Table1
        :return:  conv下采样卷积结果
        """
        out = input_layer
        for i in range(num_blocks - 1):
            out = self.encoder_res_conv(out, channel_out, 1)
        out = self.encoder_res_conv(out, channel_out, 2)
        return out

    def maxpool(self, input_layer, kernel_size):
        """
        最大池化层
        :param input_layer:  上一层输入
        :param kernel_size:   池化的核大小
        :return:  池化后的结果
        """
        p = (kernel_size - 1) // 2
        p_x = tf.pad(input_layer, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    """
        encoder 部分用VGG 网络建立
    """

    def build_vgg(self):
        # set convenience functions
        conv = self.encoder_conv_single
        """
           same with source code
        """
        if self.params.use_deconv:
            upconv = self.decoder_deconv
        else:
            upconv = self.decoder_upconv

        with tf.variable_scope('encoder'):
            """
             缩小层 特征提取
             通道数逐渐增加,图片大小逐渐减少
             最终返回的结果是 conv7b
            """
            conv1 = self.encoder_conv_twin(self.model_input, 32, 7)  # H/2
            conv2 = self.encoder_conv_twin(conv1, 64, 5)  # H/4
            conv3 = self.encoder_conv_twin(conv2, 128, 3)  # H/8
            conv4 = self.encoder_conv_twin(conv3, 256, 3)  # H/16
            conv5 = self.encoder_conv_twin(conv4, 512, 3)  # H/32
            conv6 = self.encoder_conv_twin(conv5, 512, 3)  # H/64
            conv7 = self.encoder_conv_twin(conv6, 512, 3)  # H/128

        with tf.variable_scope('skips'):
            """
                copy the convx to skipx
                why the scope'encoder' do not use stack
            """
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7, 512, 3, 2)  # H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    """
         encoder 部分用ImageNet 网络建立
    """

    def build_resnet50(self):
        # set convenience functions
        conv = self.encoder_conv_single

        if self.params.use_deconv:
            upconv = self.decoder_deconv
        else:
            upconv = self.decoder_upconv

        with tf.variable_scope('encoder'):

            """
             缩小层 特征提取
             通道数逐渐增加,图片大小逐渐减少
             最终返回的结果是 conv5
             实现模型  https://arxiv.org/abs/1512.03385 table3  第三列

            """

            conv1 = conv(self.model_input, 64, 7, 2)  # H/2  -   64D
            pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
            conv2 = self.encoder_res_block(pool1, 64, 3)  # H/8  -  256D
            conv3 = self.encoder_res_block(conv2, 128, 4)  # H/16 -  512D
            conv4 = self.encoder_res_block(conv3, 256, 6)  # H/32 - 1024D
            conv5 = self.encoder_res_block(conv4, 512, 3)  # H/64 - 2048D

        """
            copy the convx to skipx
        """
        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            """
                just copy from the source code
            """

            upconv6 = upconv(conv5, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)


        # ###############################################net####################################################################

    def scale_pyramid(self, img, num_scales):
        """
         图像缩放金字塔
        :param img:   原图像
        :param num_scales:  缩小到别率 {int} 若为3 则返回搜小 1,2,4,8倍的列表
        :return:  缩放后的图像列表 {list}
        """
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]

        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = int(h / ratio)
            nw = int(w / ratio)
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def SSIM(self, x, y):
        """
        求出两张图像的结构相似性  论文上指出的观察窗口 为3*3
        :param x:  输入图像x
        :param y:  输入图像y
        :return:   两张图像的结构相似性
        """

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        """
            分块的均值
        """
        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')
        """
            分块的方差
        """
        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2

        """
            分块的协方差
        """
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)  # 将（1-ssim)/2 的值 划分到0-1之间

    """
        获取图像宽的梯度
    """
    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    """
        获取图像高的梯度
    """
    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def get_disparity_smoothness(self, disp, pyramid):
        """
        孙水函数
        :param disp:       多张视差图像
        :param pyramid:    多张大小转换图像
        :return:           损失函数值
        """
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    """
       根据参数制造基本的模型
    """
    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid = self.scale_pyramid(self.left, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                """
                    判断是否是双目图像的输入
                """
                if self.params.do_stereo:
                    """
                        从第三个维度进行连接（宽）
                    """
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left

                # build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):

            # TODO disp的意义 是什么我又忘了
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            """
                将disp扩充为标准的tensor形式（可输入卷积层的那种）
                用同样的输出分别扩充成左视差和右视差
            """
            self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

        """
            At test time, our network predicts the disparity at the finest scale level for the left image dl,
            which has the same resolution as the input image.
        """
        if self.mode == 'test':
            return

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            """
                 用原放缩图和视差图生成新的原图
            """
            self.left_est = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):

            """
                 用右视差图和做视差图生成左图
            """
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in
                                       range(4)]

            """
                 用左视差图和右视差图生成左图
            """
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in
                                       range(4)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)


    def build_losses(self):
        """
        分别计算三种不同的三种代价函数
        包括： 图像整体的外观
              图像深度的连续型  也就是平滑性
              左右视差的一致性（这块对于为什么期待代价函数最小或者回归为0 还有待商榷）
        :return:
        """
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [
                self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left = [
                self.params.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_left[i] for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY
            self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in
                                 range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in
                                  range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss \
                              + self.params.lr_loss_weight * self.lr_loss


    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i], collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)
