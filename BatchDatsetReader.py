# coding=utf-8
"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list       # 文件列表
        self.image_options = image_options  # 图片操作方式 resize  224
        self._read_images()

    def _read_images(self):
        self.__channels = True
        # 扫描files字典中所有image 图片全路径
        # 根据文件全路径读取图像，并将其扩充为RGB格式
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False

        # 扫描files字典中所有annotation 图片全路径
        # 根据文件全路径读取图像，并将其扩充为三通道格式
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        print (self.images.shape)
        print (self.annotations.shape)

    def _transform(self, filename):
        # 读取文件图片
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            # 将图片三个通道设置为一样的图片
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:

            resize_size = int(self.image_options["resize_size"])
            # 使用最近邻插值法resize图片
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)       # 返回已经resize的图片

    def get_records(self):
        """
        返回图片和标签全路径
        :return:
        """
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        """
        剩下的batch
        :param offset:
        :return:
        """
        self.batch_offset = offset

    def next_batch(self, batch_size):
        # 当前第几个batch
        start = self.batch_offset
        # 读取下一个batch  所有offset偏移量+batch_size
        self.batch_offset += batch_size
        # iamges存储所有图片信息 images.shape(len, h, w)
        if self.batch_offset > self.images.shape[0]:      # 如果下一个batch的偏移量超过了图片总数 说明完成了一个epoch
            # Finished epoch
            self.epochs_completed += 1      # epochs完成总数+1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])      # arange生成数组(0 - len-1) 获取图片索引
            np.random.shuffle(perm)         # 对图片索引洗牌
            self.images = self.images[perm]     # 洗牌之后的图片顺序
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0           # 下一个epoch从0开始
            self.batch_offset = batch_size  # 已完成的batch偏移量

        end = self.batch_offset             # 开始到结束self.batch_offset   self.batch_offset+batch_size
        return self.images[start:end], self.annotations[start:end]      # 取出batch

    def get_random_batch(self, batch_size):
        # 按照一个batch_size一个块  进行对所有图片总数进行随机操作， 相当于洗牌工作
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
