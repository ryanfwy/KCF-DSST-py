import numpy as np
import cv2
import fhog

import sys
PY3 = sys.version_info >= (3,)
 
if PY3:
    xrange = range


# ffttools
# 离散傅里叶变换、逆变换
def fftd(img, backwards=False, byRow=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    # return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))  # 'flags =' is necessary!
    # DFT_INVERSE: 用一维或二维逆变换取代默认的正向变换,
    # DFT_SCALE: 缩放比例标识符，根据数据元素个数平均求出其缩放结果，如有N个元素，则输出结果以1/N缩放输出，常与DFT_INVERSE搭配使用。 
    # DFT_COMPLEX_OUTPUT: 对一维或二维的实数数组进行正向变换，这样的结果虽然是复数阵列，但拥有复数的共轭对称性

    if byRow:
        return cv2.dft(np.float32(img), flags=(cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT))
    else:
        return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))

# 实部图像
def real(img):
    return img[:, :, 0]

# 虚部图像
def imag(img):
    return img[:, :, 1]

# 两个复数，它们的积 (a+bi)(c+di)=(ac-bd)+(ad+bc)i
def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)
 
    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res

# 两个复数，它们相除 (a+bi)/(c+di)=(ac+bd)/(c*c+d*d) +((bc-ad)/(c*c+d*d))i
def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)
 
    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res

def complexDivisionReal(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / b
 
    res[:, :, 0] = a[:, :, 0] * divisor
    res[:, :, 1] = a[:, :, 1] * divisor
    return res

# 可以将 FFT 输出中的直流分量移动到频谱的中央
def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))
 
    assert (img.ndim == 2) # 断言，必须为真，否则抛出异常；ndim 为数组维数
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2 # shape[0] 为行，shape[1] 为列
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_



# recttools
# rect = {x, y, w, h}
# x 右边界
def x2(rect):
    return rect[0] + rect[2]

# y 下边界
def y2(rect):
    return rect[1] + rect[3]

# 限宽、高
def limit(rect, limit):
    if rect[0] + rect[2] > limit[0] + limit[2]:
        rect[2] = limit[0] + limit[2] - rect[0]
    if rect[1] + rect[3] > limit[1] + limit[3]:
        rect[3] = limit[1] + limit[3] - rect[1]
    if rect[0] < limit[0]:
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if rect[1] < limit[1]:
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect

# 取超出来的边界
def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res

# 经常需要空域或频域的滤波处理，在进入真正的处理程序前，需要考虑图像边界情况。
# 通常的处理方法是为图像增加一定的边缘，以适应 卷积核 在原图像边界的操作。
def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]
 
    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res

def cutOutsize(num, limit):
    if num < 0: num = 0
    elif num > limit - 1: num = limit - 1
    return int(num)

def extractImage(img, cx, cy, patch_width, patch_height):
    xs_s = np.floor(cx) - np.floor(patch_width / 2)
    xs_s = cutOutsize(xs_s, img.shape[1])

    xs_e = np.floor(cx + patch_width - 1) - np.floor(patch_width / 2)
    xs_e = cutOutsize(xs_e, img.shape[1])

    ys_s = np.floor(cy) - np.floor(patch_height / 2)
    ys_s = cutOutsize(ys_s, img.shape[0])

    ys_e = np.floor(cy + patch_height - 1) - np.floor(patch_height / 2)
    ys_e = cutOutsize(ys_e, img.shape[0])

    return img[ys_s:ys_e, xs_s:xs_e]
 


# KCF tracker
class KCFTracker:
    def __init__(self, hog=False, fixed_window=True, multi_scale=False):
        self.lambdar = 0.0001 # regularization; 正则化
        self.padding = 2.5 # extra area surrounding the target; 目标扩展出来的区域
        self.output_sigma_factor = 0.125 # bandwidth of gaussian target; 高斯目标的带宽

        self._multiscale = multi_scale
        if multi_scale:
            self.template_size = 96 # 模板大小，在计算_tmpl_sz时，较大边长被归一成96，而较小边长按比例缩小

            self.scale_padding = 1.0
            self.scale_step = 1.05 # default: 1.02，多尺度估计的时候的尺度步长
            self.scale_sigma_factor = 0.25
            self.n_scales = 33 # default: 33，尺度估计器样本数
            self.scale_lr = 0.025
            self.scale_max_area = 512
            self.scale_lambda = 0.01

            if hog == False:
                print('HOG feature is forced to turn on.')

        elif fixed_window:
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._hogfeatures = True if hog or multi_scale else False
        if self._hogfeatures: # HOG feature
            # VOT
            self.interp_factor = 0.012 # linear interpolation factor for adaptation; 自适应的线性插值因子
            self.sigma = 0.6 # gaussian kernel bandwidth; 高斯卷积核带宽
            # TPAMI   #interp_factor = 0.02   #sigma = 0.5
            self.cell_size = 4 # HOG cell size; HOG元胞数组尺寸

            print('Numba Compiler initializing, wait for a while.')
            
        else: # raw gray-scale image # aka CSK tracker
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False
 
        self._tmpl_sz = [0, 0]
        self._roi = [0., 0., 0., 0.]
        self.size_patch = [0, 0, 0]
        self._scale = 1.
        self._alphaf = None # numpy.ndarray (size_patch[0], size_patch[1], 2)
        self._prob = None # numpy.ndarray (size_patch[0], size_patch[1], 2)
        self._tmpl = None # numpy.ndarray raw: (size_patch[0], size_patch[1]) hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None # numpy.ndarray raw: (size_patch[0], size_patch[1]) hog: (size_patch[2], size_patch[0]*size_patch[1])

        # Scale properties
        self.currentScaleFactor = 1
        self.base_width = 0 # initial ROI widt
        self.base_height = 0 # initial ROI height
        self.scaleFactors = None # all scale changing rate, from larger to smaller with 1 to be the middle
        self.scale_model_width = 0 # the model width for scaling
        self.scale_model_height = 0 # the model height for scaling
        self.min_scale_factor = 0. # min scaling rate
        self.max_scale_factor = 0. # max scaling rate
        
        # self._num = None
        # self._den = None

        self.sf_den = None
        self.sf_num = None

        self.s_hann = None
        self.ysf = None


    #################
    ### 位置估计器 ###
    #################

    # 计算一维亚像素峰值
    def subPixelPeak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)

    # 初始化hanning窗口，函数只在第一帧被执行
    # 目的是采样时为不同的样本分配不同的权重，0.5*0.5 是用汉宁窗归一化[0,1]，得到矩阵的值就是每样样本的权重
    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]
 
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t
 
        if self._hogfeatures:
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
            #相当于把1D汉宁窗复制成多个通道
        else:
            self.hann = hann2d
        
        self.hann = self.hann.astype(np.float32)

    # 创建高斯峰函数，函数只在第一帧的时候执行（高斯响应）
    def createGaussianPeak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    # 使用带宽SIGMA计算高斯卷积核以用于所有图像X和Y之间的相对位移
    # 必须都是MxN大小。二者必须都是周期的（即，通过一个cos窗口进行预处理）
    def gaussianCorrelation(self, x1, x2):
        if self._hogfeatures:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in xrange(self.size_patch[2]):
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                caux = real(fftd(caux, True))
                # caux = rearrange(caux)
                c += caux
            c = rearrange(c)
        else:
            # 'conjB=' is necessary!在做乘法之前取第二个输入数组的共轭.
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)
 
        if x1.ndim == 3 and x2.ndim == 3:
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif x1.ndim == 2 and x2.ndim == 2:
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
 
        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))
 
        return d

    # 使用第一帧和它的跟踪框，初始化KCF跟踪器
    def init(self, roi, image):
        self._roi = list(map(float,roi))
        assert (roi[2] > 0 and roi[3] > 0)

        # _tmpl是截取的特征的加权平均
        self._tmpl = self.getFeatures(image, 1)
        # _prob是初始化时的高斯响应图
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        # _alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)

        if self._multiscale:
            self.dsstInit(self._roi, image)

        self.train(self._tmpl, 1.0)

    # 从图像得到子窗口，通过赋值填充并检测特征
    def getFeatures(self, image, inithann, scale_adjust=1.):
        extracted_roi = [0, 0, 0, 0]
        cx = self._roi[0] + self._roi[2] / 2
        cy = self._roi[1] + self._roi[3] / 2
 
        if inithann:
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding
 
            if self.template_size > 1:
                # 把最大的边缩小到96，_scale是缩小比例
                # _tmpl_sz是滤波模板的大小也是裁剪下的PATCH大小
                if padded_w >= padded_h:
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.
 
            if self._hogfeatures:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2
 
        # 选取从原图中扣下的图片位置大小
        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0] * self.currentScaleFactor)
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1] * self.currentScaleFactor)

        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        # z是当前帧被裁剪下的搜索区域
        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]: # 缩小到96
            z = cv2.resize(z, tuple(self._tmpl_sz))
 
        if self._hogfeatures:
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)
            # size_patch为列表，保存裁剪下来的特征图的【长，宽，通道】
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
            FeaturesMap = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1], self.size_patch[2])).T # (size_patch[2], size_patch[0]*size_patch[1])

        else: # 将RGB图变为单通道灰度图
            if z.ndim == 3 and z.shape[2] == 3:
                FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            elif z.ndim == 2:
                FeaturesMap = z
            
            # 从此FeatureMap从-0.5到0.5
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
            # size_patch为列表，保存裁剪下来的特征图的【长，宽，1】
            self.size_patch = [z.shape[0], z.shape[1], 1]
 
        if inithann:
            self.createHanningMats()
 
        FeaturesMap = self.hann * FeaturesMap # 加汉宁（余弦）窗减少频谱泄露
        return FeaturesMap

    # 使用当前图像的检测结果进行训练
    # x是当前帧当前尺度下的特征， train_interp_factor是interp_factor
    def train(self, x, train_interp_factor):
        k = self.gaussianCorrelation(x, x)
        # alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
        # _prob是初始化时的高斯响应图，相当于y
        alphaf = complexDivision(self._prob, fftd(k) + self.lambdar)

        # _tmpl是截取的特征的加权平均
        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x
        # _alphaf是频域中相关滤波模板的加权平均
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf

    # 检测当前帧的目标
    # z是前一帧的训练/第一帧的初始化结果，x是当前帧当前尺度下的特征，peak_value是检测结果峰值
    def detect(self, z, x):
        k = self.gaussianCorrelation(x, z)
        # 得到响应图
        res = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))

        # pv:响应最大值 pi:相应最大点的索引数组
        _, pv, _, pi = cv2.minMaxLoc(res)
        # 得到响应最大的点索引的float表示
        p = [float(pi[0]), float(pi[1])]

        # 使用幅值做差来定位峰值的位置
        if pi[0] > 0 and pi[0] < res.shape[1] - 1:
            p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if pi[1] > 0 and pi[1] < res.shape[0] - 1:
            p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        # 得出偏离采样中心的位移
        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.
        
        # 返回偏离采样中心的位移和峰值
        return p, pv

    # 基于当前帧更新目标位置
    def update(self, image):
        # 修正边界
        if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 1
        if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[3] + 1
        if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 2
        if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 2
 
        # 跟踪框、尺度框中心
        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.
 
        # 尺度不变时检测峰值结果
        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

        # 因为返回的只有中心坐标，使用尺度和中心坐标调整目标框
        # loc是中心相对移动量
        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale * self.currentScaleFactor
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale * self.currentScaleFactor

        # 使用尺度估计
        if self._multiscale:
            if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 1
            if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 1
            if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 2
            if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[3] + 2
            
            # 更新尺度
            scale_pi = self.detect_scale(image)
            self.currentScaleFactor = self.currentScaleFactor * self.scaleFactors[scale_pi[0]]
            if self.currentScaleFactor < self.min_scale_factor:
                self.currentScaleFactor = self.min_scale_factor
            # elif self.currentScaleFactor > self.max_scale_factor:
            #     self.currentScaleFactor = self.max_scale_factor

            self.train_scale(image)

        if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        # 使用当前的检测框来训练样本参数
        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)
 
        return self._roi


    #################
    ### 尺度估计器 ###
    #################

    def computeYsf(self):
        scale_sigma2 = (self.n_scales / self.n_scales ** 0.5 * self.scale_sigma_factor) ** 2
        _, res = np.ogrid[0:0, 0:self.n_scales]
        ceilS = np.ceil(self.n_scales / 2.0)
        res = np.exp(- 0.5 * (np.power(res + 1 - ceilS, 2)) / scale_sigma2)
        return fftd(res)

    def createHanningMatsForScale(self):
        _, hann_s = np.ogrid[0:0, 0:self.n_scales]
        hann_s = 0.5 * (1 - np.cos(2 * np.pi * hann_s / (self.n_scales - 1)))
        return hann_s

    # 初始化尺度估计器
    def dsstInit(self, roi, image):
        self.base_width = roi[2]
        self.base_height = roi[3]
        
        # Guassian peak for scales (after fft)
        self.ysf = self.computeYsf()
        self.s_hann = self.createHanningMatsForScale()

        # Get all scale changing rate
        scaleFactors = np.arange(self.n_scales)
        ceilS = np.ceil(self.n_scales / 2.0)
        self.scaleFactors = np.power(self.scale_step, ceilS - scaleFactors - 1)

        # Get the scaling rate for compressing to the model size
        scale_model_factor = 1.
        if self.base_width * self.base_height > self.scale_max_area:
            scale_model_factor = (self.scale_max_area / (self.base_width * self.base_height)) ** 0.5

        self.scale_model_width = int(self.base_width * scale_model_factor)
        self.scale_model_height = int(self.base_height * scale_model_factor)

        # Compute min and max scaling rate
        self.min_scale_factor = np.power(self.scale_step, np.ceil(np.log((max(5 / self.base_width, 5 / self.base_height) * (1 + self.scale_padding))) / 0.0086))
        self.max_scale_factor = np.power(self.scale_step, np.floor(np.log((min(image.shape[0] / self.base_width, image.shape[1] / self.base_height) * (1 + self.scale_padding))) / 0.0086))

        self.train_scale(image, True)

    # 获取尺度样本
    def get_scale_sample(self, image):
        xsf = None
        for i in range(self.n_scales):
            # Size of subwindow waiting to be detect
            patch_width = self.base_width * self.scaleFactors[i] * self.currentScaleFactor
            patch_height = self.base_height * self.scaleFactors[i] * self.currentScaleFactor

            cx = self._roi[0] + self._roi[2] / 2.
            cy = self._roi[1] + self._roi[3] / 2.

            # Get the subwindow
            im_patch = extractImage(image, cx, cy, patch_width, patch_height)
            if self.scale_model_width > im_patch.shape[1]:
                im_patch_resized = cv2.resize(im_patch, (self.scale_model_width, self.scale_model_height), None, 0, 0, 1)
            else:
                im_patch_resized = cv2.resize(im_patch, (self.scale_model_width, self.scale_model_height), None, 0, 0, 3)

            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            mapp = fhog.getFeatureMaps(im_patch_resized, self.cell_size, mapp)
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)

            if i == 0:
                totalSize = mapp['numFeatures'] * mapp['sizeX'] * mapp['sizeY']
                xsf = np.zeros((totalSize, self.n_scales))

            # Multiply the FHOG results by hanning window and copy to the output
            FeaturesMap = mapp['map'].reshape((totalSize, 1))
            FeaturesMap = self.s_hann[0][i] * FeaturesMap
            xsf[:, i] = FeaturesMap[:, 0]

        return fftd(xsf, False, True)

    # 训练尺度估计器
    def train_scale(self, image, ini=False):
        xsf = self.get_scale_sample(image)

        # Adjust ysf to the same size as xsf in the first time
        if ini:
            totalSize = xsf.shape[0]
            self.ysf = cv2.repeat(self.ysf, totalSize, 1)

        # Get new GF in the paper (delta A)
        new_sf_num = cv2.mulSpectrums(self.ysf, xsf, 0, conjB=True)

        new_sf_den = cv2.mulSpectrums(xsf, xsf, 0, conjB=True)
        new_sf_den = cv2.reduce(real(new_sf_den), 0, cv2.REDUCE_SUM)

        if ini:
            self.sf_den = new_sf_den
            self.sf_num = new_sf_num
        else:
            # Get new A and new B
            self.sf_den = cv2.addWeighted(self.sf_den, (1 - self.scale_lr), new_sf_den, self.scale_lr, 0)
            self.sf_num = cv2.addWeighted(self.sf_num, (1 - self.scale_lr), new_sf_num, self.scale_lr, 0)

        self.update_roi()

    # 检测当前图像尺度
    def detect_scale(self, image):
        xsf = self.get_scale_sample(image)

        # Compute AZ in the paper
        add_temp = cv2.reduce(complexMultiplication(self.sf_num, xsf), 0, cv2.REDUCE_SUM)

        # compute the final y
        scale_response = cv2.idft(complexDivisionReal(add_temp, (self.sf_den + self.scale_lambda)), None, cv2.DFT_REAL_OUTPUT)

        # Get the max point as the final scaling rate
        # pv:响应最大值 pi:相应最大点的索引数组
        _, pv, _, pi = cv2.minMaxLoc(scale_response)
        
        return pi

    # 更新尺度
    def update_roi(self):
        # 跟踪框、尺度框中心
        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        # Recompute the ROI left-upper point and size
        self._roi[2] = self.base_width * self.currentScaleFactor
        self._roi[3] = self.base_height * self.currentScaleFactor

        # 因为返回的只有中心坐标，使用尺度和中心坐标调整目标框
        self._roi[0] = cx - self._roi[2] / 2.0
        self._roi[1] = cy - self._roi[3] / 2.0

