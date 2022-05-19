import numpy as np
import matplotlib.pyplot as plt
from .utils.tf import K, IS_TF_1, tf

if IS_TF_1:
    from skimage.measure import compare_psnr, compare_ssim
else:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    
    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
    
    return flops.total_float_ops  # Prints the "flops" of the model.


def compute_psnr_and_ssim(image1, image2, border_size=0):
    """
    Computes PSNR and SSIM index from 2 images.
    We round it and clip to 0 - 255. Then shave 'scale' pixels from each border.
    """
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)
    
    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None
    
    if border_size > 0:
        image1 = image1[border_size:-border_size, border_size:-border_size, :]
        image2 = image2[border_size:-border_size, border_size:-border_size, :]
    
    if IS_TF_1:
        psnr = compare_psnr(image1, image2, data_range=255)
        ssim = compare_ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                            sigma=1.5, data_range=255)
    else:
        (ssim, diff) = structural_similarity(image1, image2, win_size=11, gaussian_weights=True, multichannel=True,
                                             data_range=255, K1=0.01, K2=0.03, sigma=1.5)  # , full=True)
        
        psnr = peak_signal_noise_ratio(image1, image2, data_range=255)
    
    return psnr, ssim


def norm(x):
    # input any range
    # Normalized to 0~1
    max = np.max(x)
    # if max < 1e-4:
    #     max = 0.0
    min = np.min(x)
    if max == min:
        normx = x
    else:
        normx = (x - min) / (max - min)
    return normx


def to_color(arr, pmin=1, pmax=99.8, gamma=1., colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """Converts a 2D or 3D stack to a colored image (maximal 3 channels).

    Parameters
    ----------
    arr : numpy.ndarray
        2D or 3D input data
    pmin : float
        lower percentile, pass -1 if no lower normalization is required
    pmax : float
        upper percentile, pass -1 if no upper normalization is required
    gamma : float
        gamma correction
    colors : list
        list of colors (r,g,b) for each channel of the input

    Returns
    -------
    numpy.ndarray
        colored image
    """
    if not arr.ndim in (2, 3):
        raise ValueError("only 2d or 3d arrays supported")
    
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    
    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)
    
    out = np.zeros(arr.shape[1:] + (3,))
    
    eps = 1.e-20
    if pmin >= 0:
        mi = np.percentile(arr, pmin, axis=(1, 2), keepdims=True)
    else:
        mi = 0
    
    if pmax >= 0:
        ma = np.percentile(arr, pmax, axis=(1, 2), keepdims=True)
    else:
        ma = 1. + eps
    
    arr_norm = (1. * arr - mi) / (ma - mi + eps)
    
    for i_stack, col_stack in enumerate(colors):
        if i_stack >= len(arr):
            break
        for j, c in enumerate(col_stack):
            out[..., j] += c * arr_norm[i_stack]
    
    return np.clip(out, 0, 1)


def savecolorim(save, im, norm=True, **imshow_kwargs):
    # im: Uint8
    imshow_kwargs['cmap'] = 'magma'
    if not norm:  # 不对当前图片归一化处理，直接保存
        imshow_kwargs['vmin'] = 0
        imshow_kwargs['vmax'] = 255
    
    im = np.asarray(im)
    im = np.stack(map(to_color, im)) if 1 < im.shape[-1] <= 3 else im
    ndim_allowed = 2 + int(1 <= im.shape[-1] <= 3)
    proj_axis = tuple(range(1, 1 + max(0, im[0].ndim - ndim_allowed)))
    im = np.max(im, axis=proj_axis)
    
    # plt.imshow(im, **imshow_kwargs)
    # cb = plt.colorbar(fraction=0.05, pad=0.05)
    # cb.ax.tick_params(labelsize=23)  # 设置色标刻度字体大小。
    # # font = {'size': 16}
    # # cb.set_label('colorbar_title', fontdict=font)
    # plt.show()
    
    plt.imsave(save, im, **imshow_kwargs)


def savecolorim1(save, im, **imshow_kwargs):
    # save 方差 另一种颜色
    imshow_kwargs['cmap'] = 'cividis'
    
    im = np.asarray(im)
    im = np.stack(map(to_color, im)) if 1 < im.shape[-1] <= 3 else im
    ndim_allowed = 2 + int(1 <= im.shape[-1] <= 3)
    proj_axis = tuple(range(1, 1 + max(0, im[0].ndim - ndim_allowed)))
    im = np.max(im, axis=proj_axis)
    plt.rc('font', family='Times New Roman')
    
    # plt.imshow(im, **imshow_kwargs)
    # cb = plt.colorbar(fraction=0.05, pad=0.05)
    # cb.ax.tick_params(labelsize=23)  # 设置色标刻度字体大小。
    # # font = {'size': 16}
    # # cb.set_label('colorbar_title', fontdict=font)
    # plt.show()
    
    plt.imsave(save, im, **imshow_kwargs)


import cv2
from PIL import Image


def savecolor_CV(save, y):
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(y, alpha=15), cv2.COLORMAP_JET)
    im = Image.fromarray(im_color)
    im.save(save)
