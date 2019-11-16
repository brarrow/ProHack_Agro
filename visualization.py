import image
import numpy as np


def vis_segmentation_cv(resized_im=None, res_img=None):
    image.show(res_img)
    # if resized_im is not None:
    #     blended_img = blend_img_and_mask(resized_im, res_img)
    #     image.show(blended_img)
    # else:
    #     image.show(res_img)


def blend_img_and_mask(resized_im, res_img):
    color_mask = np.zeros_like(resized_im)
    color_mask.fill(40)
    blended_img = resized_im.copy()
    seg_mask = res_img.copy()
    seg_mask = [[1 if max(pix) > 0 else 0 for pix in line]for line in seg_mask]
    blended_img[seg_mask] = color_mask[seg_mask]
    return blended_img
