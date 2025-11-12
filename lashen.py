import numpy as np

def gamma_trans(img, gamma):
    img=np.power(img/255,gamma)*255
    img_out=np.round(img).astype(np.uint8)
    return img_out


def min_max_trans(img,max,min):
    i_min=np.min(img)
    i_max=np.max(img)
    img=(img-i_min)/(i_max-i_min)
    img=img*(max-min)+min
    img_out=np.round(img).astype(np.uint8)
    return img_out



def percentage_trans(img,max,min):
    i_min=np.min(img)
    i_max=np.max(img)
    img_clip = np.clip(img, i_min + (i_max - i_min) * min, i_min + (i_max - i_min) * (1-max))
    img_stren = ((img_clip - np.min(img_clip)) / (np.max(img_clip) - np.min(img_clip)))*255
    img_stren = img_stren.astype(np.uint8)
    return img_stren
