import math
import numpy as np

'''
assume same padding, relu, no stride
'''
def fpsimulateconv(image, weights, strides):
    print(strides)
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    nift = np.shape(image)[2]
    w0 = weights[0]
    w1 = weights[1]
    nkx = np.shape(w0)[0]
    nky = np.shape(w0)[1]
    noft = np.shape(w0)[3]
    nsx = strides[0]
    nsy = strides[1]
    imageout = np.zeros(((nx - nsx) // nsx + 1, (ny - nsy) // nsy + 1, noft), dtype='float32')
    for x in range(0, nx, nsx):
        for y in range(0, ny, nsy):
            for oft in range(noft):
                for dx in range(nkx):
                    xx = x + dx - (nkx-1)//2
                    if xx < 0 or xx >= nx:
                        continue
                    for dy in range(nky):
                        yy = y + dy - (nky-1)//2
                        if yy < 0 or yy >= ny:
                            continue
                        for ift in range(nift):
                            imageout[x//nsx][y//nsy][oft] += image[xx][yy][ift] * w0[dx][dy][ift][oft]
                imageout[x//nsx][y//nsy][oft] += w1[oft]
    return np.maximum(imageout, 0)

'''
assume epsilon=0.001 for fdiv==True
'''
def fpsimulatenorm(image, weights, fdiv = False):
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    nft = np.shape(image)[2]
    w0 = weights[0]
    w1 = weights[1]
    imageout = np.zeros((nx, ny, nft), dtype='float32')
    if fdiv:
        div = np.float32(math.sqrt(1 + 0.001))
    else:
        div = 1
    for ft in range(nft):
        imageout[:,:,ft] = (image[:,:,ft] / div) * w0[ft] + w1[ft]
    return imageout

'''
specific clipping for this model
'''
def fpsimulateclip(image):
    return np.where(image == 0, image, np.clip(np.floor(image * 8), -32, 31) / 8)

'''
assume valid padding, as much stride
'''
def fpsimulatemaxp(image, pool_size):
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    nft = np.shape(image)[2]
    nkx = pool_size[0]
    nky = pool_size[1]
    imageout = np.zeros(((nx - nkx) // nkx + 1, (ny - nky) // nky + 1, nft), dtype='float32')
    for x in range(0, nx - nkx + 1, nkx):
        for y in range(0, ny - nky + 1, nky):
            for ft in range(nft):
                imageout[x//nkx][y//nky][ft] = np.max(image[x:x+nkx,y:y+nky,ft])
    return imageout

'''
assume valid padding, as much stride
'''
def fpsimulateavep(image, pool_size):
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    nft = np.shape(image)[2]
    nkx = pool_size[0]
    nky = pool_size[1]
    imageout = np.zeros(((nx - nkx) // nkx + 1, (ny - nky) // nky + 1, nft), dtype='float32')
    for x in range(0, nx - nkx + 1, nkx):
        for y in range(0, ny - nky + 1, nky):
            for ft in range(nft):
                imageout[x//nkx][y//nky][ft] = np.average(image[x:x+nkx,y:y+nky,ft])
    return imageout

'''
assume 1d input image
softmax activation if fsoftmax
'''
def fpsimulatedense(image, weights, fsoftmax):
    n = np.shape(image)[0]
    w0 = weights[0]
    w1 = weights[1]
    m = np.shape(w0)[1]
    imageout = np.zeros(m, dtype='float32')
    for j in range(m):
        for i in range(n):
            imageout[j] += image[i] * w0[i][j]
        imageout[j] += w1[j]
    if fsoftmax:
        imageout = np.exp(imageout)/np.sum(np.exp(imageout))
    return imageout

def fpsimulate(layers, image, fdiv = True, fsoftmax = True):
    images = []
    for i, layer in enumerate(layers):
        layername = layer.__class__.__name__
        print(layername)
        if layername == 'Conv2D':
            image = fpsimulateconv(image, layer.get_weights(), layer.strides)
        elif layername == 'BatchNormalization':
            image = fpsimulatenorm(image, layer.get_weights(), fdiv)
        elif layername == 'Lambda':
            image = fpsimulateclip(image)
        elif layername == 'MaxPooling2D':
            image = fpsimulatemaxp(image, layer.pool_size)
        elif layername == 'AveragePooling2D':
            image = fpsimulateavep(image, layer.pool_size)
        elif layername == 'Flatten':
            image = image.flatten()
        elif layername == 'Dense':
            image = fpsimulatedense(image, layer.get_weights(), fsoftmax)
        images.append(image)
    return images
