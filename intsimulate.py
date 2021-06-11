import math
import numpy as np

'''
assume same padding, relu
'''
def intsimulateconv(image, weights, strides):
    cshamt = image[1]
    image = image[0]
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    nift = np.shape(image)[2]
    w0 = np.where(weights[0] != 0, np.log2(abs(weights[0])), 1000).astype('int64')
    shamt = -np.min(w0)
    w0sign = weights[0] > 0
    w0prune = weights[0] == 0
    w1 = weights[1] * (1 << (shamt + cshamt))
    w1 = w1.astype('int64')
    nkx = np.shape(w0)[0]
    nky = np.shape(w0)[1]
    noft = np.shape(w0)[3]
    nsx = strides[0]
    nsy = strides[1]
    nox = (nx + nsx - 1) // nsx
    noy = (ny + nsy - 1) // nsy
    npx = ((nox - 1) * nsx + nkx - nx) // 2
    npy = ((noy - 1) * nsy + nky - ny) // 2
    imageout = np.zeros((nox, noy, noft), dtype='int64')
    for x in range(0, nx, nsx):
        for y in range(0, ny, nsy):
            for oft in range(noft):
                for dx in range(nkx):
                    xx = x + dx - npx
                    if xx < 0 or xx >= nx:
                        continue
                    for dy in range(nky):
                        yy = y + dy - npy
                        if yy < 0 or yy >= ny:
                            continue
                        for ift in range(nift):
                            if w0prune[dx][dy][ift][oft]:
                                continue
                            val = image[xx][yy][ift] << (shamt + w0[dx][dy][ift][oft])
                            if w0sign[dx][dy][ift][oft]:
                                imageout[x//nsx][y//nsy][oft] += val
                            else:
                                imageout[x//nsx][y//nsy][oft] -= val
                imageout[x//nsx][y//nsy][oft] += w1[oft]
    return (np.maximum(imageout, 0), cshamt + shamt)

'''
no fdiv
'''
def intsimulatenorm(image, weights):
    cshamt = image[1]
    image = image[0]
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    nft = np.shape(image)[2]
    w0 = np.log2(abs(weights[0])).astype('int64')
    shamt = -np.min(w0)
    w0sign = weights[0] > 0
    w1 = weights[1] * (1 << (shamt + cshamt))
    w1 = w1.astype('int64')
    imageout = np.zeros((nx, ny, nft), dtype='int64')
    for ft in range(nft):
        val = image[:,:,ft] << (shamt + w0[ft])
        if w0sign[ft]:
            imageout[:,:,ft] = val
        else:
            imageout[:,:,ft] = -val
        imageout[:,:,ft] += w1[ft]
    return (imageout, cshamt + shamt)

'''
specific clipping for this model
'''
def intsimulateclip(image):
    cshamt = image[1]
    image = image[0]
    return (np.where(image > (31 << (cshamt - 3)), 31, np.where(image < -(32 << (cshamt - 3)), -32, image >> (cshamt - 3))), 3)

'''
assume valid padding, as much stride
'''
def intsimulatemaxp(image, pool_size):
    cshamt = image[1]
    image = image[0]
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    nft = np.shape(image)[2]
    nkx = pool_size[0]
    nky = pool_size[1]
    imageout = np.zeros(((nx - nkx) // nkx + 1, (ny - nky) // nky + 1, nft), dtype='int64')
    for x in range(0, nx - nkx + 1, nkx):
        for y in range(0, ny - nky + 1, nky):
            for ft in range(nft):
                imageout[x//nkx][y//nky][ft] = np.max(image[x:x+nkx,y:y+nky,ft])
    return (imageout, cshamt)

'''
assume valid padding, as much stride
'''
def intsimulateavep(image, pool_size):
    cshamt = image[1]
    image = image[0]
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    nft = np.shape(image)[2]
    nkx = pool_size[0]
    nky = pool_size[1]
    shamt = int(round(math.log2(nkx * nky)))
    imageout = np.zeros(((nx - nkx) // nkx + 1, (ny - nky) // nky + 1, nft), dtype='int64')
    for x in range(0, nx - nkx + 1, nkx):
        for y in range(0, ny - nky + 1, nky):
            for ft in range(nft):
                imageout[x//nkx][y//nky][ft] += np.sum(image[x:x+nkx,y:y+nky,ft])
    return (imageout, cshamt + shamt)

'''
assume 1d input image
ignore activatoin function
'''
def intsimulatedense(image, weights):
    cshamt = image[1]
    image = image[0]
    n = np.shape(image)[0]
    w0 = np.where(weights[0] != 0, np.log2(abs(weights[0])), 1000).astype('int64')
    shamt = -np.min(w0)
    w0sign = weights[0] > 0
    w0prune = weights[0] == 0
    w1 = weights[1] * (1 << (shamt + cshamt))
    w1 = w1.astype('int64')
    m = np.shape(w0)[1]
    imageout = np.zeros(m, dtype='int64')
    for j in range(m):
        for i in range(n):
            if w0prune[i][j]:
                continue
            val = image[i] << (shamt + w0[i][j])
            if w0sign[i][j]:
                imageout[j] += val
            else:
                imageout[j] -= val
        imageout[j] += w1[j]
    return (imageout, cshamt + shamt)

def intsimulate(layers, image):
    images = []
    for i, layer in enumerate(layers):
        layername = layer.__class__.__name__
        print(layername)
        if layername == 'Conv2D':
            image = intsimulateconv(image, layer.get_weights(), layer.strides)
        elif layername == 'BatchNormalization':
            image = intsimulatenorm(image, layer.get_weights())
        elif layername == 'Lambda':
            image = intsimulateclip(image)
        elif layername == 'MaxPooling2D':
            image = intsimulatemaxp(image, layer.pool_size)
        elif layername == 'AveragePooling2D':
            image = intsimulateavep(image, layer.pool_size)
        elif layername == 'Flatten':
            image = (image[0].flatten(), image[1])
        elif layername == 'Dense':
            image = intsimulatedense(image, layer.get_weights())
        images.append(image)
    return images
