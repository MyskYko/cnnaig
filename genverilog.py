import math
import numpy as np

'''
assume same padding, relu
'''
def genverilogconv(image, weights, strides, name):
    cbitwidth = image[2]
    cshamt = image[1]
    nx = image[0][0]
    ny = image[0][1]
    nift = image[0][2]
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
    ninputs = nx * ny * nift
    noutputs = nox * noy * noft
    n = nift * nkx * nky
    m = cbitwidth + shamt
    bitwidth = m + (n+1-1).bit_length()
    f = open(f'{name}.v', mode='w')
    f.write(f'module {name} (\n')
    f.write(f'input [{ninputs*cbitwidth}-1:0] in,\n')
    f.write(f'output [{noutputs*bitwidth}-1:0] out);\n')
    f.write(f'wire signed [{cbitwidth}-1:0] p [0:{ninputs}-1];\n')
    f.write('genvar i;\n')
    f.write(f'generate for(i = 0; i < {ninputs}; i = i + 1) begin : parse\n')
    f.write(f'assign p[i] = {{1\'b0, in[{cbitwidth}*(i+1)-1:{cbitwidth}*i]}};\n')
    f.write('end endgenerate\n')
    f.write(f'wire signed [{bitwidth}-1:0] po [0:{noutputs}-1];\n')
    f.write(f'generate for(i = 0; i < {noutputs}; i = i + 1) begin : parseout\n')
    f.write(f'assign out[{bitwidth}*(i+1)-1:{bitwidth}*i] = po[i] > 0? po[i]: 0;\n')
    f.write('end endgenerate\n')
    for x in range(0, nx, nsx):
        for y in range(0, ny, nsy):
            for oft in range(noft):
                f.write(f'assign po[{oft+(y//nsy)*noft+(x//nsx)*noft*noy}] =')
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
                            d = shamt + w0[dx][dy][ift][oft]
                            sval = f'p[{ift+yy*nift+xx*nift*ny}]'
                            if d != 0:
                                sval = f'{{{sval}, {d}\'d0}}'
                            if w0sign[dx][dy][ift][oft]:
                                f.write(f' + {sval}')
                            else:
                                f.write(f' - {sval}')
                sval = f'{m}\'d{abs(w1[oft])}'
                if w1[oft] > 0:
                    f.write(f' + {sval};\n')
                elif w1[oft] < 0:
                    f.write(f' - {sval};\n')
                else:
                    f.write(' + 0;\n')
    f.write('endmodule\n')
    return ((nox, noy, noft), cshamt + shamt, bitwidth)

'''
no fdiv
'''
def genverilognorm(image, weights, name):
    cbitwidth = image[2]
    cshamt = image[1]
    nx = image[0][0]
    ny = image[0][1]
    nft = image[0][2]
    w0 = np.log2(abs(weights[0])).astype('int64')
    shamt = -np.min(w0)
    w0sign = weights[0] > 0
    w1 = weights[1] * (1 << (shamt + cshamt))
    w1 = w1.astype('int64')
    ninputs = nx * ny * nft
    noutputs = nx * ny * nft
    bitwidth = cbitwidth + shamt + 1
    f = open(f'{name}.v', mode='w')
    f.write(f'module {name} (\n')
    f.write(f'input [{ninputs*cbitwidth}-1:0] in,\n')
    f.write(f'output [{noutputs*bitwidth}-1:0] out);\n')
    f.write(f'wire signed [{cbitwidth}-1:0] p [0:{ninputs}-1];\n')
    f.write('genvar i;\n')
    f.write(f'generate for(i = 0; i < {ninputs}; i = i + 1) begin : parse\n')
    f.write(f'assign p[i] = {{1\'b0, in[{cbitwidth}*(i+1)-1:{cbitwidth}*i]}};\n')
    f.write('end endgenerate\n')
    f.write(f'wire signed [{bitwidth}-1:0] po [0:{noutputs}-1];\n')
    f.write(f'generate for(i = 0; i < {noutputs}; i = i + 1) begin : parseout\n')
    f.write(f'assign out[{bitwidth}*(i+1)-1:{bitwidth}*i] = po[i];\n')
    f.write('end endgenerate\n')
    f.write(f'generate for(i = 0; i < {nx*ny}; i = i + 1) begin : normalize\n')
    for ft in range(nft):
        f.write(f'assign po[{ft}+i*{nft}] =')
        d = shamt + w0[ft]
        sval = f'p[{ft}+i*{nft}]'
        if d != 0:
            sval = f'{{{sval}, {d}\'d0}}'
        if w0sign[ft]:
            f.write(f' + {sval}')
        else:
            f.write(f' - {sval}')
        sval = f'{bitwidth}\'d{abs(w1[ft])}'
        if w1[ft] > 0:
            f.write(f' + {sval};\n')
        elif w1[ft] < 0:
            f.write(f' - {sval};\n')
        else:
            f.write(';\n')
        f.write(f'')
    f.write('end endgenerate\n')
    f.write('endmodule\n')
    return ((nx, ny, nft), cshamt + shamt, bitwidth)

'''
specific clipping for this model
'''
def genverilogclip(image, cliprange, name):
    cbitwidth = image[2]
    cshamt = image[1]
    nx = image[0][0]
    ny = image[0][1]
    nft = image[0][2]
    ninputs = nx * ny * nft
    noutputs = nx * ny * nft
    bitwidth = cliprange[0] + cliprange[1]
    a = cliprange[0]
    b = 1 << (bitwidth - 1)
    f = open(f'{name}.v', mode='w')
    f.write(f'module {name} (\n')
    f.write(f'input [{ninputs*cbitwidth}-1:0] in,\n')
    f.write(f'output [{noutputs*bitwidth}-1:0] out);\n')
    f.write(f'wire signed [{cbitwidth}-1:0] p [0:{ninputs}-1];\n')
    f.write('genvar i;\n')
    f.write(f'generate for(i = 0; i < {ninputs}; i = i + 1) begin : parse\n')
    f.write(f'assign p[i] = {{1\'b0, in[{cbitwidth}*(i+1)-1:{cbitwidth}*i]}};\n')
    f.write('end endgenerate\n')
    f.write(f'wire signed [{bitwidth}-1:0] po [0:{noutputs}-1];\n')
    f.write(f'generate for(i = 0; i < {noutputs}; i = i + 1) begin : parseout\n')
    f.write(f'assign out[{bitwidth}*(i+1)-1:{bitwidth}*i] = po[i];\n')
    f.write('end endgenerate\n')
    f.write(f'generate for(i = 0; i < {nx*ny*nft}; i = i + 1) begin : clip\n')
    f.write(f'assign po[i] =\n')
    f.write(f'(p[i] > $signed({cbitwidth}\'d{(b-1) << (cshamt - a)}))? {bitwidth}\'d{b-1}:\n')
    f.write(f'(p[i] < -$signed({cbitwidth}\'d{b << (cshamt - a)}))? -{bitwidth}\'d{b}:\n')
    f.write(f'p[i] >> {cshamt - a};\n')
    f.write('end endgenerate\n')
    f.write('endmodule\n')
    return ((nx, ny, nft), a, bitwidth)

'''
assume valid padding, as much stride
'''
def genverilogmaxp(image, pool_size, name):
    cbitwidth = image[2]
    cshamt = image[1]
    nx = image[0][0]
    ny = image[0][1]
    nft = image[0][2]
    nkx = pool_size[0]
    nky = pool_size[1]
    nox = (nx - nkx) // nkx + 1
    noy = (ny - nky) // nky + 1
    ninputs = nx * ny * nft
    noutputs = nox * noy * nft
    bitwidth = cbitwidth
    f = open(f'{name}.v', mode='w')
    f.write(f'module {name} (\n')
    f.write(f'input [{ninputs*cbitwidth}-1:0] in,\n')
    f.write(f'output [{noutputs*bitwidth}-1:0] out);\n')
    f.write(f'wire signed [{cbitwidth}-1:0] p [0:{ninputs}-1];\n')
    f.write('genvar i;\n')
    f.write(f'generate for(i = 0; i < {ninputs}; i = i + 1) begin : parse\n')
    f.write(f'assign p[i] = {{1\'b0, in[{cbitwidth}*(i+1)-1:{cbitwidth}*i]}};\n')
    f.write('end endgenerate\n')
    f.write(f'wire signed [{bitwidth}-1:0] po [0:{noutputs}-1];\n')
    f.write(f'generate for(i = 0; i < {noutputs}; i = i + 1) begin : parseout\n')
    f.write(f'assign out[{bitwidth}*(i+1)-1:{bitwidth}*i] = po[i];\n')
    f.write('end endgenerate\n')
    f.write('genvar j, k;\n')
    f.write(f'generate for(i = 0; i < {nx - nkx + 1}; i = i + {nkx}) begin : poolx\n')
    f.write(f'for(j = 0; j < {ny - nky + 1}; j = j + {nky}) begin : pooly\n')
    f.write(f'for(k = 0; k < {nft}; k = k + 1) begin : poolz\n')
    f.write(f'{name}_max max ( po[k+j/{nky}*{nft}+i/{nkx}*{nft}*{noy}] ')
    for dx in range(nkx):
        for dy in range(nky):
            f.write(f', p[k+(j+{dy})*{nft}+(i+{dx})*{nft}*{ny}] ')
    f.write(');\n')
    f.write('end end end endgenerate\n')
    f.write('endmodule\n')
    f.write(f'module {name}_max (\n')
    f.write(f'output reg [{bitwidth}-1:0] out\n')
    for i in range(nky * nkx):
        f.write(f', input [{bitwidth}-1:0] in{i}\n')
    f.write(');\n')
    f.write('always @* begin\n')
    f.write('out = in0;\n')
    for i in range(1, nky * nkx):
        f.write(f'if(out < in{i}) begin\n')
        f.write(f'out = in{i};\n')
        f.write('end\n')
    f.write('end\n')
    f.write('endmodule\n')
    return ((nox, noy, nft), cshamt, bitwidth)

'''
assume valid padding, as much stride
'''
def genverilogavep(image, pool_size, name):
    cbitwidth = image[2]
    cshamt = image[1]
    nx = image[0][0]
    ny = image[0][1]
    nft = image[0][2]
    nkx = pool_size[0]
    nky = pool_size[1]
    shamt = int(round(math.log2(nkx * nky)))
    nox = (nx - nkx) // nkx + 1
    noy = (ny - nky) // nky + 1
    ninputs = nx * ny * nft
    noutputs = nox * noy * nft
    bitwidth = cbitwidth + shamt
    f = open(f'{name}.v', mode='w')
    f.write(f'module {name} (\n')
    f.write(f'input [{ninputs*cbitwidth}-1:0] in,\n')
    f.write(f'output [{noutputs*bitwidth}-1:0] out);\n')
    f.write(f'wire signed [{cbitwidth}-1:0] p [0:{ninputs}-1];\n')
    f.write('genvar i;\n')
    f.write(f'generate for(i = 0; i < {ninputs}; i = i + 1) begin : parse\n')
    f.write(f'assign p[i] = {{1\'b0, in[{cbitwidth}*(i+1)-1:{cbitwidth}*i]}};\n')
    f.write('end endgenerate\n')
    f.write(f'wire signed [{bitwidth}-1:0] po [0:{noutputs}-1];\n')
    f.write(f'generate for(i = 0; i < {noutputs}; i = i + 1) begin : parseout\n')
    f.write(f'assign out[{bitwidth}*(i+1)-1:{bitwidth}*i] = po[i];\n')
    f.write('end endgenerate\n')
    f.write('genvar j, k;\n')
    f.write(f'generate for(i = 0; i < {nx - nkx + 1}; i = i + {nkx}) begin : poolx\n')
    f.write(f'for(j = 0; j < {ny - nky + 1}; j = j + {nky}) begin : pooly\n')
    f.write(f'for(k = 0; k < {nft}; k = k + 1) begin : poolz\n')
    f.write(f'assign po[k+j/{nky}*{nft}+i/{nkx}*{nft}*{noy}] =')
    for dx in range(nkx):
        for dy in range(nky):
            f.write(f' + p[k+(j+{dy})*{nft}+(i+{dx})*{nft}*{ny}]')
    f.write(';\n')
    f.write('end end end endgenerate\n')
    f.write('endmodule\n')
    return ((nox, noy, nft), cshamt + shamt, bitwidth)

'''
assume 1d input image
ignore activatoin function
'''
def genverilogdense(image, weights, name):
    cbitwidth = image[2]
    cshamt = image[1]
    n = image[0][0]
    w0 = np.where(weights[0] != 0, np.log2(abs(weights[0])), 1000).astype('int64')
    shamt = -np.min(w0)
    w0sign = weights[0] > 0
    w0prune = weights[0] == 0
    w1 = weights[1] * (1 << (shamt + cshamt))
    w1 = w1.astype('int64')
    m = np.shape(w0)[1]
    ninputs = n
    noutputs = m
    bitwidth = cbitwidth + (n+1-1).bit_length() + shamt
    f = open(f'{name}.v', mode='w')
    f.write(f'module {name} (\n')
    f.write(f'input [{ninputs*cbitwidth}-1:0] in,\n')
    f.write(f'output [{noutputs*bitwidth}-1:0] out);\n')
    f.write(f'wire signed [{cbitwidth}-1:0] p [0:{ninputs}-1];\n')
    f.write('genvar i;\n')
    f.write(f'generate for(i = 0; i < {ninputs}; i = i + 1) begin : parse\n')
    f.write(f'assign p[i] = {{1\'b0, in[{cbitwidth}*(i+1)-1:{cbitwidth}*i]}};\n')
    f.write('end endgenerate\n')
    f.write(f'wire signed [{bitwidth}-1:0] po [0:{noutputs}-1];\n')
    f.write(f'generate for(i = 0; i < {noutputs}; i = i + 1) begin : parseout\n')
    f.write(f'assign out[{bitwidth}*(i+1)-1:{bitwidth}*i] = po[i];\n')
    f.write('end endgenerate\n')
    for j in range(m):
        f.write(f'assign po[{j}] =');
        for i in range(n):
            if w0prune[i][j]:
                continue
            d = shamt + w0[i][j]
            sval = f'p[{i}]'
            if d != 0:
                sval = f'{{{sval}, {d}\'d0}}'
            if w0sign[i][j]:
                f.write(f' + {sval}')
            else:
                f.write(f' - {sval}')
        sval = f'{bitwidth}\'d{abs(w1[j])}'
        if w1[j] > 0:
            f.write(f' + {sval};\n')
        elif w1[j] < 0:
            f.write(f' - {sval};\n')
        else:
            f.write(';\n')
    f.write('endmodule\n')
    return ((m, 1, 1), cshamt + shamt, bitwidth)

def genverilog(layers, image, cliprange, name):
    ninputs = image[0][0] * image[0][1] * image[0][2] * image[2]
    submodules = []
    images = []
    for i, layer in enumerate(layers):
        layername = layer.__class__.__name__
        modulename = f'{name}_{layername}_{i}'
        print(layername)
        if layername == 'Conv2D':
            image = genverilogconv(image, layer.get_weights(), layer.strides, modulename)
            submodules.append((i, modulename))
        elif layername == 'BatchNormalization':
            image = genverilognorm(image, layer.get_weights(), modulename)
            submodules.append((i, modulename))
        elif layername == 'Lambda':
            image = genverilogclip(image, cliprange, modulename)
            submodules.append((i, modulename))
        elif layername == 'MaxPooling2D':
            image = genverilogmaxp(image, layer.pool_size, modulename)
            submodules.append((i, modulename))
        elif layername == 'AveragePooling2D':
            image = genverilogavep(image, layer.pool_size, modulename)
            submodules.append((i, modulename))
        elif layername == 'Flatten':
            image = ((image[0][0] * image[0][1] * image[0][2], 1, 1), image[1], image[2])
        elif layername == 'Dense':
            image = genverilogdense(image, layer.get_weights(), modulename)
            submodules.append((i, modulename))
        images.append(image)
    noutputs = image[0][0] * image[0][1] * image[0][2] * image[2]
    f = open(f'{name}.v', mode='w')
    f.write(f'module {name} (\n')
    f.write(f'input [{ninputs}-1:0] in,\n')
    f.write(f'output [{noutputs}-1:0] out);\n')
    inname = 'in'
    for i, modulename in submodules:
        image = images[i]
        noutputs = image[0][0] * image[0][1] * image[0][2] * image[2]
        outname = f'p{i}'
        f.write(f'wire [{noutputs}-1:0] {outname};\n')
        f.write(f'{modulename} m{i} ({inname}, {outname});\n')
        inname = outname
    f.write(f'assign out = {inname};\n')
    f.write('endmodule')
    return images
