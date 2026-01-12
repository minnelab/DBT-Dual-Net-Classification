介绍每个函数的功能

1.crop_dbt_region.py  根据csv文件的内容，将dbt数据的存储地址、对应的病况分类、肿瘤具体区域（x,y,z坐标）提取出来，并把slice部分转为nii.gz文件
                        可以用作后续的分类，png文件作为2D slice的预览，用box 矩形框标出了 肿瘤的2D 区域。代码中depth和surround_dis分别为5和10，depth表示提取中心slice前后各五张，surround_dis表示在官方提供的box矩形外多截取周围10个height/width单位。



2.crop_dbt_background.py 根据csv文件的内容，提取出dbt文件，并根据dbt文件中的 分类 cancer,benign,normal进行分块处理,
            对每张slices进行区域切割，将背景部分较少的区域保存下来，方便整合成patch



3.convert_alldicomimage_rgb.py
                     对目标区域的dbt文件进行裁剪，并且每3张合成一个rgb 三通道图像；代码中offset = 3，代表着这三通道上的slice，每个之间的间距为 3，而不是紧连着的3张slice。


