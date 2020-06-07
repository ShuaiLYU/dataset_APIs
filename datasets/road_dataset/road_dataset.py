import os
from random import  shuffle
import  pandas as pd
import  numpy as np
import  cv2
def list_folder(root, use_absPath=True, func=None):
	"""
	:param root:  文件夹根目录
	:param func:  定义一个函数，过滤文件
	:param use_absPath:  是否返回绝对路径， false ：返回相对于root的路径
	:return:
	"""
	root = os.path.abspath(root)
	if os.path.exists(root):
		print("遍历文件夹【{}】......".format(root))
	else:
		raise Exception("{} is not existing!".format(root))
	files = []
	# 遍历根目录,
	for cul_dir, _, fnames in sorted(os.walk(root)):
		for fname in sorted(fnames):
			path = os.path.join(cul_dir, fname)#.replace('\\', '/')
			if  func is not None and not func(path):
				continue
			if use_absPath:
				files.append(path)
			else:
				files.append(os.path.relpath(path,root))
	return files



class RoadDataset(object):
    def __init__(self,root,transforms=None):
        super().__init__()
        self.root=root
        self.transforms=transforms
        self.samples=self.list_dirs()
        print("	gen {} samples".format(len( self.samples), ))
        # for sample in self.samples:
        #     print(sample)
    def list_dirs(self):
        def is_img(img_path):
            return img_path.endswith(".jpg")
        def is_label(img_path):
            return img_path.endswith("_bin.png")
        # fnames=list_folder(self.root,use_absPath=False,func=None)
        # print(len(fnames))
        imgs=list_folder(self.root,use_absPath=False,func=is_img)
        print("	find {} images".format(len(imgs), ))
        labels=list_folder(self.root,use_absPath=False,func=is_label)
        print("	find {} labels".format(len(labels), ))
        labels_dict={ os.path.basename(path):path for path in labels}
        samples=[]
        for img  in  imgs:
            label_name = os.path.basename(img).replace(".jpg", "_bin.png")
            if label_name  not in  labels_dict.keys():
                print(label_name)
                continue
            samples.append((img,labels_dict[label_name]))
        return samples

    def save_to_csv(self):
        config_path=os.path.join(root,"config")
        if not os.path.exists(config_path):
            print("make dir: {}".format(config_path))
            os.makedirs(config_path)
        images=[ sample[0] for sample in self.samples]
        labels=[ sample[1] for sample in self.samples]
        data=pd.DataFrame({"image":images,"label":labels})
        # print(len(data))
        # shuffle(data)
        data_sz=len(data)
        train_offset=int(data_sz*0.6)
        valid_offset = int(data_sz * 0.8)
        train_data=data[:train_offset]
        valid_data =data[train_offset:valid_offset]
        test_data=data[valid_offset:]
        train_data.to_csv(os.path.join(config_path,"train.csv",),index=False)
        valid_data.to_csv(os.path.join(config_path, "valid.csv"),index=False)
        test_data.to_csv(os.path.join(config_path, "test.csv"),index=False)

    def make_dataset(self):
        pass

    def gen_a_sample(self,sample):
        file_basename_image, file_basename_label =sample

        image_path = os.path.join(self.root, file_basename_image)
        image=cv_imread(image_path,-1)
        #print(image.shape)
        image = np.array(image).astype(np.uint8)
        if file_basename_label is not None:
            label_path = os.path.join(self.root, file_basename_label)
            pixel_label = cv_imread(label_path, -1)
            label_pixel = np.array(pixel_label).astype(np.uint8)
        else:
            label_pixel=np.zeros_like(image).astype(np.uint8)
        if self.transform_PIL is not None:
             image,label_pixel=self.transform_PIL([image,label_pixel])
       # utils.plt_utils.plt_show_imgs([image, label_pixel])
        image, label_pixel = self.transform_array([image, label_pixel])
       # utils.plt_utils.plt_show_imgs([image.squeeze(), label_pixel.squeeze()])
        return image, label_pixel, int(label), file_basename_image
if  __name__=="__main__":
    root="F:\学习\百度课程\dataset\无人车车道线检测训练集-初赛"
    data=RoadDataset(root)
    data.save_to_csv()
