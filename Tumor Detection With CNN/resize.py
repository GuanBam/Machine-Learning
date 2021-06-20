import glob
import os
from PIL import Image

# path for original data folder
# path for resized data folder
# modify the "yes" to "no" to resize the figure for data in "no" folder
path = r'data\original\yes'
path1=r'data\resize\yes'
images = glob.glob(path+r"\*")
i=0

for img in images:
    try:
        im = Image.open(img)
        im.convert("LA")
        size= 100,100
        name = os.path.join(path1,str(i)+".JPEG")
        im.thumbnail(size)
        im.save(name,"JPEG")
        i+=1
    except OSError:
        pass

