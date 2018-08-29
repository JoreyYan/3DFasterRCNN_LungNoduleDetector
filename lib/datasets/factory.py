# --------------------------------------------------------
# Written by HusonChen
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
#新建一个私有字典
__sets = {}
#tianchi是作者编辑的一个类
from datasets.tianchi import tianchi
import numpy as np
#这个字典里有 train val test 20xx等各种key
for split in ['train', 'val', 'test']:
    #给字典设置参数
    #分别向'tianchi_{}'.format这里输入 ['train', 'val', 'test'] 并且将结果返回给name 这个format并不是tianchi类的方法 并没有定义
    #字典的前三个name是 tianchi_train tianchi_val tianchi_test 这样在输入参数时--imdb_train tianchi_train 实际上返回的 
    name = 'tianchi_{}'.format( split)
    
    #字典里tianchi_train的值就是tianchi（split）的输出
    __sets[name] = (lambda split=split : tianchi(split))

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))
#按理说应该是返回的数据位置
def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
