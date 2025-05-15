import sys
from PIL import Image
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

path_figure = root_path + "/figure/png/"
path_figure_paper = root_path + "/figure/jpeg/"
pngs = [x for x in os.listdir(path_figure) if x.endswith(".png")]
for figname in pngs:
    fig_file = path_figure + figname
    im = Image.open(fig_file)
    ## convert to RGB
    rgb_im = im.convert('RGB')
    ## save jpeg files
    name, extension = os.path.splitext(figname)
    jpg_extension = '.jpg'
    jpeg_fig_file = path_figure_paper+name+jpg_extension
    rgb_im.save(jpeg_fig_file, quality=95)
