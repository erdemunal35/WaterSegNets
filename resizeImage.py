import os, sys
from PIL import Image

size = 256, 256

one_image = "train28.png"
outfile = "train28-45.png"

try:
	im = Image.open(one_image)
	im = im.resize(size, Image.ANTIALIAS)
	im.save(outfile, "PNG")
	
except IOError:
	print("Error occured")