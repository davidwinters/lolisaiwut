__author__ = 'Isai Olvera'
 
import wx
from PIL import Image
import wx.lib.inspection
import matplotlib.pyplot as plt
import matplotlib
import pylab
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
        FigureCanvasWxAgg as FigCanvas, \
        NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import data
from skimage.filter.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage import color
import skimage
from skimage import measure
 
 
# set global variable here
THRESH = 140
 
class Panel(wx.Panel):
 
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
 
        self.quote1 = wx.StaticText(self, label="EL Image and Grayscale Analyzer", pos=(20, 30))
        self.quote2 = wx.StaticText(self, label="Cropped EL Raw Image", pos=(300, 30))
        self.quote3 = wx.StaticText(self, label="Binarized Image", pos=(575, 30))
        self.quote3 = wx.StaticText(self, label="Colorized Image", pos=(840, 30))
        self.quote4 = wx.StaticText(self, label="Black Pixel Count: ", pos = (20, 300)).SetBackgroundColour('Yellow')
        self.quote5 = wx.StaticText(self, label="Grayscale Area: ", pos = (20, 340)).SetBackgroundColour('Yellow')
        self.quote6 = wx.StaticText(self, label="Binarize Threshold Adjustment", pos=(80, 480))
 
        self.selected_image = wx.GenericDirCtrl(self, -1, size=(250, 225), pos=(20, 50), style=wx.DIRCTRL_SHOW_FILTERS, filter="JPG files (*.jpg)|*.jpg")   # Directory Tree that filters for JPG files
        selected_image = self.selected_image.GetTreeCtrl()
        self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.ondoubleclick, selected_image)
 
        sliderpanel = wx.Panel(self, -1, size=(300, 110), pos = (0, 360))
        slider = wx.Slider(sliderpanel, 100, 140, 50, 200, (30, 60), (250, -1), wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS)
        slider.SetTickFreq(5,1)
        slider.Bind(wx.EVT_SCROLL_CHANGED, self.sliderchanged)
 
        self.MaxImageSize_x = 235
        self.MaxImageSize_y = 500
 
        self.Image = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(self.MaxImageSize_x, self.MaxImageSize_y),
                                     pos=(300, 50))
        self.binImage = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(self.MaxImageSize_x, self.MaxImageSize_y),
                                        pos=(575, 50))
        self.colorizedImage = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(self.MaxImageSize_x, self.MaxImageSize_y),
                                        pos=(850, 50))
 
 
    def ondoubleclick(self, event):
        global THRESH
 
 
        self.jpg_path = self.selected_image.GetFilePath()
 
        if (self.jpg_path):
            Img = wx.Image(self.jpg_path, wx.BITMAP_TYPE_JPEG)
 
            resized_image = resize(Img, self.MaxImageSize_x, self.MaxImageSize_y)   # function that re-sizes an image based on def resize()
            self.Image.SetBitmap(wx.BitmapFromImage(resized_image))     # displays image in wx.Python GUI/frame
 
            self.image = Image.open(self.jpg_path)                      # Opens the file path and saves it into self.im
 
            if self.image.mode != "RGB":
                dlg = wx.MessageDialog(self, "The image you selected is not a RGB image", style=wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
 
 
            self.graylevel_image = rgb_to_gray_level(self.image)
            self.vert_array = vertical_grayscale(self.graylevel_image)
            self.interpolation_result = sum(self.vert_array)
 
            self.plotpanel = wx.Panel(self, -1, size=(800, 200), pos = (10, 570))
            #self.plotpanel = wx.BoxSizer(wx.HORIZONTAL)
 
            self.figure = matplotlib.figure.Figure(dpi=100, figsize=(8.5,2))
            self.axes = self.figure.add_subplot(111)
            self.axes.plot(self.vert_array)
            self.canvas = FigCanvas(self.plotpanel, -1, self.figure)
 
            self.binarized_image = rgb_to_gray_level(self.image)        # Converts the image that was just imported to grayscale
            binarize(self.binarized_image, THRESH)                              # Binarizes the now grayscale image
            #print THRESH
            self.defect_count = pixel_counter(self.binarized_image)     # Counts the black pixels in the image and returns them to defect_count
 
            self.quote7 = wx.StaticText(self, label= str(self.defect_count), pos = (200, 300))  # Displays the defect count number in the GUI
            self.quote8 = wx.StaticText(self, label= str(self.interpolation_result), pos = (200, 340))   # Displays the grayscale summation in the GUI
 
            wx_image = piltoimage(self.binarized_image)
            resized_binarized = resize(wx_image, self.MaxImageSize_x, self.MaxImageSize_y)
 
            self.binImage.SetBitmap(wx.BitmapFromImage(resized_binarized))     # displays image in wx.Python GUI/frame
 
 
            # This section is to import the image from the selected path, then convert it to entropy#
            # Once the entropy ndarray is generated, we have to convert it so wx_image can display it in the frame#
            entropyimage = skimage.color.rgb2gray(mpimg.imread(self.jpg_path))
            ubyte_entropyimage = img_as_ubyte(entropyimage)
 
 
 
 
        self.Refresh()
        event.Skip()
 
 
    def sliderchanged(self, event):
        global THRESH
        self.jpg_path1 = self.selected_image.GetFilePath()
        self.new_thresh = event.EventObject.GetValue()
        if (self.jpg_path1):
            #print self.new_thresh
            self.image1 = Image.open(self.jpg_path1)
 
            if self.image1.mode != "RGB":
               return
 
            else:
                self.binarized_image1 = rgb_to_gray_level(self.image1)
                binarize(self.binarized_image1, self.new_thresh)
                self.defect_count_updated = pixel_counter(self.binarized_image1)
                THRESH = self.new_thresh
 
                self.quote7.SetLabel(str(self.defect_count_updated))
                wx_image1 = piltoimage(self.binarized_image1)
                resized_binarized1 = resize(wx_image1, self.MaxImageSize_x, self.MaxImageSize_y)
                self.binImage.SetBitmap(wx.BitmapFromImage(resized_binarized1))
 
        self.Refresh()
 
def piltoimage(pil, alpha=True):
    """Convert PIL Image to wx.Image."""
    if alpha:
        image = apply(wx.EmptyImage, pil.size)
        image.SetData(pil.convert("RGB").tostring())
        image.SetAlphaData(pil.convert("RGBA").tostring()[3::4])
    else:
        image = wx.EmptyImage(pil.size[0], pil.size[1])
        new_image = pil.convert('RGB')
        data = new_image.tostring()
        image.SetData(data)
    return image
 
 
def imagetopil(image):
    """Convert wx.Image to PIL Image."""
    w, h = image.GetSize()
    data = image.GetData()
 
    redimage = Image.new("L", (w, h))
    redimage.fromstring(data[0::3])
    greenimage = Image.new("L", (w, h))
    greenimage.fromstring(data[1::3])
    blueimage = Image.new("L", (w, h))
    blueimage.fromstring(data[2::3])
 
    if image.HasAlpha():
        alphaimage = Image.new("L", (w, h))
        alphaimage.fromstring(image.GetAlphaData())
        pil = Image.merge('RGBA', (redimage, greenimage, blueimage, alphaimage))
    else:
        pil = Image.merge('RGB', (redimage, greenimage, blueimage))
    return pil
 
 
def resize(input_image, max_x=235, max_y=500):
    w = input_image.GetWidth()
    h = input_image.GetHeight()
    if w > h:
        neww = max_x
        newh = max_x * h / w
    else:
        newh = max_y
        neww = max_y * w / h
    img = input_image.Scale(neww, newh)
    return img
 
 
#Luminance conversion formula from http://en.wikipedia.org/wiki/Luminance_(relative)
def luminosity(rgb, rcoeff=0.2126, gcoeff=0.7152, bcoeff=0.0722):
    return rcoeff * rgb[0] + gcoeff * rgb[1] + bcoeff * rgb[2]
 
 
# Take a PIL rgb image and produce a factory that yields
# ((x,y), r,g,b)), where (x,y) are the coordinates
# of a pixel, (x,y), and its RGB values.
def gen_pix_factory(im):
    num_cols, num_rows = im.size
    r, c = 0, 0
    while r != num_rows:
        c = c % num_cols
        yield ((c, r), im.getpixel((c, r)))
        if c == num_cols - 1: r += 1
        c += 1
 
 
# take a PIL RGB image and a luminosity conversion formula,
# and return a new gray level PIL image in which each pixel
# is obtained by applying the luminosity formula to the
# corresponding pixel in the RGB values.
def rgb_to_gray_level(rgb_img, conversion=luminosity):
    gl_img = Image.new('L', rgb_img.size)
    gen_pix = gen_pix_factory(rgb_img)
    lum_pix = ((gp[0], conversion(gp[1])) for gp in gen_pix)
    for lp in lum_pix:
        gl_img.putpixel(lp[0], int(lp[1]))
    return gl_img
 
 
# Take a gray level image and a gray level threshold and
# replace a pixel's gray level with 0 (black) if it's gray
# level value is <= than the threshold and with
# 255 (white) if it's > than the threshold.
def binarize(gl_img, thresh):
    gen_pix = gen_pix_factory(gl_img)
    for pix in gen_pix:
        if pix[1] <= thresh:
            gl_img.putpixel(pix[0], 0)
        else:
            gl_img.putpixel(pix[0], 255)
 
 
# Take a binarized image and count every pixel that is black
def pixel_counter(binarized_image, black=0):
    gen_pix = gen_pix_factory(binarized_image)
    count = 0
    for pix in gen_pix:
        if pix[1] == black:
            count += 1
    return count
 
 
# Calculates and returns a list from the vertical grayscale (row average) values.
def vertical_grayscale(gl_img):
    gen_pix = gen_pix_factory(gl_img)
    vert_list = []
    avg = 0
    row = 0
    num_cols, num_rows = gl_img.size
    # print gl_img.size
    for pix in gen_pix:
        x, y = pix[0]
        if y == row:
            avg += pix[1]
        else:
            avg = avg / num_rows
            vert_list.append(avg)
            avg = 0
            row += 1
    return vert_list
 
 
app = wx.App(False)
frame = wx.Frame(None, size=(1400, 820))  # x, y for window
#wx.lib.inspection.InspectionTool().Show()
panel = Panel(frame)
frame.Show()
app.MainLoop()
