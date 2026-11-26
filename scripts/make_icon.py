import sys
sys.path.insert(0,r'VoxTrace')
from PySide6.QtWidgets import QApplication
from voxtrace.ui.icons import app_icon
from PIL import Image
app=QApplication([])
icon=app_icon()
pix=icon.pixmap(256,256)
pix.save(r'VoxTrace\docs\app_icon.png')
img=Image.open(r'VoxTrace\docs\app_icon.png').convert('RGBA')
img.save(r'VoxTrace\app.ico', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
print('icon saved')
