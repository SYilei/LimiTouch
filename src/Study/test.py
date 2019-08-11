from PIL import Image 
import os
import time
import subprocess


imgs = os.listdir('../Study/study2_figure/')
for file in imgs:
	p = subprocess.Popen(["display", '../Study/study2_figure/' + file])
	# img = Image.open('../Study/study2_figure/' + file)
	# img.show()
	time.sleep(1)
	p.kill()
	# img.close()