import cv2
import numpy as np

def Impresion(namme,imagen,x,y):
    cv2.namedWindow(namme)
    cv2.moveWindow(namme, x,y)
    cv2.imshow(namme, imagen)

img = cv2.imread('Figuras.PNG')
tem = cv2.imread('Recortado.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tem_gray = cv2.cvtColor(tem, cv2.COLOR_BGR2GRAY)

res = cv2.matchTemplate(img_gray, tem_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.85
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + tem.shape[1], pt[1] + tem.shape[0]), (0,255,255), 1)

d = np.shape(img)
g= np.shape(tem)

ptf = [pt[0] + tem.shape[1], pt[1] + tem.shape[0]]
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (pt[0],pt[1],ptf[0],ptf[1])

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img = img*mask2[:,:,np.newaxis]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)

Impresion('Figuras',img,50,50)

cv2.waitKey(0)
cv2.destroyAllWindows()





