import cv2, time
from datetime  import datetime
import argparse 
import os

face_casacde=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


video = cv2.VideoCapture(0)

while True:
	check,frame=video.read()
	if frame is not None:
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		face = face_casacde.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)
		for x,y,w,h in faces:
			img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
			exact_time=datetime.now().strftime('%Y-%b-%d-%H-%S-%f')
			cv2.imwrite("face detected"+str(exact_time)+".jpg",img)


		cv2.imshow("home surv",frame)
		key=cv2.waitKey(1)

		if Key==ord('q'):
			ap=argparse.ArgumentParser()
			ap.add_argument("-ext","--extension",required=False,default='jpg')
			ap.add_argument("-o","--output",required=False,defalut='output.mp4')
			args=vars(ap.parse_args())


			dir_path='.'
			ext=args['extension']
			output=args['output']


			image=[]

			for f in os.listdir(dir_path):
				if f.endswith(ext):
					image.append(f)



			image_path=os.path.join(dir_path,image[0])
			frame=cv2.imread(image_path)
			height,width,channels=frame.shape


			forcc=cv2.VideoWriter_forcc(*'mp4v')
			out=cv2.VideoWriter_forcc(output,forcc,5.0,(width,height))


			for image in images:
				image_path=os.path.join(dir_path,image)
				frame=cv2.imread(image_path)
				out.write(frame)

			break


video.release()
cv2.destroyAllWindows