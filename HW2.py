from re import T
import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets
from scipy import signal
from UI import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__() # in python3, super(Class, self).xxx = super().xxx
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.setup_control()
		self.cnt=0                              	# cannot draw
		# points
		self.img1Start=np.empty((0,2),int); self.img2End=np.empty((0,2),int)	# P
		self.img1End=np.empty((0,2),int); self.img2Start=np.empty((0,2),int)	# Q
		self.mapX=np.zeros((self.img1.shape[0],self.img1.shape[1]),dtype=np.float32)
		self.mapY=np.zeros((self.img2.shape[0],self.img2.shape[1]),dtype=np.float32)

	def setup_control(self):
		# TODO
		self.ui.btnMorph.clicked.connect(self.btnMorphClicked)
		self.ui.btnDraw.clicked.connect(self.btnDrawClicked)
		self.img1=cv2.imread('.\images\cheetah.jpg')
		self.img1Temp=self.img1.copy(); self.img1Ori=self.img1.copy()
		self.img2=cv2.imread('.\images\women.jpg')
		self.img2Temp=self.img2.copy(); self.img2Ori=self.img2.copy()
		cv2.imshow('cheetah', self.img1)
		cv2.imshow('women', self.img2)
		self.img12Temp=np.zeros((self.img1.shape[0],self.img1.shape[1],3),np.uint8)
		self.img21Temp=np.zeros((self.img2.shape[0],self.img2.shape[1],3),np.uint8)
		self.result=np.zeros((self.img1.shape[0],self.img1.shape[1],3),np.uint8)

	def btnDrawClicked(self):
		self.cnt=self.cnt+2
		# use two function to not detect action on the other image       
		cv2.setMouseCallback('cheetah',self.drawLineOnImg1)
		cv2.setMouseCallback('women',self.drawLineOnImg2)

	def drawLineOnImg1(self,event,x,y,flag,param):
		if self.cnt>0 and self.cnt%2==0:
			if event==cv2.EVENT_LBUTTONDOWN:        # start
				self.startx=x
				self.starty=y
				print("cheetah start(",x,",",y,")")
			if flag==cv2.EVENT_FLAG_LBUTTON:        # drag
				self.img1=self.img1Temp.copy()
				cv2.line(self.img1, (self.startx,self.starty), (x,y), (220,220,110), 2)
				cv2.imshow('cheetah', self.img1)
			if event==cv2.EVENT_LBUTTONUP:          # finish
				self.img1Start=np.append(self.img1Start,np.array([[self.startx,self.starty]]),axis=0)
				self.img1End=np.append(self.img1End,np.array([[x,y]]),axis=0)
				cv2.line(self.img1, (self.startx,self.starty), (x,y), (220,220,110), 2)
				cv2.imshow('cheetah', self.img1)
				self.img1Temp=self.img1.copy()
				print("cheetah end(",x,",",y,")","\n")
				self.cnt-=1
	   
	def drawLineOnImg2(self,event,x,y,flag,param):
		if self.cnt>0 and self.cnt%2==1:
			if event==cv2.EVENT_LBUTTONDOWN:        # start
				self.startx=x
				self.starty=y
				print("women start(",x,",",y,")")
			if flag==cv2.EVENT_FLAG_LBUTTON:        # drag
				self.img2=self.img2Temp.copy()
				cv2.line(self.img2, (self.startx,self.starty), (x,y), (220,220,110), 2)
				cv2.imshow('women', self.img2)
			if event==cv2.EVENT_LBUTTONUP:          # finish
				self.img2Start=np.append(self.img2Start,np.array([[self.startx,self.starty]]),axis=0)
				self.img2End=np.append(self.img2End,np.array([[x,y]]),axis=0)
				cv2.line(self.img2, (self.startx,self.starty), (x,y), (220,220,110), 2)
				cv2.imshow('women', self.img2)
				self.img2Temp=self.img2.copy()
				print("women end(",x,",",y,")","\n")
				self.cnt+=1

	def perpendicular(self,a) :
		b=np.empty_like(a)
		b[0]=-a[1]
		b[1]=a[0]
		return b

	def btnMorphClicked(self):
		for f in range(100):
			# for each pixel X in the destination image (img1->img2)
			for x in range (self.img2.shape[1]):
				for y in range (self.img2.shape[0]):
					X=np.array([x,y]); dsum=np.array([0,0]); wsum=0.0
					for k in range(len(self.img2Start)):	# pairs of lines
						# u=(X-P)．(Q-P)/|(Q-P)|^2
						u=np.dot(X-self.img2Start[k],self.img2End[k]-self.img2Start[k])/np.sum(np.square(self.img2End[k]-self.img2Start[k]))
						# v=(X-P)．Perpendicular(Q-P)/|(Q-P)|
						v=np.dot(X-self.img2Start[k],self.perpendicular(self.img2End[k]-self.img2Start[k]))/np.sqrt(np.sum(np.square(self.img2End[k]-self.img2Start[k])))
						# X'=P'+u．(Q'-P')+v．Perpendicular(Q'-P')/|(Q'-P')|
						newX=self.img1Start[k]+u*(self.img1End[k]-self.img1Start[k])+v*self.perpendicular(self.img1End[k]-self.img1Start[k])/np.sqrt(np.sum(np.square(self.img1End[k]-self.img1Start[k])))
						# weight=(length^p/(a+dist))^b
						a=1; b=2; p=0
						weight=math.pow((math.pow(np.sqrt(np.sum(np.square(self.img1End[k]-self.img1Start[k]))),p)/a+abs(v)),b)
						dsum=dsum+newX*weight
						wsum=wsum+weight
					self.mapX[y,x]=(dsum[0]/wsum-X[0])*f/100+X[0]
					self.mapY[y,x]=(dsum[1]/wsum-X[1])*f/100+X[1]
			self.img12Temp=cv2.remap(self.img2Ori,self.mapX,self.mapY,cv2.INTER_LINEAR)

			# for each pixel X in the destination image (img2->img1)
			for x in range (self.img1.shape[1]):
				for y in range (self.img1.shape[0]):
					X=np.array([x,y]); dsum=np.array([0,0]); wsum=0.0
					for k in range(len(self.img1Start)):		# pairs of lines
						u=np.dot(X-self.img1Start[k],self.img1End[k]-self.img1Start[k])/np.sum(np.square(self.img1End[k]-self.img1Start[k]))
						v=np.dot(X-self.img1Start[k],self.perpendicular(self.img1End[k]-self.img1Start[k]))/np.sqrt(np.sum(np.square(self.img1End[k]-self.img1Start[k])))
						newX=self.img2Start[k]+u*(self.img2End[k]-self.img2Start[k])+v*self.perpendicular(self.img2End[k]-self.img2Start[k])/np.sqrt(np.sum(np.square(self.img2End[k]-self.img2Start[k])))
						# weight=(length^p/(a+dist))^b
						a=1; b=2; p=0
						weight=math.pow((math.pow(np.sqrt(np.sum(np.square(self.img2End[k]-self.img2Start[k]))),p)/a+abs(v)),b)
						dsum=dsum+newX*weight
						wsum=wsum+weight
					self.mapX[y,x]=(dsum[0]/wsum-X[0])*(1-f/100)+X[0]
					self.mapY[y,x]=(dsum[1]/wsum-X[1])*(1-f/100)+X[1]
			self.img21Temp=cv2.remap(self.img1Ori,self.mapX,self.mapY,cv2.INTER_LINEAR)

			cv2.addWeighted(self.img12Temp,(1-f/100),self.img21Temp,f/100,0.0,self.result)
			filename=".\\result\\result"+str(1)+".jpg"
			if not os.path.exists('.\\result'):
				os.makedirs('.\\result')
			cv2.imwrite(filename, self.result)
		# pass
			

if __name__ == '__main__':
	import sys
	app = QtWidgets.QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())