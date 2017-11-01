#qpy:kivy

import kivy
kivy.require('1.10.0')

import numpy as np
import cv2
import time
from time import sleep


from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.core.camera import Camera as CoreCamera
from jnius import autoclass

Parameters = autoclass('android.hardware.Camera$Parameters')

Builder.load_string('''
<CameraClick>:
	orientation: 'vertical'
	Label:
		id: rate
		text: 'Press Capture to start'
		height: '48dp'
	Button:
		text: 'Capture'
		size_hint_y: None
		height: '48dp'
		on_press: app.capture()
''')

#Cooley-Tukey implementation (only N = 2^j for some j)
def fft_ct_1(x, N):
	ans = None
	if N == 1:
		ans = np.empty((1,)).astype(complex)
		ans[0] = x[0]
	else:
		t1 = fft_ct_1(x[::2], N/2)
		t2 = fft_ct_1(x[1::2], N/2)
		ans = np.empty((N,)).astype(complex)
		for k in xrange(0, N/2):
			t = t1[k]
			ans[k] = t + np.exp((-2*np.pi*1j*k)/N)*t2[k]
			ans[k+N/2] = t - np.exp((-2*np.pi*1j*k)/N)*t2[k]
	return ans

def fft_ct(x):
	N = len(x)
	if not (N & (N-1)) == 0:
		raise ValueError("Length must be a power of 2")
	return fft_ct_1(x, N)

def compute_hr(n, fps, r):

	f = (np.linspace(-n/2,n/2-1,n)*fps/n)*60

	r = r[0,0:n]-np.mean(r[0,0:n])

	R = np.abs(np.fft.fftshift(fft_ct(r)))**2

	filter = np.zeros(n)
	inc = (fps/n)*60
	filter[int(n/2+60/inc):int(n/2+110/inc)] = 1
	R *= filter

	for k in range(0, n):
		print "%f " % R[k]

	return abs(f[np.argmax(R)])


class CameraBundle:

	def __init__(self):
		self.cam = CoreCamera(index=0,resolution=(640,480))
		self.caminstance = self.cam._android_camera

	def start(self):
		self.cam.start()

	def stop(self):
		self.cam.stop()

	def fps(self):
		return self.cam.fps

	def capture(self, n):
		frames = []
		st = 1.0/29.97

		self._flash_on()		
		
		for i in range(0,n):
			frames.append(self.cam.grab_frame())
			sleep(st)
		
		self._flash_off()

		return frames

	def decode(self,frames):
		r = np.zeros((1,len(frames)))
		for i in range(0,len(frames)):
			frame = self.cam.decode_frame(frames[i])
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			r[0,i] = np.mean(frame[180:260:5,270:340:5])
			print r[0,i]
		return r

	def _flash_on(self):
		p = self.caminstance.getParameters()
		p.setFlashMode(Parameters.FLASH_MODE_TORCH)
		p.setExposureCompensation(-2)
		self.caminstance.setParameters(p)

	def _flash_off(self):
		p = self.caminstance.getParameters()
		p.setFlashMode(Parameters.FLASH_MODE_OFF)
		p.setExposureCompensation(0)
		self.caminstance.setParameters(p)
		


class CameraClick(BoxLayout):

	def __init__(self):
		super(CameraClick, self).__init__()

	def change_text(self,text):
		self.ids.rate.text = text


class TestCamera(App):

	def build(self):
		self.box = CameraClick()
		self.cb = CameraBundle()
		self.cb.start()        	
		return self.box

	def capture(self):
		n = 1024
		
		fps = 29.97		
		frames = self.cb.capture(n)
		
		r = self.cb.decode(frames)

		r = r[:,100:100+n/2]

		bpm = compute_hr(n/2, fps, r)

		self.box.change_text("Rate: %f bpm." % (bpm))


TestCamera().run()
