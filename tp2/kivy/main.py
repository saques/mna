#qpy:kivy

import kivy
kivy.require('1.10.0')

import numpy as np
import cv2
import time


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
		text: 'Rate: '
		height: '48dp'
	Button:
		text: 'Capture'
		size_hint_y: None
		height: '48dp'
		on_press: root.capture()
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

def compute_hr(n, fps, r, g, b):
	f = np.linspace(-n/2,n/2-1,n)*fps/n

	r = r[0,0:n]-np.mean(r[0,0:n])
	g = g[0,0:n]-np.mean(g[0,0:n])
	b = b[0,0:n]-np.mean(b[0,0:n])

	R = np.abs(np.fft.fftshift(fft_ct(r)))**2
	G = np.abs(np.fft.fftshift(fft_ct(g)))**2
	B = np.abs(np.fft.fftshift(fft_ct(b)))**2

	print(abs(f[np.argmax(G)])*60)

	return abs(f[np.argmax(G)])*60


class CameraBundle:

	def __init__(self):
		self.cam = CoreCamera(index=0,resolution=(640,480))
		self.cam.start()
		self.caminstance = self.cam._android_camera

	def fps(self):
		return self.cam.fps

	def capture(self, n):

		r = np.zeros((1,n))
		g = np.zeros((1,n))
		b = np.zeros((1,n))
		
		self._flash_on()		

		for i in range(0,n):
			print(i)
			frame = self.cam.read_frame()
			r[0,i] = np.mean(frame[210:240,290:320,0])
			g[0,i] = np.mean(frame[210:240,290:320,1])
			b[0,i] = np.mean(frame[210:240,290:320,2])
			
			print("%f %f %f" % (r[0,i], g[0,i], b[0,i]))
			
		self._flash_off()

		return r, g, b

	def _flash_on(self):
		p = self.caminstance.getParameters()
		p.setFlashMode(Parameters.FLASH_MODE_TORCH)
		self.caminstance.setParameters(p)

	def _flash_off(self):
		p = self.caminstance.getParameters()
		p.setFlashMode(Parameters.FLASH_MODE_OFF)
		self.caminstance.setParameters(p)
		


class CameraClick(BoxLayout):

	def __init__(self):
		super(CameraClick, self).__init__()
		self.cb = CameraBundle()

	def capture(self):
		n = 1024
		fps = self.cb.fps()
		r, g, b = self.cb.capture(n)
		self.ids.rate.text = "Rate: %f ppm." % (compute_hr(n, fps, r, g, b))



class TestCamera(App):

	def build(self):
        	return CameraClick()

TestCamera().run()
