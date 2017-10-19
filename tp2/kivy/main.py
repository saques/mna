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

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
        
        
        ''')


class CameraClick(BoxLayout):

	def __init__(self):
		super(CameraClick, self).__init__()
		self.cam = CoreCamera(index=0,resolution=(640,480))
		self.cam.start()

	def capture(self):
		for i in range(0,50):
			frm = self.cam.read_frame()
			print(frm.shape)
			print(frm[:][:][0])


class TestCamera(App):

	def build(self):
        	return CameraClick()

TestCamera().run()
