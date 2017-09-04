import pyglet
from pygame import mixer
import pyglet

mixer.init()
mixer.music.load('bullet.wav')
# music = pyglet.resource.media('Katy Perry - Chained to the Rhythm.mp3')
mixer.music.play()

if __name__ == '__main__':
    pyglet.app.run()
