import pyglet

window = pyglet.window.Window()

label = pyglet.text.Label('Hello,pyglet',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2,y=window.height//2,
                          anchor_x='center',anchor_y='center')
image = pyglet.resource.image('player.png')

@window.event
def on_draw():
    window.clear()
    image.blit(0,0)
    # label.draw()

if __name__ == '__main__':
    pyglet.app.run()