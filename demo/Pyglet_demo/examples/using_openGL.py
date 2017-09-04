from pyglet.gl import *
import pyglet

# Direct OpenGL commands to this window
window = pyglet.window.Window()

vertices = [0,0,window.width,0,window.width,window.height]
vertices_gl = (GLfloat*len(vertices))(*vertices)
print(vertices)
glEnableClientState(GL_VERTEX_ARRAY)
glVertexPointer(2,GL_FLOAT,0,vertices_gl)


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    # glBegin(GL_TRIANGLES)
    # glVertex2f(0,0)
    # glVertex2f(window.width,0)
    # glVertex2f(window.width, window.height)
    # glEnd()
    glDrawArrays(GL_TRIANGLES,0,len(vertices)//2)

if __name__ == '__main__':
    pyglet.app.run()