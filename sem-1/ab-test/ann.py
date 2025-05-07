import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *


def init_3d():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, 800 / 600, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, 0.0, -5)


def create_text_texture(text, font_size, color):
    pygame.font.init()
    font = pygame.font.SysFont("Arial", font_size, True)
    text_surface = font.render(text, True, color, (0, 0, 0, 0))  # Прозрачный фон

    # Поворот текстуры по оси Y
    flipped_surface = pygame.transform.flip(text_surface, False, True)

    text_data = pygame.image.tostring(flipped_surface, "RGBA", True)
    width, height = flipped_surface.get_size()

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return texture_id, width, height


def draw_textured_quad(texture_id, width, height, position):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex3f(-width / 200.0, -height / 200.0, 0)
    glTexCoord2f(1, 1)
    glVertex3f(width / 200.0, -height / 200.0, 0)
    glTexCoord2f(1, 0)
    glVertex3f(width / 200.0, height / 200.0, 0)
    glTexCoord2f(0, 0)
    glVertex3f(-width / 200.0, height / 200.0, 0)
    glEnd()

    glPopMatrix()
    glDisable(GL_TEXTURE_2D)


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Please forgive me darling")
    clock = pygame.time.Clock()

    init_3d()

    texture_id, width, height = create_text_texture("Анюта золотце, а не булль <3", 64, (255, 182, 193))

    angle = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()

        # Rotate the quad
        glRotatef(angle, 0, 1, 0)
        draw_textured_quad(texture_id, width, height, (0.0, 0.0, 0.0))

        glPopMatrix()

        angle += 1  # Increment rotation angle

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
