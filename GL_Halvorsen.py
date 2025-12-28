# Halvorsen chaotic attractor

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

WIDTH = 960
HEIGHT = 540
POINTS_ITER = 5000

OFFSET_X = 0.25
OFFSET_Y = 0.2

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in float id;
uniform float time_offset;
uniform vec2 offset;
out float v_color;

vec3 halvorsen(vec3 p) {
    float a = 1.4;

    return vec3(
        -a * p.x - 4.0 * p.y - 4.0 * p.z - p.y * p.y,
        -a * p.y - 4.0 * p.z - 4.0 * p.x - p.z * p.z,
        -a * p.z - 4.0 * p.x - 4.0 * p.y - p.x * p.x
    );
}

void main() {
    vec3 p = vec3(0.1, 0.0, 0.0);
    float dt = 0.003;
    int steps = int(id + time_offset);

    for (int i = 0; i < steps; i++)
        p += halvorsen(p) * dt;

    float scale = 0.06;
    gl_Position = vec4(
        p.x * scale + offset.x,
        p.y * scale + offset.y,
        0.0,
        1.0
    );

    gl_PointSize = 2.0;
    v_color = length(p);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in float v_color;
out vec4 fragColor;

void main() {
    float brightness = clamp(v_color * 0.05, 0.4, 1.0);
    fragColor = vec4(brightness, brightness, brightness, 1.0);
}
"""

def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Halvorsen Attractor")

    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    ids = np.arange(POINTS_ITER, dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, ids.nbytes, ids, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 1, GL_FLOAT, False, 0, None)

    glUseProgram(shader)
    glClearColor(0, 0, 0, 1)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_PROGRAM_POINT_SIZE)

    time_offset_loc = glGetUniformLocation(shader, "time_offset")
    offset_loc = glGetUniformLocation(shader, "offset")

    clock = pygame.time.Clock()
    running = True
    batch = 0

    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False

        glClear(GL_COLOR_BUFFER_BIT)

        glUniform1f(time_offset_loc, float(batch * POINTS_ITER))
        glUniform2f(offset_loc, OFFSET_X, OFFSET_Y)

        glDrawArrays(GL_POINTS, 0, POINTS_ITER)

        batch += 1

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()