# Lorenz chaotic attractor

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

WIDTH = 960
HEIGHT = 540
POINTS_ITER = 5000

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in float id;
uniform float time_offset;
out float v_color;

vec3 lorenz(vec3 p) {
    float sigma = 10.0;
    float rho = 28.0;
    float beta = 8.0 / 3.0;
    return vec3(
        sigma * (p.y - p.x),
        p.x * (rho - p.z) - p.y,
        p.x * p.y - beta * p.z
    );
}

void main() {
    vec3 p = vec3(0.1, 0.0, 0.0);
    float dt = 0.005;
    int steps = int(id + time_offset);
    
    for (int i = 0; i < steps; i++)
        p += lorenz(p) * dt;

    float scale = 0.03;
    gl_Position = vec4(p.x * scale, p.y * scale, 0.0, 1.0);
    gl_PointSize = 2.0;
    v_color = p.z;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in float v_color;
out vec4 fragColor;
void main() {
    float brightness = clamp((v_color + 30.0) / 60.0, 0.5, 1.0);
    fragColor = vec4(brightness, brightness, brightness, 1.0);
}
"""

def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Lorenz Attractor")

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

    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_PROGRAM_POINT_SIZE)

    time_offset_loc = glGetUniformLocation(shader, "time_offset")
    
    clock = pygame.time.Clock()
    running = True
    
    current_batch = 0
    points_drawn = 0
    
    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False

        if points_drawn >= POINTS_ITER:
            glClear(GL_COLOR_BUFFER_BIT)
            current_batch += 1
            points_drawn = 0
        
        glUniform1f(time_offset_loc, float(current_batch * POINTS_ITER))
        
        glDrawArrays(GL_POINTS, 0, POINTS_ITER)
        
        points_drawn += POINTS_ITER
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()