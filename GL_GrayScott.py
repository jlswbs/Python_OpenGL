# Gray-Scott reaction diffusion system

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import os

W, H = 960, 540
Du = 0.0385
Dv = 0.008
F = 0.0069999984
K = 0.031000046
BoxSize = 32
SKIP_FRAMES = 6

VERT = """
#version 330 core
out vec2 uv;
vec2 verts[4] = vec2[](
    vec2(-1,-1), vec2(1,-1),
    vec2(-1,1), vec2(1,1)
);
void main() {
    gl_Position = vec4(verts[gl_VertexID],0,1);
    uv = verts[gl_VertexID]*0.5+0.5;
}
"""

FRAG = """
#version 330 core
in vec2 uv;
out vec4 frag;
uniform sampler2D state;
uniform vec2 px;
uniform float Du;
uniform float Dv;
uniform float F;
uniform float K;
void main() {
    vec4 c = texture(state, uv);
    float U = c.r;
    float V = c.g;
    vec4 n = texture(state, uv+vec2(0,px.y));
    vec4 s = texture(state, uv-vec2(0,px.y));
    vec4 e = texture(state, uv+vec2(px.x,0));
    vec4 w = texture(state, uv-vec2(px.x,0));
    float lapU = n.r+s.r+e.r+w.r-4.0*U;
    float lapV = n.g+s.g+e.g+w.g-4.0*V;
    for(int i=0;i<4;i++){
        float uvv = U*V*V;
        float dU = Du*lapU-uvv+F*(1.0-U);
        float dV = Dv*lapV+uvv-(F+K)*V;
        U += dU;
        V += dV;
    }
    frag = vec4(U,V,0,1);
}
"""

SHOW = """
#version 330 core
in vec2 uv;
out vec4 frag;
uniform sampler2D tex;
void main() {
    vec4 c = texture(tex, uv);
    frag = vec4(c.g,c.g,c.g,1.0);
}
"""

def make_tex(data):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,W,H,0,GL_RGBA,GL_FLOAT,data)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
    return tex

def reseed(state):
    cx,cy = W//2,H//2
    r=6
    noise = (np.random.rand(2*r,2*r)-0.5)*0.02
    state[cy-r:cy+r,cx-r:cx+r,1] += noise
    np.clip(state,0.0,1.0,out=state)

def tweak_params():
    global F,K
    F *= 1.0+np.random.uniform(-0.3,0.3)
    K *= 1.0+np.random.uniform(-0.3,0.3)
    print(f"F={F:.8f}  K={K:.8f}")

def save_frame(counter):
    if not os.path.exists("data"):
        os.makedirs("data")
    pixels = glReadPixels(0,0,W,H,GL_RGBA,GL_UNSIGNED_BYTE)
    surf = pygame.image.fromstring(pixels,(W,H),"RGBA",True)
    pygame.image.save(surf,f"data/frame_{counter:05d}.png")

def main():
    pygame.init()
    pygame.display.set_mode((W,H),DOUBLEBUF|OPENGL)
    pygame.display.set_caption("Gray–Scott GPU")

    sim = compileProgram(
        compileShader(VERT,GL_VERTEX_SHADER),
        compileShader(FRAG,GL_FRAGMENT_SHADER)
    )
    show = compileProgram(
        compileShader(VERT,GL_VERTEX_SHADER),
        compileShader(SHOW,GL_FRAGMENT_SHADER)
    )

    state = np.zeros((H,W,4),np.float32)
    state[...,0]=1.0
    state[H//2-BoxSize:H//2+BoxSize,W//2-BoxSize:W//2+BoxSize,1]=1.0

    texA = make_tex(state)
    texB = make_tex(None)

    fbo = glGenFramebuffers(1)
    glDisable(GL_DEPTH_TEST)

    clock = pygame.time.Clock()
    ping = True
    running = True
    save_frames = False
    frame_counter = 0
    skip_counter = 0

    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running=False
            if e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    running=False
                if e.key == K_r:
                    reseed(state)
                    tweak_params()
                    glBindTexture(GL_TEXTURE_2D, texA)
                    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,W,H,GL_RGBA,GL_FLOAT,state)
                    glBindTexture(GL_TEXTURE_2D, texB)
                    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,W,H,GL_RGBA,GL_FLOAT,state)
                if e.key == K_s:
                    save_frames = not save_frames
                    if save_frames:
                        frame_counter = 0
                        skip_counter = 0
                        glBindTexture(GL_TEXTURE_2D, texA)
                        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,W,H,GL_RGBA,GL_FLOAT,state)
                        glBindTexture(GL_TEXTURE_2D, texB)
                        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,W,H,GL_RGBA,GL_FLOAT,state)
                    print(f"Saving frames: {save_frames}")

        src,dst = (texA,texB) if ping else (texB,texA)

        glBindFramebuffer(GL_FRAMEBUFFER,fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,dst,0)
        glUseProgram(sim)
        glUniform2f(glGetUniformLocation(sim,"px"),1/W,1/H)
        glUniform1f(glGetUniformLocation(sim,"Du"),Du)
        glUniform1f(glGetUniformLocation(sim,"Dv"),Dv)
        glUniform1f(glGetUniformLocation(sim,"F"),F)
        glUniform1f(glGetUniformLocation(sim,"K"),K)
        glBindTexture(GL_TEXTURE_2D,src)
        glDrawArrays(GL_TRIANGLE_STRIP,0,4)

        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glUseProgram(show)
        glBindTexture(GL_TEXTURE_2D,dst)
        glDrawArrays(GL_TRIANGLE_STRIP,0,4)

        pygame.display.flip()

        if save_frames:
            if skip_counter % SKIP_FRAMES == 0:
                save_frame(frame_counter)
                frame_counter += 1
            skip_counter += 1

        ping = not ping
        clock.tick(60)

    pygame.quit()

if __name__=="__main__":
    main()