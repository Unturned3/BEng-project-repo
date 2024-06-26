
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cassert>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "AppState.hpp"
#include "GUI.hpp"
#include "Image.hpp"
#include "PanoContainer.hpp"
#include "Pose.hpp"
#include "Shader.hpp"
#include "VertexBuffer.hpp"
#include "Window.hpp"
#include "cnpy.h"
#include "utils.hpp"

static void glErrorCallback_(GLenum source, GLenum type, GLuint id,
                             GLenum severity, GLsizei length, const GLchar* msg,
                             const void* userParam)
{
    LOG("OpenGL error:");
    LOG(msg);
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Need 4 args: pano_path, traj_path, out_path, start_frame."
                  << std::endl;
        return 1;
    }

    std::filesystem::path cwd = std::filesystem::current_path();

    std::filesystem::path exe_path = cwd / argv[0];
    exe_path = std::filesystem::absolute(exe_path);
    exe_path = std::filesystem::canonical(exe_path);

    std::filesystem::path proj_dir = exe_path.parent_path().parent_path();

    std::string pano_path = argv[1];
    std::string traj_path = argv[2];
    std::string out_path = argv[3];
    try {
        AppState::get().start_frame = std::stoi(argv[4]);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    AppState::get().poses = cnpy::npy_load(traj_path);

#ifdef USE_EGL
    HeadlessGLContext window(640, 480, "OpenGL Test");
#else
    InteractiveGLContext window(640, 480, "OpenGL Test");
#endif

    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    cv::VideoWriter videoWriter(out_path, fourcc, 30, {640, 480}, true);
    check(videoWriter.isOpened(), "Error opening cv::VideoWriter");

    PanoContainer pano;
    if (pano_path.substr(pano_path.length() - 4) == ".mp4") {
        pano = PanoContainer(cv::VideoCapture(pano_path, cv::CAP_FFMPEG));
    }
    else {
        pano = PanoContainer(Image(pano_path, false));
    }

    if (glewInit() != GLEW_OK) throw std::runtime_error("GLEW init failed.");

    if (GLEW_KHR_debug) glDebugMessageCallback(glErrorCallback_, nullptr);

#ifndef USE_EGL
    GUI gui(window);
#endif

    uint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = {0.2f, 0.2f, 0.2f, 1.0f};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pano.width, pano.height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, pano.data);

    // Rectangle (two triangles) covering the screen.
    constexpr int stride = 3;
    // clang-format off
    std::vector<float> vertices {
        -1, -1,  0,
         1,  1,  0,
        -1,  1,  0,
        -1, -1,  0,
         1, -1,  0,
         1,  1,  0,
    };
    // clang-format on

    VertexBuffer vb(vertices.data(),
                    static_cast<uint>(vertices.size() * sizeof(float)));

    Shader shader(proj_dir / "shaders/vertex.glsl",
                  proj_dir / "shaders/frag.glsl");
    shader.use();
    {
        // proportion of the missing sphere that's below the horizon
        // NOTE: feature not used for now.
        double m = 1;
        // if (argc >= 4) m = std::stod(argv[3]);
        double u = (double)pano.width / 2 / (double)pano.height;
        float v_n = (float)(u * M_1_PI);
        float v_b = (float)(u / 2 + (1 - u) * m);
        shader.setFloat("v_norm", v_n);
        shader.setFloat("v_bias", v_b);
    }

    uint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride * sizeof(float),
                          (void*)0);
    vb.bind();

    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

#ifndef USE_EGL
    // Desired FPS
    const double desiredFps = 30.0;
    const double desiredFrameTime = 1.0 / desiredFps;

    // Timing variables
    /*
    double lastTime = glfwGetTime();
    double currentTime = 0.0;
    double deltaTime = 0.0;
    */
#endif

    while (!window.shouldClose()) {
#ifndef USE_EGL
        /*
        currentTime = glfwGetTime();
        deltaTime = currentTime - lastTime;
        */
#endif

        AppState& s = AppState::get();

        if (s.poses.has_value()) {
            updatePose();
        }

#ifndef USE_EGL
        window.handleKeyDown();
#endif

        // Recalculate LoD, perspective, & view.
        /*  NOTE: this is actually not used since we our rendered FoV is
            usually less than 75 degrees. No aliasing effects are observed. */
        float fov_thresh = 75.0f;
        if (s.hfov <= fov_thresh)
            shader.setFloat("lod", 0.0f);
        else
            shader.setFloat(
                "lod", (s.hfov - fov_thresh) / (s.max_fov - fov_thresh) + 1.0f);

        float focal_len = static_cast<float>(window.width()) * 0.5f /
                          tanf(glm::radians(s.hfov * 0.5f));
        float vfov_radians = 2.0f * atanf(static_cast<float>(window.height()) *
                                          0.5f / focal_len);
        glm::mat4 M_proj =
            glm::perspective(vfov_radians, window.aspectRatio(), 0.1f, 2.0f);
        s.M_proj = M_proj;
        shader.setMat4("proj", M_proj);

        // NOTE: by convention, camera faces -Z, and Y is up, X is right.
        glm::mat4 M_view =
            glm::lookAt(glm::vec3(0), -glm::vec3(glm::column(s.M_rot, 2)),
                        glm::vec3(glm::column(s.M_rot, 1)));
        shader.setMat4("view", M_view);

        // Clear frame
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw projected panorama
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0,
                     static_cast<int>(vertices.size() / stride));

#ifndef USE_EGL
        if (s.drawUI) {
            gui.update();
            gui.render();
        }
#endif
        auto [w, h] = window.framebufferShape();
        cv::Mat mat(cv::Size(w, h), CV_8UC3);

        /*  NOTE: we tell OpenGL that everything is in RGB, while in reality
            the frames loaded by OpenCV is in BGR, and cv::VideoWriter expects
            BGR input. If OpenGL renders everything as RGB, the output frame
            in the window will have R-B swapped, but the encoded video will
            look correct.
        */
        {
            GLenum fmt = GL_RGB;
            if (!pano.isVideo) {
                /*  Unlike OpenCV, stbi_image loads images as RGB, so we need
                    to swap the channels to give cv::VideoWriter what it wants
                    i.e. BGR. */
                fmt = GL_BGR;
            }
            glReadPixels(0, 0, w, h, fmt, GL_UNSIGNED_BYTE, mat.data);
        }

        /*  On macOS, the FB shape might be larger than window shape due
            to the use of HiDPI displays, so we explicitly downscale the
            frame to the desired size.
        */
#if __APPLE__
        cv::resize(mat, mat, {640, 480});
#endif
        videoWriter.write(mat);

        window.swapBuffers();

        // Update frame as appropriate
        if (pano.isVideo) {
            pano.nextFrame();
            if (pano.frame.empty()) {
                break;
            }
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pano.width, pano.height,
                            GL_RGB, GL_UNSIGNED_BYTE, pano.data);
        }

        if (s.poses.has_value()) {
            s.pose_idx += 1;
            if (static_cast<size_t>(s.pose_idx) == s.poses.value().shape[0]) {
                break;
            }
        }
    }

    videoWriter.release();
    return 0;
}
