foreach(DEP IN ITEMS cargs TCLAP Stb X11 glm OpenGL GLEW Freetype SFML)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/Dependencies/${DEP}")
  include(${DEP})
endforeach()