set(MPM_TARGET "mpm-opencl")

add_executable(${MPM_TARGET} main.cpp)

target_link_libraries(${MPM_TARGET}
    PRIVATE
        OpenCL::Headers
        OpenCL::HeadersCpp
        OpenCL::OpenCL
        OpenCL::Utils
        OpenCL::UtilsCpp
        OpenCL::SDK
        OpenCL::SDKCpp
        glm::glm
)
target_compile_definitions(${MPM_TARGET}
    PRIVATE
        CL_HPP_MINIMUM_OPENCL_VERSION=300
        CL_HPP_ENABLE_EXCEPTIONS
        $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS> # TODO: remove
)
set_target_properties(${MPM_TARGET}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    FOLDER "${MPM_TARGET}"
)

# Copy dependent glsl and cl source code to bin
set(OPENCL_KERNELS kernel.cl)
set(OPENCL_SHADERS shader.vert shader.frag)
if (NOT TARGET ${MPM_TARGET}-device-code)
    set(DEVICE_CODE_OUTPUTS)
    foreach(DEVICE_CODE_SOURCE IN LISTS OPENCL_KERNELS OPENCL_SHADERS)
    # NOTE: if() and foreach() could be omitted if CMake ver > 3.20 (COMMAND and OUTPUT needs genexpr)
    if(CMAKE_CONFIGURATION_TYPES)
        foreach(CONFIG_TYPE IN LISTS CMAKE_CONFIGURATION_TYPES)
        add_custom_command(
            OUTPUT "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${CONFIG_TYPE}/${DEVICE_CODE_SOURCE}"
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy_if_different
                "${CMAKE_CURRENT_LIST_DIR}/${DEVICE_CODE_SOURCE}"
                "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${CONFIG_TYPE}/${DEVICE_CODE_SOURCE}"
            DEPENDS ${DEVICE_CODE_SOURCE}
            COMMENT "Copying ${DEVICE_CODE_SOURCE}"
        )
        list(APPEND DEVICE_CODE_OUTPUTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${CONFIG_TYPE}/${DEVICE_CODE_SOURCE}")
        endforeach()
    else()
        add_custom_command(
        OUTPUT "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${DEVICE_CODE_SOURCE}"
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy_if_different
                "${CMAKE_CURRENT_LIST_DIR}/${DEVICE_CODE_SOURCE}"
                "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${DEVICE_CODE_SOURCE}"
        DEPENDS ${DEVICE_CODE_SOURCE}
        COMMENT "Copying ${DEVICE_CODE_SOURCE}"
        )
        list(APPEND DEVICE_CODE_OUTPUTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${DEVICE_CODE_SOURCE}")
    endif()
    endforeach()
    add_custom_target(${MPM_TARGET}-device-code
        DEPENDS ${DEVICE_CODE_OUTPUTS}
    )
endif()
add_dependencies(${MPM_TARGET}
    ${MPM_TARGET}-device-code
)