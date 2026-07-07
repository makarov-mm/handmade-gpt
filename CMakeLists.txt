cmake_minimum_required(VERSION 3.16)
project(HandmadeGPT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(HandmadeGPT main.cpp)
target_link_libraries(HandmadeGPT PRIVATE Qt6::Widgets)

if(MSVC)
    target_compile_options(HandmadeGPT PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:/O2>
        /fp:fast)
    set_target_properties(HandmadeGPT PROPERTIES WIN32_EXECUTABLE ON)
else()
    target_compile_options(HandmadeGPT PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:-O3 -ffast-math>)
endif()
