cmake_minimum_required(VERSION 3.16)
project(HandmadeGPT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(Threads REQUIRED)

add_executable(HandmadeGPT
    src/main.cpp
    src/Corpus.h
    src/Model.h
    src/Model.cpp
    src/Trainer.h
    src/Trainer.cpp
    src/Canvas.h
    src/Canvas.cpp
    src/MainWindow.h
    src/MainWindow.cpp)
target_link_libraries(HandmadeGPT PRIVATE Qt6::Widgets Threads::Threads)

if(MSVC)
    target_compile_options(HandmadeGPT PRIVATE /W4 /fp:fast $<$<NOT:$<CONFIG:Debug>>:/O2>)
    set_target_properties(HandmadeGPT PROPERTIES WIN32_EXECUTABLE ON)
else()
    target_compile_options(HandmadeGPT PRIVATE -Wall -Wextra
        $<$<NOT:$<CONFIG:Debug>>:-O3 -ffast-math>)
endif()
