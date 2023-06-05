cmake_minimum_required(VERSION 3.10)
project(SimpleNN-C)

set(CMAKE_C_STANDARD 11)

add_executable(SimpleNN-C initialize_parameters.c main.c)
