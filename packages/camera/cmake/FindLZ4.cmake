find_path(LZ4_INCLUDE_DIR NAMES lz4.h PATHS /usr/local/include)

find_library(LZ4_LIBRARY NAMES lz4 PATHS /usr/local/lib)

if(LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
    set(LZ4_FOUND TRUE)
    set(LZ4_LIBRARIES ${LZ4_LIBRARY})
    set(LZ4_INCLUDE_DIRS ${LZ4_INCLUDE_DIR})

    # Create an imported target
    add_library(LZ4::LZ4 UNKNOWN IMPORTED)

    # Set the properties for the imported target
    set_target_properties(LZ4::LZ4 PROPERTIES
        IMPORTED_LOCATION ${LZ4_LIBRARIES}
        INTERFACE_INCLUDE_DIRECTORIES ${LZ4_INCLUDE_DIRS}
    )
else()
    set(LZ4_FOUND FALSE)
endif()

mark_as_advanced(LZ4_INCLUDE_DIR LZ4_LIBRARY)