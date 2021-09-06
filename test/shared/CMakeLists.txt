set( LIB_NAME lib_nn_test_shared )

file( GLOB_RECURSE  SHARED_SOURCES_C   *.c    )
file( GLOB_RECURSE  SHARED_SOURCES_CPP *.cpp  )
file( GLOB_RECURSE  SHARED_SOURCES_XC  *.xc   )

list( APPEND  INCLUDE_DIRS  include
                            src             )

###########

## cmake doesn't recognize XC files. Tell it to treat them as C files
set_source_files_properties( ${SHARED_SOURCES_XC} PROPERTIES LANGUAGE C )

list( APPEND  SHARED_SOURCES        ${SHARED_SOURCES_C  }  )
list( APPEND  SHARED_SOURCES        ${SHARED_SOURCES_CPP}  )

list( APPEND  SHARED_SOURCES   ${SHARED_SOURCES_${CMAKE_SYSTEM_NAME}} )

## Add library called "test_shared"
add_library( ${LIB_NAME} ${SHARED_SOURCES} )

target_include_directories( ${LIB_NAME} PUBLIC ${INCLUDE_DIRS} )

target_link_libraries( ${LIB_NAME} lib_nn )
