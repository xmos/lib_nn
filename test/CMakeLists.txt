## Just adds more subdirs to the cmake build.

## Some simple local libraries used by unit test apps
add_subdirectory( shared )

## The unit test apps

add_subdirectory( unit_test  )
# add_subdirectory( gtests  )