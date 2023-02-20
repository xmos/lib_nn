#!/bin/sh
echo "\nRunning version check for lib_nn..."

# in lib_nn folder
TAG=$(git describe --tags --abbrev=0)
GIT_VERSION=$(echo ${TAG} | sed 's/v//')

echo "Git version = "$GIT_VERSION

function get_version()
{
    local filename=$1
    MAJOR=$(grep 'major' $filename | awk '{print $6}' | sed 's/;//')
    MINOR=$(grep 'minor' $filename | awk '{print $6}' | sed 's/;//')
    PATCH=$(grep 'patch' $filename | awk '{print $6}' | sed 's/;//')
    echo "$MAJOR.$MINOR.$PATCH"
}

VERSION_H="lib_nn/api/version.h"

VERSION_H_STR=$(get_version $VERSION_H)
echo "Version header = "$VERSION_H_STR

if [ "$GIT_VERSION" != "$VERSION_H_STR" ]
then echo "Version mismatch!" && exit 1
fi

MODULE_BUILD_INFO="lib_nn/module_build_info"
MODULE_BUILD_INFO_STR=$(grep 'VERSION' $MODULE_BUILD_INFO | awk '{print $3}')

echo "Module build info version = "$MODULE_BUILD_INFO_STR

if [ "$VERSION_H_STR" != "$MODULE_BUILD_INFO_STR" ]
then echo "Version mismatch!" && exit 1
fi

exit 0
