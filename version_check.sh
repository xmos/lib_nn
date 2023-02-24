#!/bin/bash
printf "\nRunning version check for lib_nn..."

# in lib_nn/lib_nn folder
TAG=$(git describe --tags --abbrev=0)
GIT_VERSION=$(printf ${TAG} | sed 's/v//')

printf "\nGit version = "$GIT_VERSION

function get_version()
{
    local filename=$1
    MAJOR=$(grep 'major' $filename | awk '{print $6}' | sed 's/;//')
    MINOR=$(grep 'minor' $filename | awk '{print $6}' | sed 's/;//')
    PATCH=$(grep 'patch' $filename | awk '{print $6}' | sed 's/;//')
    printf "$MAJOR.$MINOR.$PATCH"
}

VERSION_H="api/version.h"

VERSION_H_STR=$(get_version $VERSION_H)
printf "\nVersion header = "$VERSION_H_STR

if [ "$GIT_VERSION" != "$VERSION_H_STR" ]
then printf "\nVersion mismatch!" && exit 1
fi

MODULE_BUILD_INFO="module_build_info"
MODULE_BUILD_INFO_STR=$(grep 'VERSION' $MODULE_BUILD_INFO | awk '{print $3}')

printf "\nModule build info version = "$MODULE_BUILD_INFO_STR

if [ "$VERSION_H_STR" != "$MODULE_BUILD_INFO_STR" ]
then printf "\nVersion mismatch!" && exit 1
fi

printf "\n"
exit 0
