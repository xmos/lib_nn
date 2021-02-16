# lib_nn Tests

## Fetching Test Dependencies

Run the following command to fetch all dependencies that are required to build the tests:

    ./fetch_dependencies.py

## Unit Tests

Run the following commands to build and execute the unit tests:

    cd unit_test
    make build
    xrun --xscope bin/xcore/unit_test.xe
