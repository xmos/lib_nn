// This file relates to internal XMOS infrastructure and should be ignored by external users

@Library('xmos_jenkins_shared_library@feature/xcore-build-arch') _

// Runs a Unity test, publishes its JUnit report, and prints its log on failure.
def UnityJunit(String command, String suiteName, Boolean allowEmpty=false) {
    def logFile = "${suiteName}.log"
    def status = sh(
        returnStatus: true,
        script: "${command} > ${logFile} 2>&1"
    )
    if (status != 0) {
        sh "cat ${logFile}"
    }
    sh "ruby ${WORKSPACE}/lib_unity/lib_unity/Unity/auto/parse_output.rb -xml -suite${suiteName} ${logFile}"
    sh "mv report.xml ${suiteName}_results.xml"
    junit allowEmptyResults: allowEmpty, testResults: "${suiteName}_results.xml"
    if (status != 0) {
        error("${suiteName} tests failed")
    }
}

def CustomCmakeTest(String toolchain, String runCommand) {
    def toolchainOption = ''
    if (toolchain != 'native') {
        sh 'git clone git@github.com:xmos/xmos_cmake_toolchain.git --depth 1 --branch develop'
        toolchainOption = " --toolchain=xmos_cmake_toolchain/${toolchain}.cmake"
    }
    sh "cmake -B build${toolchainOption}"
    sh 'cmake --build build'
    sh runCommand
}

getApproval()
pipeline {
    agent none
    environment {
        REPO = "lib_nn"
    }

    parameters {
        string(
            name: 'TOOLS_VERSION_XS',
            defaultValue: '15.3.1',
            description: 'XS XTC tools version'
        )
        string(
            name: 'TOOLS_VERSION_VX',
            defaultValue: 'bob@develop',
            description: 'VX Clang tools version'
        )
        string( // temporary support
            name: 'TOOLS_VERSION_VX_XCC',
            defaultValue: 'slipgate@develop',
            description: 'VX XCC tools version'
        )
        string(
            name: 'XMOSDOC_VERSION',
            defaultValue: 'v8.1.2',
            description: 'xmosdoc version'
        )
        string(
            name: 'INFR_APPS_VERSION',
            defaultValue: 'v3.6.0',
            description: 'The infr_apps version'
        )
        choice(
            name: 'TEST_LEVEL', choices: ['smoke', 'default', 'extended'],
            description: 'The level of test coverage to run'
        )
    }

    options {
        buildDiscarder(xmosDiscardBuildSettings(onlyArtifacts = false))
        timestamps()
        skipDefaultCheckout()
    }

    stages {
        stage("Build and test") {
            parallel {
                stage("Native test") {
                    when {expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }}
                    agent {label "linux && x86_64"}

                    stages {
                        stage("Setup") {
                            steps {
                                dir(REPO) {
                                    checkoutScmShallow()
                                    dir("test") {
                                        xcoreBuild(
                                            buildDir: 'build_native',
                                            cmakeOpts: "-DBUILD_NATIVE=ON -DTEST_LEVEL=${params.TEST_LEVEL}"
                                        )
                                    }
                                }
                            }
                        } // Setup

                        stage("Unit test") {
                            steps {
                                dir("${REPO}/test/unit_test") {
                                    UnityJunit("./bin/unit_test -v", "NativeUnit")
                                }
                            }
                        } // Unit test

                        stage("Custom CMake test") {
                            steps {
                                dir("${REPO}/test/custom_cmake_build") {
                                    CustomCmakeTest('native', './bin/add_tensor')
                                }
                            }
                        } // Custom CMake test

                        stage("Integration test") {
                            steps {
                                dir("${REPO}/test/integration") {
                                    UnityJunit("./bin/integration_test -v", "NativeIntegration")
                                }
                            }
                        } // Integration test

                    } // stages
                    post {cleanup {xcoreCleanSandbox()}}
                } // Native test

                stage("XS3 test") {
                    when {expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }}
                    agent {label "linux && x86_64"}

                    steps {
                        script {
                            withTools(params.TOOLS_VERSION_XS) {
                                stage("Setup") {
                                    dir(REPO) {
                                        checkoutScmShallow()
                                        dir("test/unit_test") {
                                            xcoreBuild()
                                        }
                                    }
                                }

                                stage("Unit test") {
                                    dir("${REPO}/test/unit_test") {
                                        UnityJunit("xsim --args bin/unit_test.xe -v", "XS3")
                                    }
                                }

                                stage("Custom CMake test") {
                                    dir("${REPO}/test/custom_cmake_build") {
                                        CustomCmakeTest('xs3a', 'xsim bin/add_tensor.xe')
                                    }
                                }
                            }
                        }
                    }
                    post {cleanup {xcoreCleanSandbox()}}
                } // XS3 test

                stage("VX4 test") {
                    when {expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }}
                    agent {label "linux && x86_64"}

                    steps {
                        script {
                            withTools(params.TOOLS_VERSION_VX) {
                                stage("Setup") {
                                    dir(REPO) {
                                        checkoutScmShallow()
                                        dir("test/unit_test") {
                                            xcoreBuild()
                                        }
                                    }
                                }

                                stage("Unit test") {
                                    dir("${REPO}/test/unit_test") {
                                        UnityJunit("xsim --args bin/unit_test.xe -v", "VX4", true)
                                    }
                                }

                                stage("Custom CMake test") {
                                    dir("${REPO}/test/custom_cmake_build") {
                                        CustomCmakeTest('vx4_clang', 'xsim bin/add_tensor --config-file config.xml')
                                    }
                                }
                            }
                        }
                    }
                    post {cleanup {xcoreCleanSandbox()}}
                } // VX4 test

                stage("VX4 XCC test") {
                    when {expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }}
                    agent {label "linux && x86_64"}

                    steps {
                        script {
                            withTools(params.TOOLS_VERSION_VX_XCC) {
                                stage("Setup") {
                                    dir(REPO) {
                                        checkoutScmShallow()
                                        dir("test/unit_test") {
                                            xcoreBuild()
                                        }
                                    }
                                }

                                stage("Unit test") {
                                    dir("${REPO}/test/unit_test") {
                                        UnityJunit("xsim --args bin/unit_test.xe -v", "VX4XCC")
                                    }
                                }

                                stage("Custom CMake test") {
                                    dir("${REPO}/test/custom_cmake_build") {
                                        CustomCmakeTest('vx4_xcc', 'xsim bin/add_tensor.xe')
                                    }
                                }
                            }
                        }
                    }
                    post {cleanup {xcoreCleanSandbox()}}
                } // VX4 XCC test

                stage("Docs and lib checks") {
                    agent {
                        label "documentation"
                    }

                    stages {
                        stage("Examples build") {
                            steps {
                                dir(REPO) {
                                    checkoutScmShallow()
                                    dir("examples") {
                                        xcoreBuildAll()
                                    }
                                }
                            }
                        } // Examples build

                        stage('Repo checks') {
                            steps {
                                warnError("Repo checks failed")
                                {
                                    runRepoChecks("${WORKSPACE}/${REPO}")
                                }
                            }
                        } // Repo checks

                        stage('Doc build') {
                            steps {
                                dir(REPO) {
                                    buildDocs()
                                }
                            }
                        } // Doc build

                        stage("Archive lib") {
                            steps {
                                archiveSandbox(REPO)
                            }
                        } // Archive lib

                    } // stages


                    post {
                        cleanup {
                            xcoreCleanSandbox()
                        }
                    }
                } // Docs and lib checks

            } // parallel
        } // Build and test

        stage('🚀 Release') {
            when {
                expression { triggerRelease.isReleasable() }
            }
            steps {
                triggerRelease()
            }
        }
    } // stages
} // pipeline
