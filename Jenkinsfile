@Library('xmos_jenkins_shared_library@v0.53.0') _

getApproval()
pipeline {
    agent none
    environment {
        REPO = "lib_nn"
    }

    parameters {
        string(
            name: 'TOOLS_VERSION',
            defaultValue: '15.3.1',
            description: 'XTC tools version'
        )
    }

    options {
        buildDiscarder(xmosDiscardBuildSettings(onlyArtifacts = false))
        timestamps()
        skipDefaultCheckout()
    }

    stages {
        stage("Build and test") {
            agent {
                label "linux && x86_64"
            }

            stages {
                stage("Setup") {
                    // Clone and install build dependencies
                    steps {
                        dir(REPO) {
                            checkoutScmShallow()
                            // fetch dependencies
                            sshagent (credentials: ['xmos-bot']) {
                                dir("test") {
                                    sh "python3 fetch_dependencies.py"
                                }
                            }
                        }
                    }
                } // Setup

                stage("Build") {
                    steps {
                        dir(REPO) {
                            withTools(params.TOOLS_VERSION) {
                                sh "cmake -B build_xs3a --toolchain etc/xs3a.cmake"
                                sh "make -C build_xs3a -j8"
                            }
                            sh "cmake -B build_native"
                            sh "make -C build_native -j8"
                            dir("test/gtests") {
                                sh "./build.sh"
                                sh "make all PLATFORM=x86"
                            }
                        }
                    }
                } // Build

                stage("Test") {
                    steps {
                        dir(REPO) { 
                           sh "./build_native/test/unit_test/unit_test"
                           sh "./test/gtests/bin/x86/unit_test"
                        }
                    }
                } // Test
            } // stages
            post {
                cleanup {
                    xcoreCleanSandbox()
                }
            }
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
