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
            parallel {
                stage("Native test") {
                    when {
                        expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }
                    }
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
                                dir("${REPO}/test") {
                                    withTools(params.TOOLS_VERSION) {
                                        sh "cmake -B build_xs3a --toolchain etc/xs3a.cmake"
                                        sh "make -C build_xs3a -j8"
                                    }
                                    // sanitizers fail for now
                                    sh "cmake -B build_native -DENABLE_SANITIZERS=0"
                                    sh "make -C build_native -j8"
                                }
                            }
                        } // Build

                        stage("Test") {
                            steps {
                                dir("${REPO}/test") {
                                   sh "./build_native/unit_test/unit_test"
                                   sh "./build_native/gtests/gtests"
                                }
                            }
                        } // Test

                    } // stages
                    post {
                        cleanup {
                            xcoreCleanSandbox()
                        }
                    }
                } // Native test

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
                                        xcoreBuild(archiveBins: false)
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
