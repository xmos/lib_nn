@Library('xmos_jenkins_shared_library@v0.53.0') _

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
            defaultValue: '-j --repo arch_vx_slipgate -b master -a XTC 131',
            description: 'VX XTC tools version'
        )
        string(
            name: 'XMOSDOC_VERSION',
            defaultValue: 'v7.4.0',
            description: 'xmosdoc version'
        )
        string(
            name: 'INFR_APPS_VERSION',
            defaultValue: 'v3.1.1',
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
                                            cmakeOpts: '-DBUILD_NATIVE=ON'
                                        )
                                    }
                                }
                            }
                        } // Setup

                        stage("Test") {
                            steps {
                                dir("${REPO}/test/unit_test") {
                                    sh "./bin/unit_test"
                                }
                                dir("${REPO}/test/integration") {
                                    sh "./bin/integration_test"
                                }
                            }
                        } // Test

                    } // stages
                    post {cleanup {xcoreCleanSandbox()}}
                } // Native test

                stage("XS3 test") {
                    when {expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }}
                    agent {label "linux && x86_64"}

                    stages {
                        stage("Setup") {
                            steps {
                                dir(REPO) {
                                    checkoutScmShallow()
                                    dir("test/unit_test") {
                                        xcoreBuild(
                                            buildDir: 'build_xs3',
                                            toolsVersion: params.TOOLS_VERSION_XS,
                                            cmakeOpts: '-DAPP_HW_TARGET=XK-EVK-XU316',
                                        )
                                    }
                                }
                            }
                        } // Setup

                        stage("Test") {
                            steps {
                                dir("${REPO}/test/unit_test") {
                                    withTools(params.TOOLS_VERSION_XS) {sh "xsim --args bin/unit_test.xe"}
                                }
                            }
                        } // Test

                    } // stages
                    post {cleanup {xcoreCleanSandbox()}}
                } // XS3 test

                stage("VX4 test") {
                    when {expression { !env.GH_LABEL_DOC_ONLY.toBoolean() }}
                    agent {label "linux && x86_64"}

                    stages {
                        stage("Setup") {
                            steps {
                                dir(REPO) {
                                    checkoutScmShallow()
                                    dir("test/unit_test") {
                                        xcoreBuild(
                                            buildDir: 'build_vx4',
                                            toolsVersion: params.TOOLS_VERSION_VX,
                                            cmakeOpts: '-DAPP_HW_TARGET=XK-EVK-XU416',
                                        )
                                    }
                                }
                            }
                        } // Setup

                        stage("Test") {
                            steps {
                                dir("${REPO}/test/unit_test") {
                                    withTools(params.TOOLS_VERSION_VX) {sh "xsim --args bin/unit_test.xe"}
                                }
                            }
                        } // Test

                    } // stages
                    post {cleanup {xcoreCleanSandbox()}}
                } // VX4 test

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
