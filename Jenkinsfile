@Library('xmos_jenkins_shared_library@v0.53.0') _

// Converts a Unity fixture-verbose log file into JUnit XML via lib_unity's
// parse_output.rb, writing "<suiteName>_results.xml" (relative to cwd), and
// publishes it via the junit step.
def UnityJunit(String logFile, String suiteName) {
    sh "ruby ${WORKSPACE}/lib_unity/lib_unity/Unity/auto/parse_output.rb -xml -suite${suiteName} ${logFile}"
    sh "mv report.xml ${suiteName}_results.xml"
    junit "${suiteName}_results.xml"
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
            defaultValue: 'smoke',
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

                        stage("Test") {
                            steps {
                                dir("${REPO}/test/unit_test") {
                                    sh "./bin/unit_test -v > NativeUnit.log"
                                    UnityJunit("NativeUnit.log", "NativeUnit")
                                }
                                dir("${REPO}/test/integration") {
                                    sh "./bin/integration_test -v > NativeIntegration.log"
                                    UnityJunit("NativeIntegration.log", "NativeIntegration")
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
                                    withTools(params.TOOLS_VERSION_XS) {sh "xsim --args bin/unit_test.xe -v > XS3.log"}
                                    UnityJunit("XS3.log", "XS3")
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
                                    withTools(params.TOOLS_VERSION_VX) {sh "xsim --args bin/unit_test.xe -v > VX4.log"}
                                    UnityJunit("VX4.log", "VX4")
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
