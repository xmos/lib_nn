@Library('xmos_jenkins_shared_library@v0.14.2') _

getApproval()

pipeline {
    agent {
        dockerfile {
            args "-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /home/jenkins/.ssh:/home/jenkins/.ssh:ro"
        }
    }

    parameters { // Available to modify on the job page within Jenkins if starting a build
        string( // use to try different tools versions
            name: 'TOOLS_VERSION',
            defaultValue: '15.0.6',
            description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
        )
    }

    options { // plenty of things could go here
        //buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    stages {
        stage("Setup") {
            // Clone and install build dependencies
            steps {
                // clean auto default checkout
                sh "rm -rf *"
                // clone
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: false,
                    extensions: [[$class: 'SubmoduleOption',
                                  threads: 8,
                                  timeout: 20,
                                  shallow: true,
                                  parentCredentials: true,
                                  recursiveSubmodules: true],
                                 [$class: 'CleanCheckout']],
                    userRemoteConfigs: [[credentialsId: 'xmos-bot',
                                         url: 'git@github.com:xmos/lib_nn']]
                ])
                // fetch dependencies
                sshagent (credentials: ['xmos-bot']) {
                    dir("${env.WORKSPACE}/test") {
                        sh "python fetch_dependencies.py"
                    }
                }
                // create venv
                sh "conda env create -q -p lib_nn_venv -f environment.yml"
                // install xmos tools version
                sh "/XMOS/get_tools.py " + params.TOOLS_VERSION
            }
        }
        stage("Update all packages") {
            // Roll all conda packages forward beyond their pinned versions
            when { expression { return params.UPDATE_ALL } }
            steps {
                sh "conda update --all -y -q -p lib_nn_venv"
            }
        }
        stage("Build") {
            steps {
                // below is how we can activate the tools, NOTE: xTIMEcomposer -> XTC at tools 15.0.5
                // sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv && // 
                sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv &&
                      . activate ./lib_nn_venv &&
                      cd test/unit_test && make all && make all PLATFORM=x86 MEMORY_SAFE=true"""
                sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv &&
                      . activate ./lib_nn_venv &&
                      cd test/gtests && ./build.sh && make all PLATFORM=x86"""
             }
         }
         stage("Test") {
             steps {
                 sh "cd test/unit_test && ./bin/x86/unit_test"
                 sh "cd test/gtests && ./bin/x86/unit_test"
            }
        }
    }
    post {
        cleanup {
            cleanWs()
        }
    }
}
