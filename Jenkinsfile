pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE_NAME = 'mlops-app'
        DOCKER_REGISTRY = 'my-docker-registry' // Replace with your registry e.g., index.docker.io/yourusername
        K8S_DEPLOYMENT_NAME = 'churn-api-deployment'
        K8S_NAMESPACE = 'default'
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo 'Pulling latest code from GitHub...'
                checkout scm
            }
        }

        // stage('Setup Environment') {
        //     steps {
        //         echo 'Installing Python dependencies for pipeline checks...'
        //         sh '''
        //             python3 -m venv venv
        //             . venv/bin/activate
        //             pip install -r requirements.txt
        //         '''
        //     }
        // }

        stage('Drift Detection & Auto-Retraining') {
            steps {
                echo 'Running Drift Detection. If drift is found, retraining will trigger automatically...'
                sh '''
                    . venv/bin/activate
                    # This script checks drift and retrains if necessary.
                    # It generates a pipeline_status.env file for Jenkins.
                    python3 scripts/pipeline_runner.py
                '''
            }
        }

        stage('Read Pipeline Status') {
            steps {
                script {
                    // Read the environment variables output by the Python script
                    def props = readProperties file: 'pipeline_status.env'
                    env.SHOULD_REBUILD = props['SHOULD_REBUILD']
                    env.NEW_VERSION = props['NEW_VERSION'] ?: 'latest'
                }
            }
        }

        stage('Build Docker Image') {
            when {
                environment name: 'SHOULD_REBUILD', value: 'true'
            }
            steps {
                echo "Rebuilding Docker image with new model v${env.NEW_VERSION}..."
                script {
                    def customImage = docker.build("${DOCKER_REGISTRY}/${DOCKER_IMAGE_NAME}:v${env.NEW_VERSION}")
                    env.BUILT_IMAGE = "${DOCKER_REGISTRY}/${DOCKER_IMAGE_NAME}:v${env.NEW_VERSION}"
                }
            }
        }

        stage('Push Docker Image') {
            when {
                environment name: 'SHOULD_REBUILD', value: 'true'
            }
            steps {
                echo 'Pushing Docker image to registry...'
                script {
                    // Requires Jenkins credentials set up for Docker Hub
                    docker.withRegistry('', 'docker-hub-credentials-id') {
                        docker.image(env.BUILT_IMAGE).push()
                        docker.image(env.BUILT_IMAGE).push('latest')
                    }
                }
            }
        }

        stage('Deploy to Kubernetes') {
            when {
                environment name: 'SHOULD_REBUILD', value: 'true'
            }
            steps {
                echo "Redeploying to Kubernetes with image: ${env.BUILT_IMAGE}"
                // Requires Jenkins to have kubectl access configured
                sh """
                    kubectl set image deployment/${K8S_DEPLOYMENT_NAME} api=${env.BUILT_IMAGE} -n ${K8S_NAMESPACE}
                    kubectl rollout status deployment/${K8S_DEPLOYMENT_NAME} -n ${K8S_NAMESPACE}
                """
            }
        }
    }
    
    post {
        always {
            echo 'Pipeline finished. Cleaning up workspace...'
            cleanWs()
        }
        success {
            echo 'Pipeline executed successfully.'
        }
        failure {
            echo 'Pipeline failed! Please check the logs.'
        }
    }
}
