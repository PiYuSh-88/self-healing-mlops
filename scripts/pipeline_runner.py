"""
Jenkins Pipeline Helper Script.

This script acts as the bridge between Jenkins and the MLOps Python modules.
It runs the drift detection logic directly. If drift is detected, it runs
the auto-retraining pipeline, which outputs new versioned model artifacts
into the `models/` folder.

Jenkins reads the exit code and the 'pipeline_status.env' file to decide
whether to rebuild the Docker image and deploy to Kubernetes.
"""

import sys
import logging
from app import drift_detector, retrainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Jenkins CI/CD Pipeline Check...")
    
    # 1. Run Drift Detection
    logger.info("\n--- Running Drift Detection ---")
    drift_result = drift_detector.run_drift_detection()
    
    status_file = "pipeline_status.env"
    
    if not drift_result["drift_detected"]:
        logger.info("\n✅ No drift detected. Pipeline stop.")
        with open(status_file, "w") as f:
            f.write("SHOULD_REBUILD=false\n")
        sys.exit(0)
        
    logger.info("\n🚨 Drift detected! Triggering Auto-Retraining...")
    
    # 2. Run Auto-Retraining
    retrain_result = retrainer.run_retraining()
    
    if not retrain_result["success"]:
        logger.error("\n❌ Retraining failed! See logs above.")
        with open(status_file, "w") as f:
            f.write("SHOULD_REBUILD=false\n")
        sys.exit(1)
        
    logger.info(f"\n🎉 Retraining successful! New Model Version: {retrain_result['version']}")
    
    # 3. Tell Jenkins to rebuild and redeploy
    with open(status_file, "w") as f:
        f.write("SHOULD_REBUILD=true\n")
        f.write(f"NEW_VERSION={retrain_result['version']}\n")
    
if __name__ == "__main__":
    main()
