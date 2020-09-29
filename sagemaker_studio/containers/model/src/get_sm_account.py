import argparse
import sagemaker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str)
    parser.add_argument('--framework', type=str)
    args = parser.parse_args()
    
    registry = sagemaker.fw_registry.registry(
        region_name=args.region,
        framework=args.framework
    )
    account_id = registry.split('.')[0]
    print(account_id)
