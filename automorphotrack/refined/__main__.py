"""Command-line access to the separately versioned particle pipeline."""
import argparse
import json
from pathlib import Path
from .pipeline import RefinedConfig, run_pipeline


def main():
    parser=argparse.ArgumentParser(description='AutoMorphoTrack 2.5 particle analysis')
    parser.add_argument('input',help='One TIFF movie')
    parser.add_argument('--out',required=True,help='New output directory')
    parser.add_argument('--config',help='Explicit JSON configuration')
    args=parser.parse_args()
    config=RefinedConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else RefinedConfig()
    result=run_pipeline(args.input,args.out,config)
    print(json.dumps({'status':'completed','source_sha256':result['manifest']['source_sha256'],
                      'observations':result['manifest']['observations']},indent=2))


if __name__=='__main__':main()
