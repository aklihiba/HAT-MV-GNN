#!/usr/bin/env python3
"""
Test script to verify .mat data loading works correctly.
Run this first to check if your data files are properly formatted.
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mat_data_processor import MatDataProcessor

def test_data_loading():
    """Test loading both Amazon and Yelp datasets."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    datasets = [
        {'name': 'amazon', 'path': './data/amazon/'},
        {'name': 'yelp', 'path': './data/yelpChi/'}
    ]
    
    for dataset in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {dataset['name'].upper()} dataset")
        logger.info(f"{'='*50}")
        
        try:
            # Create processor
            processor = MatDataProcessor(dataset['name'])
            
            # Check if file exists
            expected_file = f"{dataset['name']}.mat"
            if dataset['name'] == 'yelp':
                expected_file = "yelpChi.mat"
            
            file_path = os.path.join(dataset['path'], expected_file)
            if not os.path.exists(file_path):
                logger.error(f"❌ File not found: {file_path}")
                continue
            else:
                logger.info(f"✅ Found file: {file_path}")
            
            # Load .mat data
            logger.info("Loading .mat file...")
            mat_data = processor.load_mat_data(dataset['path'])
            
            # Print basic info
            logger.info(f"Features shape: {mat_data['features'].shape}")
            logger.info(f"Labels shape: {mat_data['labels'].shape}")
            logger.info(f"Available relations: {list(mat_data['relations'].keys())}")
            logger.info(f"All keys in .mat file: {mat_data['mat_keys']}")
            
            # Convert to HAT-MV-GNN format
            logger.info("Converting to HAT-MV-GNN format...")
            graph_data = processor.convert_to_hat_mv_gnn_format(mat_data)
            
            logger.info(f"✅ Successfully processed {dataset['name']} dataset:")
            logger.info(f"   Nodes: {graph_data['node_features'].size(0)}")
            logger.info(f"   Node features: {graph_data['node_features'].size(1)}")
            logger.info(f"   Relations: {list(graph_data['edge_data'].keys())}")
            logger.info(f"   Total edges: {sum(edge_data[0].size(1) for edge_data in graph_data['edge_data'].values())}")
            logger.info(f"   Labels: {graph_data['labels'].unique()}")
            
            # Test train/test split
            logger.info("Creating train/test splits...")
            graph_data = processor.create_train_test_split(graph_data)
            
            logger.info(f"✅ Splits created successfully:")
            logger.info(f"   Train: {graph_data['train_mask'].sum()}")
            logger.info(f"   Val: {graph_data['val_mask'].sum()}")
            logger.info(f"   Test: {graph_data['test_mask'].sum()}")
            
            logger.info(f"🎉 {dataset['name'].upper()} dataset is ready for training!")
            
        except Exception as e:
            logger.error(f"❌ Error processing {dataset['name']} dataset: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info("Data loading test completed!")
    logger.info("If both datasets loaded successfully, you can now run:")
    logger.info("  python train_mat.py --config configs/amazon_mat_config.yaml")
    logger.info("  python train_mat.py --config configs/yelp_mat_config.yaml")
    logger.info(f"{'='*50}")

if __name__ == '__main__':
    test_data_loading()