import csv
import os
import argparse
from datasets import Dataset


# Label mapping for different classification tasks
label_dict = {'AD': 0, 'CN': 1, 'EMCI': 0, 'LMCI': 0, 'MCI': 0, 'SMC': 0, 'Patient': 0}


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='AlzCLIP Dataset Creation')
    
    parser.add_argument('--snp_path', type=str, default='../AD_43SNP/AD_43SNP/', 
                       help='Path to SNP data directory')
    parser.add_argument('--img_path', type=str, default='../output_merged.csv',
                       help='Path to MRI features CSV file')
    parser.add_argument('--output_path', type=str, default='./snp_lbl_img_dataset_ova',
                       help='Output dataset path')
    parser.add_argument('--label_type', type=str, default='binary', choices=['binary', 'multiclass'],
                       help='Label type: binary (AD vs CN) or multiclass')
    
    return parser.parse_args()


def raw2dataset_ova(snp_path, img_path, output_path='./snp_lbl_img_dataset_ova', label_type='binary'):
    """
    Convert raw SNP and MRI data to AlzCLIP dataset format
    
    Args:
        snp_path: Path to directory containing SNP files (one per subject)
        img_path: Path to CSV file containing MRI features and labels
        output_path: Where to save the processed dataset
        label_type: 'binary' for AD vs CN, 'multiclass' for all categories
    """
    print('Processing raw data...')
    print(f'SNP data path: {snp_path}')
    print(f'MRI data path: {img_path}')
    print(f'Output path: {output_path}')
    
    # Check if paths exist
    if not os.path.exists(snp_path):
        print(f"Error: SNP path {snp_path} does not exist!")
        print("Please check your --snp_path argument")
        return False
        
    if not os.path.exists(img_path):
        print(f"Error: MRI path {img_path} does not exist!")
        print("Please check your --img_path argument")
        return False
    
    # Set up label dictionary based on type
    if label_type == 'binary':
        # Binary classification: AD vs CN (others mapped to AD)
        current_label_dict = {'AD': 1, 'CN': 0, 'EMCI': 1, 'LMCI': 1, 'MCI': 1, 'SMC': 1, 'Patient': 1}
        print("Using binary classification: AD (1) vs CN (0)")
    else:
        # Multi-class classification
        current_label_dict = {'AD': 0, 'CN': 1, 'EMCI': 2, 'LMCI': 3, 'MCI': 4, 'SMC': 5, 'Patient': 6}
        print("Using multi-class classification")
    
    # Process SNP files
    files = os.listdir(snp_path)
    snp_files = [f for f in files if f.endswith('.csv')]
    print(f'Found {len(snp_files)} SNP files')
    
    snp_dict = {}
    for file in snp_files:
        file_path = os.path.join(snp_path, file)
        try:
            with open(file_path, 'r') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader)
                snp = []
                for row in csv_reader:
                    if len(row) > 5:  # Ensure row has enough columns
                        snp.append(row[5])  # SNP data in column 5
                
                subject_id = file[:-4]  # Remove .csv extension
                snp_dict[subject_id] = snp
                
        except Exception as e:
            print(f"Warning: Could not process SNP file {file}: {e}")
            continue
    
    print(f'Successfully processed {len(snp_dict)} SNP files')
    
    # Process MRI and label data
    snp_all = []
    img_all = []
    lbl_all = []
    
    try:
        with open(img_path, 'r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            print(f"MRI CSV headers: {headers}")
            
            matched_count = 0
            total_count = 0
            
            for row in csv_reader:
                total_count += 1
                
                if len(row) < 13:  # Ensure row has enough columns
                    print(f"Warning: Row {total_count} has insufficient columns")
                    continue
                
                subject_id = row[1]  # Subject ID in column 1
                
                if subject_id in snp_dict:
                    group = row[2]      # Group label in column 2
                    img_features = row[12]  # MRI features in column 12
                    snp_features = snp_dict[subject_id]
                    
                    # Check if group label is valid
                    if group in current_label_dict:
                        snp_all.append(snp_features)
                        img_all.append(img_features)
                        lbl_all.append(current_label_dict[group])
                        matched_count += 1
                    else:
                        print(f"Warning: Unknown group label '{group}' for subject {subject_id}")
                
    except Exception as e:
        print(f"Error processing MRI file: {e}")
        return False
    
    print(f'Total MRI samples: {total_count}')
    print(f'Matched SNP+MRI samples: {matched_count}')
    
    if matched_count == 0:
        print("Error: No matched samples found!")
        print("Please check that:")
        print("1. Subject IDs match between SNP files and MRI CSV")
        print("2. File formats are correct")
        print("3. Paths point to the right data")
        return False
    
    # Create dataset dictionary
    dataset_dict = {
        'images': img_all,
        'labels': lbl_all,
        'snp': snp_all
    }
    
    # Create Dataset object and save
    try:
        _dataset = Dataset.from_dict(dataset_dict)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        _dataset.save_to_disk(output_path)
        print(f'✓ Dataset saved successfully to: {output_path}')
        print(f'✓ Total samples: {len(lbl_all)}')
        
        # Print label distribution
        unique_labels, counts = zip(*[(label, lbl_all.count(label)) for label in set(lbl_all)])
        print(f'✓ Label distribution: {dict(zip(unique_labels, counts))}')
        
        return True
        
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False


def test_dataset_creation():
    """Test dataset creation with mock data"""
    print("Testing dataset creation with mock data...")
    
    # Create temporary test directory
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    snp_dir = os.path.join(temp_dir, 'snp_data')
    os.makedirs(snp_dir)
    
    try:
        # Create mock SNP files
        for i in range(3):
            snp_file = os.path.join(snp_dir, f'subject_{i}.csv')
            with open(snp_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['col1', 'col2', 'col3', 'col4', 'col5', 'SNP'])
                for j in range(234):  # 234 SNP features
                    writer.writerow(['', '', '', '', '', str(j % 3)])  # Mock SNP data
        
        # Create mock MRI file
        mri_file = os.path.join(temp_dir, 'mri_data.csv')
        with open(mri_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['col0', 'subject_id', 'group'] + [f'col{i}' for i in range(3, 12)] + ['mri_features']
            writer.writerow(headers)
            
            groups = ['AD', 'CN', 'AD']
            for i in range(3):
                mri_features = ' '.join([str(j % 10) for j in range(128)])  # Mock MRI features
                row = [''] + [f'subject_{i}', groups[i]] + [''] * 9 + [mri_features]
                writer.writerow(row)
        
        # Test dataset creation
        output_dir = os.path.join(temp_dir, 'test_dataset')
        success = raw2dataset_ova(snp_dir, mri_file, output_dir)
        
        if success:
            print("✓ Mock dataset creation test passed!")
        else:
            print("✗ Mock dataset creation test failed!")
            
        return success
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main function"""
    args = get_args()
    
    print("AlzCLIP Dataset Creation")
    print("=" * 40)
    
    success = raw2dataset_ova(
        snp_path=args.snp_path,
        img_path=args.img_path, 
        output_path=args.output_path,
        label_type=args.label_type
    )
    
    if success:
        print("\n✓ Dataset creation completed successfully!")
        print(f"✓ Dataset saved to: {args.output_path}")
        print(f"✓ You can now run: python main.py --dataset_path {args.output_path}")
    else:
        print("\n✗ Dataset creation failed!")
        print("Please check the error messages above and try again.")


if __name__ == "__main__":
    # If no command line arguments provided, run test
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running test mode...")
        test_dataset_creation()
    else:
        main()
